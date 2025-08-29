namespace TarsEngine.FSharp.Core.Context

open System
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// Configuration for context compression
type CompressionConfig = {
    MaxCompressionRatio: float
    QualityThreshold: float
    PreservePatterns: string list
    CompressionTargets: string list
    TokenEstimator: string -> int
}

/// Extractive context compressor (LLMLingua-style)
type ExtractiveContextCompressor(config: CompressionConfig, logger: ILogger<ExtractiveContextCompressor>) =
    
    /// Patterns to always preserve (high importance)
    let preservePatterns = [
        @"```[\s\S]*?```"                    // Code blocks
        @"Exception:.*$"                     // Exception messages
        @"Error:.*$"                         // Error messages
        @"TODO:.*$"                          // TODO items
        @"FIXME:.*$"                         // FIXME items
        @"^\s*at .*$"                        // Stack traces
        @"^#.*$"                             // Headers
        @"\b\d+M\+\s+searches/second\b"     // Performance metrics
        @"\bCUDA\b.*\bperformance\b"        // CUDA performance
        @"\bFS0988\b"                        // F# warnings
        @"\b\d+%\s+coverage\b"              // Test coverage
        @"autonomous.*improvement"           // Autonomous improvements
        @"Agent\s+OS"                        // Agent OS references
    ]
    
    /// Patterns that indicate low-value content for compression
    let compressionTargets = [
        @"(?i)verbose.*log"
        @"(?i)debug.*trace"
        @"(?i)intermediate.*step"
        @"(?i)temporary.*result"
        @"(?i)processing.*details"
    ]
    
    /// Extract high-value content that should be preserved
    let extractPreservedContent (text: string) =
        let preservedMatches = ResizeArray<string>()
        
        for pattern in preservePatterns do
            let matches = Regex.Matches(text, pattern, RegexOptions.Multiline ||| RegexOptions.IgnoreCase)
            for m in matches do
                preservedMatches.Add(m.Value)
        
        preservedMatches |> List.ofSeq
    
    /// Check if text contains compression targets
    let isCompressionTarget (text: string) =
        compressionTargets
        |> List.exists (fun pattern -> 
            Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase))
    
    /// Remove low-value content patterns
    let removeVerboseContent (text: string) =
        let mutable cleaned = text
        
        // Remove verbose logging patterns
        cleaned <- Regex.Replace(cleaned, @"(?im)^(DEBUG|TRACE|VERBOSE):.*$", "")
        
        // Remove excessive whitespace
        cleaned <- Regex.Replace(cleaned, @"\n\s*\n\s*\n", "\n\n")
        
        // Remove instruction-like patterns that could be prompt injection
        cleaned <- Regex.Replace(cleaned, @"(?im)^(system:|assistant:|user:|#\s*instruction:).*$", "")
        
        cleaned.Trim()
    
    /// Compress text using extractive approach
    let compressText (text: string) (targetRatio: float) =
        // Extract content to preserve
        let preservedContent = extractPreservedContent text
        let preservedText = String.concat "\n" preservedContent
        
        // Clean the text
        let cleanedText = removeVerboseContent text
        
        // If preserved content is substantial, use it as base
        let baseText = 
            if preservedText.Length > text.Length / 3 then
                preservedText
            else
                cleanedText
        
        // Apply length-based compression if needed
        let finalText = 
            if float baseText.Length / float text.Length > targetRatio then
                baseText
            else
                // Truncate to target ratio, keeping beginning and end
                let targetLength = int (float text.Length * targetRatio)
                if baseText.Length <= targetLength then
                    baseText
                else
                    let keepStart = targetLength * 2 / 3
                    let keepEnd = targetLength - keepStart
                    let startPart = baseText.Substring(0, Math.Min(keepStart, baseText.Length))
                    let endPart = 
                        if baseText.Length > keepEnd then
                            baseText.Substring(baseText.Length - keepEnd)
                        else
                            ""
                    startPart + "\n...\n" + endPart
        
        finalText
    
    /// Estimate compression quality
    let estimateQuality (original: string) (compressed: string) =
        // Simple quality estimation based on preserved important patterns
        let originalImportant = extractPreservedContent original
        let compressedImportant = extractPreservedContent compressed
        
        if originalImportant.IsEmpty then
            0.8 // Default quality if no important patterns
        else
            let preservedRatio = float compressedImportant.Length / float originalImportant.Length
            Math.Min(1.0, preservedRatio + 0.2) // Boost base quality
    
    /// Compress a single context span
    let compressSpan (span: ContextSpan) =
        task {
            try
                // Check if this span should be compressed
                if not (isCompressionTarget span.Text) && span.Salience > 0.7 then
                    // High salience, non-target spans are not compressed
                    return span
                
                let compressedText = compressText span.Text config.MaxCompressionRatio
                let quality = estimateQuality span.Text compressedText
                
                if quality >= config.QualityThreshold then
                    let compressedSpan = {
                        span with
                            Text = compressedText
                            Tokens = config.TokenEstimator compressedText
                            Metadata = span.Metadata.Add("compressed", "true").Add("quality", quality.ToString("F2"))
                    }
                    
                    logger.LogDebug("Compressed span {SpanId}: {OriginalTokens} -> {CompressedTokens} tokens (quality: {Quality:F2})",
                        span.Id, span.Tokens, compressedSpan.Tokens, quality)
                    
                    return compressedSpan
                else
                    logger.LogDebug("Compression quality {Quality:F2} below threshold {Threshold:F2} for span {SpanId}",
                        quality, config.QualityThreshold, span.Id)
                    return span
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to compress span {SpanId}", span.Id)
                return span
        }
    
    interface IContextCompressor with
        
        member _.CompressSpans(spans) =
            task {
                logger.LogInformation("Starting compression of {SpanCount} spans", spans.Length)
                
                let! compressedSpans = 
                    spans
                    |> List.map compressSpan
                    |> Task.WhenAll
                
                let compressedSpansList = compressedSpans |> Array.toList
                
                // Calculate overall compression metrics
                let originalTokens = spans |> List.sumBy (fun s -> s.Tokens)
                let compressedTokens = compressedSpansList |> List.sumBy (fun s -> s.Tokens)
                let compressionRatio = 
                    if originalTokens > 0 then
                        float compressedTokens / float originalTokens
                    else
                        1.0
                
                let compressedCount = 
                    compressedSpansList 
                    |> List.filter (fun s -> s.Metadata.ContainsKey("compressed"))
                    |> List.length
                
                let averageQuality = 
                    compressedSpansList
                    |> List.choose (fun s -> 
                        match s.Metadata.TryFind("quality") with
                        | Some q -> Double.TryParse(q) |> function | (true, v) -> Some v | _ -> None
                        | None -> None)
                    |> function
                        | [] -> 1.0
                        | qualities -> List.average qualities
                
                let result = {
                    OriginalSpans = spans
                    CompressedSpans = compressedSpansList
                    CompressionRatio = compressionRatio
                    QualityEstimate = averageQuality
                    Notes = $"Compressed {compressedCount}/{spans.Length} spans, saved {originalTokens - compressedTokens} tokens"
                }
                
                logger.LogInformation("Compression completed: {Notes}", result.Notes)
                logger.LogInformation("Compression ratio: {Ratio:F2}, Quality: {Quality:F2}", 
                    compressionRatio, averageQuality)
                
                return result
            }
        
        member _.EstimateQuality(original, compressed) =
            estimateQuality original compressed
