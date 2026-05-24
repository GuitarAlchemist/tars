namespace TarsEngine.FSharp.FLUX.Conversion

open System
open System.IO
open System.Text.RegularExpressions

/// Converts TARS metascripts (.trsx) to FLUX metascripts (.flux)
module TarsToFluxConverter =

    /// Conversion result
    type ConversionResult = {
        Success: bool
        FluxContent: string
        Errors: string list
        Warnings: string list
        OriginalFile: string
        OutputFile: string
    }

    /// Convert DESCRIBE block to META block
    let convertDescribeToMeta (content: string) =
        let pattern = @"DESCRIBE\s*\{([^}]*)\}"
        let replacement = "META {$1}"
        Regex.Replace(content, pattern, replacement, RegexOptions.Singleline)

    /// Convert CONFIG block to FLUX configuration
    let convertConfigBlock (content: string) =
        // CONFIG blocks in TARS become part of META in FLUX
        let pattern = @"CONFIG\s*\{([^}]*)\}"
        let configMatch = Regex.Match(content, pattern, RegexOptions.Singleline)

        if configMatch.Success then
            let configContent = configMatch.Groups.[1].Value
            let metaAddition = sprintf "\n    config: {%s\n    }" configContent
            let withoutConfig = Regex.Replace(content, pattern, "", RegexOptions.Singleline)

            // Add config to META block
            let metaPattern = @"(META\s*\{[^}]*)"
            let metaReplacement = sprintf "$1%s" metaAddition
            Regex.Replace(withoutConfig, metaPattern, metaReplacement, RegexOptions.Singleline)
        else
            content

    /// Convert VARIABLE blocks to FLUX variables
    let convertVariableBlocks (content: string) =
        let pattern = @"VARIABLE\s+(\w+)\s*\{([^}]*)\}"
        let replacement = "VARIABLE $1 {$2}"
        Regex.Replace(content, pattern, replacement, RegexOptions.Singleline)

    /// Convert ACTION blocks to appropriate FLUX blocks
    let convertActionBlocks (content: string) =
        let pattern = @"ACTION\s*\{([^}]*)\}"
        let matches = Regex.Matches(content, pattern, RegexOptions.Singleline)

        let mutable result = content
        for m in matches do
            let actionContent = m.Groups.[1].Value

            // Determine the type of action and convert accordingly
            if actionContent.Contains("type: \"log\"") then
                let logPattern = @"type:\s*""log""\s*message:\s*""([^""]*)""|message:\s*""([^""]*)""\s*type:\s*""log"""
                let logMatch = Regex.Match(actionContent, logPattern)
                if logMatch.Success then
                    let message = if logMatch.Groups.[1].Success then logMatch.Groups.[1].Value else logMatch.Groups.[2].Value
                    let fluxLog = sprintf "FSHARP {\n    printfn \"%s\"\n}" message
                    result <- result.Replace(m.Value, fluxLog)
            else
                // Convert to DIAGNOSTIC block for other actions
                let diagnostic = sprintf "DIAGNOSTIC {\n    action: %s\n}" actionContent
                result <- result.Replace(m.Value, diagnostic)

        result

    /// Convert IF/ELSE blocks to FSHARP conditional logic
    let convertConditionalBlocks (content: string) =
        let pattern = @"IF\s*\{[^}]*condition:\s*""([^""]*)""\s*([^}]*)\}\s*ELSE\s*\{([^}]*)\}"
        let replacement = """FSHARP {
    if $1 then
        $2
    else
        $3
}"""
        Regex.Replace(content, pattern, replacement, RegexOptions.Singleline)

    /// Convert FSHARP blocks (already compatible, just clean up)
    let convertFSharpBlocks (content: string) =
        // FSHARP blocks are mostly compatible, just ensure proper formatting
        let pattern = @"FSHARP\s*\{([^}]*output_variable:\s*""([^""]*)""\s*)\}"
        let replacement = """FSHARP {$1
    // Output stored in variable: $2
}"""
        Regex.Replace(content, pattern, replacement, RegexOptions.Singleline)

    /// Add FLUX-specific enhancements
    let addFluxEnhancements (content: string) (originalFileName: string) =
        let timestamp = DateTime.Now.ToString("yyyy-MM-dd")
        let enhancement = sprintf """

REASONING {
    This FLUX metascript was automatically converted from TARS format.
    Original file: %s
    Conversion date: %s

    The conversion process:
    1. DESCRIBE -> META block conversion
    2. CONFIG integration into META
    3. ACTION blocks -> FSHARP/DIAGNOSTIC blocks
    4. Enhanced with FLUX-specific capabilities

    This demonstrates the evolution from TARS to FLUX metascript system,
    providing improved multi-language support, agent orchestration,
    and advanced reasoning capabilities.
}""" originalFileName timestamp

        content + enhancement

    /// Convert a single TARS metascript to FLUX format
    let convertTarsToFlux (inputPath: string) (outputPath: string) : ConversionResult =
        try
            let originalContent = File.ReadAllText(inputPath)
            let fileName = Path.GetFileName(inputPath)

            let mutable fluxContent = originalContent
            let mutable warnings = []

            // Apply conversions in sequence
            fluxContent <- convertDescribeToMeta fluxContent
            fluxContent <- convertConfigBlock fluxContent
            fluxContent <- convertVariableBlocks fluxContent
            fluxContent <- convertActionBlocks fluxContent
            fluxContent <- convertConditionalBlocks fluxContent
            fluxContent <- convertFSharpBlocks fluxContent
            fluxContent <- addFluxEnhancements fluxContent fileName

            // Write the converted content
            File.WriteAllText(outputPath, fluxContent)

            {
                Success = true
                FluxContent = fluxContent
                Errors = []
                Warnings = warnings
                OriginalFile = inputPath
                OutputFile = outputPath
            }

        with
        | ex ->
            {
                Success = false
                FluxContent = ""
                Errors = [ex.Message]
                Warnings = []
                OriginalFile = inputPath
                OutputFile = outputPath
            }

    /// Convert all TARS metascripts in a directory to FLUX format
    let convertDirectoryTarsToFlux (inputDir: string) (outputDir: string) : ConversionResult list =
        if not (Directory.Exists(outputDir)) then
            Directory.CreateDirectory(outputDir) |> ignore

        Directory.GetFiles(inputDir, "*.trsx", SearchOption.AllDirectories)
        |> Array.map (fun inputFile ->
            let relativePath = Path.GetRelativePath(inputDir, inputFile)
            let outputFile = Path.Combine(outputDir, relativePath.Replace(".trsx", ".flux"))
            let outputFileDir = Path.GetDirectoryName(outputFile)

            if not (Directory.Exists(outputFileDir)) then
                Directory.CreateDirectory(outputFileDir) |> ignore

            convertTarsToFlux inputFile outputFile)
        |> Array.toList

    /// Generate conversion report
    let generateConversionReport (results: ConversionResult list) : string =
        let successful = results |> List.filter (fun r -> r.Success)
        let failed = results |> List.filter (fun r -> not r.Success)

        sprintf "# TARS to FLUX Conversion Report\nGenerated: %s\n\n## Summary\n- Total files processed: %d\n- Successfully converted: %d\n- Failed conversions: %d\n- Success rate: %.1f%%\n\n## Successful Conversions\n%s\n\n## Failed Conversions\n%s\n\n## Conversion Statistics\n- Average file size: %d characters\n- Total warnings: %d\n- FLUX enhancements added: %d\n\nThis conversion enables the TARS system to leverage the advanced\ncapabilities of the FLUX metascript language system."
            (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))
            results.Length
            successful.Length
            failed.Length
            (float successful.Length / float results.Length * 100.0)
            (successful |> List.map (fun r -> sprintf "✅ %s -> %s" r.OriginalFile r.OutputFile) |> String.concat "\n")
            (failed |> List.map (fun r -> sprintf "❌ %s: %s" r.OriginalFile (String.concat "; " r.Errors)) |> String.concat "\n")
            (successful |> List.map (fun r -> r.FluxContent.Length) |> List.average |> int)
            (results |> List.sumBy (fun r -> r.Warnings.Length))
            successful.Length
