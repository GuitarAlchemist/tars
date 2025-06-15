namespace TarsEngine

open System
open System.Collections.Generic
open System.Text.RegularExpressions
open System.IO

/// TARS Tokenizer - Real tokenization with BPE and SentencePiece-like functionality
module TarsTokenizer =
    
    // ============================================================================
    // TOKENIZER TYPES
    // ============================================================================
    
    type TokenizerConfig = {
        VocabSize: int
        MaxSequenceLength: int
        PadToken: string
        UnkToken: string
        BosToken: string  // Beginning of sequence
        EosToken: string  // End of sequence
        UseByteLevel: bool
        CaseSensitive: bool
    }
    
    type TokenizerVocab = {
        TokenToId: Dictionary<string, int>
        IdToToken: Dictionary<int, string>
        SpecialTokens: Set<string>
        MergeRules: (string * string * int)[] // (token1, token2, priority)
    }
    
    type TokenizationResult = {
        TokenIds: int[]
        Tokens: string[]
        AttentionMask: int[]
        OriginalText: string
        ProcessingTimeMs: float
    }
    
    // ============================================================================
    // BYTE-LEVEL BPE IMPLEMENTATION
    // ============================================================================
    
    module ByteLevelBPE =
        
        /// Convert text to byte-level representation
        let textToBytes (text: string) : string[] =
            System.Text.Encoding.UTF8.GetBytes(text)
            |> Array.map (fun b -> $"<{b:X2}>")
        
        /// Convert byte-level tokens back to text
        let bytesToText (byteTokens: string[]) : string =
            try
                let bytes = 
                    byteTokens
                    |> Array.choose (fun token ->
                        if token.StartsWith("<") && token.EndsWith(">") && token.Length = 4 then
                            let hexStr = token.Substring(1, 2)
                            match System.Byte.TryParse(hexStr, System.Globalization.NumberStyles.HexNumber, null) with
                            | true, b -> Some b
                            | false, _ -> None
                        else None)
                
                System.Text.Encoding.UTF8.GetString(bytes)
            with
            | _ -> String.concat "" byteTokens
        
        /// Get character frequency from text
        let getCharFrequency (text: string) : Dictionary<string, int> =
            let freq = Dictionary<string, int>()
            
            for char in text do
                let charStr = string char
                if freq.ContainsKey(charStr) then
                    freq.[charStr] <- freq.[charStr] + 1
                else
                    freq.[charStr] <- 1
            
            freq
        
        /// Get pair frequency from tokens
        let getPairFrequency (tokens: string[][]) : Dictionary<string * string, int> =
            let freq = Dictionary<string * string, int>()
            
            for tokenSeq in tokens do
                for i in 0..tokenSeq.Length-2 do
                    let pair = (tokenSeq.[i], tokenSeq.[i+1])
                    if freq.ContainsKey(pair) then
                        freq.[pair] <- freq.[pair] + 1
                    else
                        freq.[pair] <- 1
            
            freq
        
        /// Apply merge rule to token sequences
        let applyMerge (tokens: string[][]) (mergeRule: string * string) : string[][] =
            let (token1, token2) = mergeRule
            let newToken = token1 + token2
            
            tokens |> Array.map (fun tokenSeq ->
                let result = ResizeArray<string>()
                let mutable i = 0
                
                while i < tokenSeq.Length do
                    if i < tokenSeq.Length - 1 && tokenSeq.[i] = token1 && tokenSeq.[i+1] = token2 then
                        result.Add(newToken)
                        i <- i + 2
                    else
                        result.Add(tokenSeq.[i])
                        i <- i + 1
                
                result.ToArray()
            )
    
    // ============================================================================
    // TARS TOKENIZER IMPLEMENTATION
    // ============================================================================
    
    type TarsTokenizer(config: TokenizerConfig) =
        let mutable vocab: TokenizerVocab option = None
        let mutable isInitialized = false
        
        /// Initialize tokenizer with vocabulary
        member _.Initialize() = async {
            let tokenToId = Dictionary<string, int>()
            let idToToken = Dictionary<int, string>()
            let specialTokens = Set.ofList [config.PadToken; config.UnkToken; config.BosToken; config.EosToken]
            
            // Add special tokens first
            let mutable currentId = 0
            
            // Add special tokens
            for token in [config.PadToken; config.UnkToken; config.BosToken; config.EosToken] do
                tokenToId.[token] <- currentId
                idToToken.[currentId] <- token
                currentId <- currentId + 1
            
            // Add basic vocabulary (simplified - would be learned from training data)
            let basicVocab = [
                // Common English characters
                "a"; "b"; "c"; "d"; "e"; "f"; "g"; "h"; "i"; "j"; "k"; "l"; "m";
                "n"; "o"; "p"; "q"; "r"; "s"; "t"; "u"; "v"; "w"; "x"; "y"; "z";
                "A"; "B"; "C"; "D"; "E"; "F"; "G"; "H"; "I"; "J"; "K"; "L"; "M";
                "N"; "O"; "P"; "Q"; "R"; "S"; "T"; "U"; "V"; "W"; "X"; "Y"; "Z";
                
                // Numbers
                "0"; "1"; "2"; "3"; "4"; "5"; "6"; "7"; "8"; "9";
                
                // Common punctuation
                " "; "."; ","; "!"; "?"; ";"; ":"; "'"; "\""; "-"; "_"; "("; ")";
                "["; "]"; "{"; "}"; "/"; "\\"; "@"; "#"; "$"; "%"; "^"; "&"; "*";
                "+"; "="; "<"; ">"; "|"; "~"; "`";
                
                // Common subwords (simplified)
                "the"; "and"; "or"; "but"; "in"; "on"; "at"; "to"; "for"; "of";
                "with"; "by"; "from"; "up"; "about"; "into"; "through"; "during";
                "before"; "after"; "above"; "below"; "between"; "among"; "under";
                "over"; "inside"; "outside"; "within"; "without"; "toward"; "towards";
                
                // Programming-related tokens
                "def"; "class"; "function"; "var"; "let"; "const"; "if"; "else";
                "for"; "while"; "do"; "try"; "catch"; "finally"; "return"; "yield";
                "import"; "export"; "from"; "as"; "true"; "false"; "null"; "undefined";
                "this"; "self"; "super"; "new"; "delete"; "typeof"; "instanceof";
                
                // Common prefixes and suffixes
                "un"; "re"; "pre"; "dis"; "mis"; "over"; "under"; "out"; "up";
                "ing"; "ed"; "er"; "est"; "ly"; "tion"; "sion"; "ness"; "ment";
                "ful"; "less"; "able"; "ible"; "ous"; "ious"; "al"; "ial"; "ic";
            ]
            
            // Add basic vocabulary
            for token in basicVocab do
                if not (tokenToId.ContainsKey(token)) && currentId < config.VocabSize then
                    tokenToId.[token] <- currentId
                    idToToken.[currentId] <- token
                    currentId <- currentId + 1
            
            // Fill remaining vocabulary with byte-level tokens if enabled
            if config.UseByteLevel then
                for i in 0..255 do
                    let byteToken = $"<{i:X2}>"
                    if not (tokenToId.ContainsKey(byteToken)) && currentId < config.VocabSize then
                        tokenToId.[byteToken] <- currentId
                        idToToken.[currentId] <- byteToken
                        currentId <- currentId + 1
            
            // Create simple merge rules (would be learned from training data)
            let mergeRules = [|
                ("t", "h", 1000)    // "th"
                ("e", "r", 999)     // "er"  
                ("i", "n", 998)     // "in"
                ("o", "n", 997)     // "on"
                ("a", "n", 996)     // "an"
                ("r", "e", 995)     // "re"
                ("e", "d", 994)     // "ed"
                ("n", "d", 993)     // "nd"
                ("o", "r", 992)     // "or"
                ("e", "n", 991)     // "en"
            |]
            
            vocab <- Some {
                TokenToId = tokenToId
                IdToToken = idToToken
                SpecialTokens = specialTokens
                MergeRules = mergeRules
            }
            
            isInitialized <- true
            
            printfn $"✅ TARS Tokenizer initialized:"
            printfn $"   📊 Vocabulary size: {tokenToId.Count:N0}"
            printfn $"   🔤 Special tokens: {specialTokens.Count}"
            printfn $"   🔄 Merge rules: {mergeRules.Length}"
            printfn $"   📏 Max sequence length: {config.MaxSequenceLength}"
            printfn $"   🔠 Case sensitive: {config.CaseSensitive}"
            printfn $"   📱 Byte-level: {config.UseByteLevel}"
            
            return true
        }
        
        /// Tokenize text into token IDs
        member _.Tokenize(text: string) = async {
            if not isInitialized then
                failwith "Tokenizer not initialized. Call Initialize() first."
            
            let startTime = DateTime.UtcNow
            
            match vocab with
            | None -> failwith "Vocabulary not loaded"
            | Some v ->
                
                let processedText = 
                    if config.CaseSensitive then text
                    else text.ToLowerInvariant()
                
                // Pre-tokenization (split on whitespace and punctuation)
                let preTokens = this.PreTokenize(processedText)
                
                // Apply BPE merges
                let bpeTokens = this.ApplyBPE(preTokens, v)
                
                // Convert to token IDs
                let tokenIds = ResizeArray<int>()
                let tokens = ResizeArray<string>()
                
                // Add BOS token
                tokenIds.Add(v.TokenToId.[config.BosToken])
                tokens.Add(config.BosToken)
                
                // Add content tokens
                for token in bpeTokens do
                    if v.TokenToId.ContainsKey(token) then
                        tokenIds.Add(v.TokenToId.[token])
                        tokens.Add(token)
                    else
                        // Handle unknown tokens
                        if config.UseByteLevel then
                            // Convert to byte-level tokens
                            let byteTokens = ByteLevelBPE.textToBytes(token)
                            for byteToken in byteTokens do
                                if v.TokenToId.ContainsKey(byteToken) then
                                    tokenIds.Add(v.TokenToId.[byteToken])
                                    tokens.Add(byteToken)
                                else
                                    tokenIds.Add(v.TokenToId.[config.UnkToken])
                                    tokens.Add(config.UnkToken)
                        else
                            tokenIds.Add(v.TokenToId.[config.UnkToken])
                            tokens.Add(config.UnkToken)
                
                // Add EOS token
                tokenIds.Add(v.TokenToId.[config.EosToken])
                tokens.Add(config.EosToken)
                
                // Truncate or pad to max sequence length
                let finalTokenIds = Array.zeroCreate config.MaxSequenceLength
                let finalTokens = Array.create config.MaxSequenceLength config.PadToken
                let attentionMask = Array.zeroCreate config.MaxSequenceLength
                
                let actualLength = min tokenIds.Count config.MaxSequenceLength
                
                for i in 0..actualLength-1 do
                    finalTokenIds.[i] <- tokenIds.[i]
                    finalTokens.[i] <- tokens.[i]
                    attentionMask.[i] <- 1
                
                // Fill remaining with padding
                let padTokenId = v.TokenToId.[config.PadToken]
                for i in actualLength..config.MaxSequenceLength-1 do
                    finalTokenIds.[i] <- padTokenId
                    finalTokens.[i] <- config.PadToken
                    attentionMask.[i] <- 0
                
                let endTime = DateTime.UtcNow
                let processingTime = (endTime - startTime).TotalMilliseconds
                
                return {
                    TokenIds = finalTokenIds
                    Tokens = finalTokens
                    AttentionMask = attentionMask
                    OriginalText = text
                    ProcessingTimeMs = processingTime
                }
        
        /// Detokenize token IDs back to text
        member _.Detokenize(tokenIds: int[]) = async {
            match vocab with
            | None -> failwith "Vocabulary not loaded"
            | Some v ->
                
                let tokens = ResizeArray<string>()
                
                for tokenId in tokenIds do
                    if v.IdToToken.ContainsKey(tokenId) then
                        let token = v.IdToToken.[tokenId]
                        if not (v.SpecialTokens.Contains(token)) then
                            tokens.Add(token)
                
                let text = 
                    if config.UseByteLevel then
                        // Handle byte-level tokens
                        let byteTokens = tokens.ToArray()
                        ByteLevelBPE.bytesToText(byteTokens)
                    else
                        String.concat "" (tokens.ToArray())
                
                return text
        
        /// Pre-tokenization step
        member _.PreTokenize(text: string) : string[] =
            // Simple regex-based pre-tokenization
            let pattern = @"\w+|[^\w\s]"
            let regex = Regex(pattern)
            let matches = regex.Matches(text)
            
            [| for m in matches -> m.Value |]
        
        /// Apply BPE merges to tokens with improved algorithm
        member _.ApplyBPE(tokens: string[], vocab: TokenizerVocab) : string[] =
            let mutable currentTokens = tokens |> Array.map (fun t -> [| for c in t -> string c |])

            // Apply merge rules in priority order (higher priority first)
            let sortedMerges = vocab.MergeRules |> Array.sortByDescending (fun (_, _, priority) -> priority)

            // Apply merges iteratively until no more merges possible
            let mutable changed = true
            let mutable iterations = 0
            let maxIterations = 10 // Prevent infinite loops

            while changed && iterations < maxIterations do
                changed <- false
                iterations <- iterations + 1

                for (token1, token2, _) in sortedMerges do
                    let newTokens = ByteLevelBPE.applyMerge currentTokens (token1, token2)
                    if not (Array.forall2 (fun a b -> Array.forall2 (=) a b) currentTokens newTokens) then
                        currentTokens <- newTokens
                        changed <- true

            // Flatten token sequences and filter out empty tokens
            currentTokens
            |> Array.collect id
            |> Array.filter (fun token -> not (String.IsNullOrEmpty(token)))
        
        /// Get vocabulary information
        member _.GetVocabInfo() =
            match vocab with
            | None -> None
            | Some v -> Some {|
                VocabSize = v.TokenToId.Count
                SpecialTokens = v.SpecialTokens |> Set.toArray
                MergeRules = v.MergeRules.Length
                Config = config
            |}
        
        /// Get token ID for a specific token
        member _.GetTokenId(token: string) =
            match vocab with
            | None -> None
            | Some v -> 
                if v.TokenToId.ContainsKey(token) then
                    Some v.TokenToId.[token]
                else
                    None
        
        /// Get token for a specific ID
        member _.GetToken(tokenId: int) =
            match vocab with
            | None -> None
            | Some v ->
                if v.IdToToken.ContainsKey(tokenId) then
                    Some v.IdToToken.[tokenId]
                else
                    None
