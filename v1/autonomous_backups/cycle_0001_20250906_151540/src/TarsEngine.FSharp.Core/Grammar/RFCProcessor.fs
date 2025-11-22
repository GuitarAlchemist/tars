namespace Tars.Engine.Grammar

open System
open System.IO
open System.Net.Http
open System.Text.RegularExpressions

/// RFC processing and grammar extraction
module RFCProcessor =
    
    /// RFC metadata
    type RFCMetadata = {
        RFCId: string
        Title: string
        Url: string
        Abstract: string option
        Authors: string list
        Date: DateTime option
        Status: string option
        Category: string option
    }
    
    /// Extracted BNF rule
    type BNFRule = {
        Name: string
        Definition: string
        LineNumber: int option
        Section: string option
    }
    
    /// RFC processing result
    type RFCProcessingResult = {
        Metadata: RFCMetadata
        ExtractedRules: BNFRule list
        RawContent: string
        ProcessingErrors: string list
    }
    
    let private httpClient = new HttpClient()
    
    /// Get standard RFC URL
    let getRFCUrl rfcId =
        let normalizedId = rfcId.ToLowerInvariant()
        if normalizedId.StartsWith("rfc") then
            sprintf "https://datatracker.ietf.org/doc/html/%s" normalizedId
        else
            sprintf "https://datatracker.ietf.org/doc/html/rfc%s" normalizedId
    
    /// Download RFC content
    let downloadRFC rfcId =
        async {
            try
                let url = getRFCUrl rfcId
                let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
                return Ok response
            with
            | ex -> return Error (sprintf "Failed to download RFC %s: %s" rfcId ex.Message)
        }
    
    /// Extract RFC metadata from content
    let extractMetadata rfcId content =
        let titlePattern = @"<title>([^<]+)</title>"
        let abstractPattern = @"<h2[^>]*>Abstract</h2>\s*<p[^>]*>([^<]+)</p>"
        let authorPattern = @"<meta name=""author"" content=""([^""]+)"">"
        
        let title = 
            match Regex.Match(content, titlePattern, RegexOptions.IgnoreCase) with
            | m when m.Success -> m.Groups.[1].Value.Trim()
            | _ -> sprintf "RFC %s" rfcId
        
        let abstractText =
            match Regex.Match(content, abstractPattern, RegexOptions.IgnoreCase ||| RegexOptions.Singleline) with
            | m when m.Success -> Some (m.Groups.[1].Value.Trim())
            | _ -> None
        
        let authors =
            Regex.Matches(content, authorPattern, RegexOptions.IgnoreCase)
            |> Seq.cast<Match>
            |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
            |> Seq.toList
        
        {
            RFCId = rfcId
            Title = title
            Url = getRFCUrl rfcId
            Abstract = abstractText
            Authors = authors
            Date = None
            Status = None
            Category = None
        }
    
    /// Extract BNF/ABNF rules from RFC content
    let extractBNFRules content =
        let rules = ResizeArray<BNFRule>()
        
        let patterns = [
            @"^\s*([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*([^;]+);?\s*$"
            @"^\s*([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*(.+)$"
            @"^\s*([a-zA-Z][a-zA-Z0-9_-]*)\s*::=\s*(.+)$"
        ]
        
        let lines = content.Split('\n')
        for i, line in Array.indexed lines do
            for pattern in patterns do
                let match' = Regex.Match(line.Trim(), pattern, RegexOptions.IgnoreCase)
                if match'.Success then
                    let ruleName = match'.Groups.[1].Value.Trim()
                    let definition = match'.Groups.[2].Value.Trim()
                    
                    if not (definition.Contains("http") || definition.Contains("www") || definition.Length > 200) then
                        rules.Add({
                            Name = ruleName
                            Definition = definition
                            LineNumber = Some (i + 1)
                            Section = None
                        })
        
        rules |> Seq.toList
    
    /// Process RFC and extract grammar rules
    let processRFC rfcId =
        async {
            match! downloadRFC rfcId with
            | Ok content ->
                let metadata = extractMetadata rfcId content
                let rules = extractBNFRules content
                
                return Ok {
                    Metadata = metadata
                    ExtractedRules = rules
                    RawContent = content
                    ProcessingErrors = []
                }
            | Error error ->
                return Error error
        }
    
    /// Filter rules by name patterns
    let filterRules patterns rules =
        let regexPatterns = patterns |> List.map (fun p -> Regex(p, RegexOptions.IgnoreCase))
        rules |> List.filter (fun rule ->
            regexPatterns |> List.exists (fun regex -> regex.IsMatch(rule.Name)))
    
    /// Convert BNF rules to EBNF format
    let convertToEBNF rules =
        rules |> List.map (fun rule ->
            let ebnfDefinition = 
                rule.Definition
                    .Replace("*", " , { ")
                    .Replace("+", " , ")
                    .Replace("?", " ]")
                    .Replace("[", " [ ")
                    .Replace("]", " ] ")
                    .Replace("(", " ( ")
                    .Replace(")", " ) ")
                    .Replace("|", " | ")
            
            sprintf "%s = %s ;" rule.Name ebnfDefinition
        )
    
    /// Generate TARS grammar from RFC rules
    let generateTARSGrammar rfcId rules =
        let metadata =
            "meta {\n" +
            sprintf "  name: \"%s_grammar\"\n" rfcId +
            "  version: \"v1.0\"\n" +
            "  source: \"rfc\"\n" +
            "  language: \"EBNF\"\n" +
            sprintf "  created: \"%s\"\n" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")) +
            sprintf "  description: \"Grammar extracted from %s\"\n" rfcId +
            "}"

        let ebnfRules = convertToEBNF rules
        let grammarContent = String.concat "\n" ebnfRules
        let indentedContent = grammarContent.Split('\n') |> Array.map (fun line -> "    " + line) |> String.concat "\n"

        metadata + "\n\n" +
        "grammar {\n" +
        "  LANG(\"EBNF\") {\n" +
        indentedContent + "\n" +
        "  }\n" +
        "}"
    
    /// Save RFC grammar to file
    let saveRFCGrammar rfcId rules =
        async {
            try
                let grammarContent = generateTARSGrammar rfcId rules
                let fileName = sprintf "%s_grammar.tars" rfcId
                let filePath = Path.Combine(".tars", "grammars", fileName)
                
                let directory = Path.GetDirectoryName(filePath)
                if not (Directory.Exists(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                
                File.WriteAllText(filePath, grammarContent)
                
                let entry = {
                    GrammarResolver.GrammarIndexEntry.Id = sprintf "%s_grammar" rfcId
                    File = fileName
                    Origin = "rfc"
                    Version = Some "v1.0"
                    Hash = None
                    LastModified = Some DateTime.Now
                }
                GrammarResolver.updateGrammarIndex entry
                
                return Ok filePath
            with
            | ex -> return Error (sprintf "Failed to save RFC grammar: %s" ex.Message)
        }
    
    /// Get well-known RFC grammars
    let getWellKnownRFCs () = [
        ("rfc3986", "URI Generic Syntax", ["URI"; "scheme"; "authority"; "path"; "query"; "fragment"])
        ("rfc5322", "Internet Message Format", ["addr-spec"; "local-part"; "domain"; "date-time"])
        ("rfc7230", "HTTP/1.1 Message Syntax", ["HTTP-message"; "request-line"; "status-line"; "header-field"])
        ("rfc3339", "Date and Time on the Internet", ["date-time"; "full-date"; "full-time"])
        ("rfc4627", "JSON", ["JSON-text"; "value"; "object"; "array"; "string"; "number"])
        ("rfc5234", "ABNF", ["rulelist"; "rule"; "elements"; "alternation"])
    ]
    
    /// List available RFC grammars
    let listRFCGrammars () =
        let grammarsDir = Path.Combine(".tars", "grammars")
        if Directory.Exists(grammarsDir) then
            Directory.GetFiles(grammarsDir, "*_grammar.tars")
            |> Array.map Path.GetFileNameWithoutExtension
            |> Array.filter (fun name -> name.Contains("rfc"))
            |> Array.toList
        else
            []
