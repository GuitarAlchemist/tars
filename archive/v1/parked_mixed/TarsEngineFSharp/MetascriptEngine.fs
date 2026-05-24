namespace TarsEngineFSharp

module MetascriptEngine =
    open System
    open System.IO
    open System.Text.RegularExpressions
    
    // A simple rule in our metascript language
    type MetaRule = {
        Name: string
        Pattern: string
        Replacement: string
        RequiredNamespaces: string list
    }
    
    // Parse a metascript file into rules
    let parseMetascript (scriptContent: string) : MetaRule list =
        let rulePattern = @"rule\s+(\w+)\s*{\s*match:\s*""(.*?)""\s*replace:\s*""(.*?)""\s*(?:requires:\s*""(.*?)"")?\s*}"
        
        Regex.Matches(scriptContent, rulePattern, RegexOptions.Singleline)
        |> Seq.cast<Match>
        |> Seq.map (fun m ->
            let name = m.Groups.[1].Value
            let pattern = m.Groups.[2].Value
            let replacement = m.Groups.[3].Value
            let requires = 
                if m.Groups.[4].Success then 
                    m.Groups.[4].Value.Split(';') |> Array.toList 
                else []
            
            { Name = name; Pattern = pattern; Replacement = replacement; RequiredNamespaces = requires })
        |> Seq.toList
    
    // Load rules from a metascript file
    let loadRules (filePath: string) : MetaRule list =
        if File.Exists(filePath) then
            let content = File.ReadAllText(filePath)
            parseMetascript content
        else
            []
    
    // Apply a rule to code
    let applyRule (rule: MetaRule) (code: string) : string =
        // Convert the rule pattern to a regex
        // This is a simplified version - a real implementation would be more sophisticated
        let regexPattern = 
            rule.Pattern
                .Replace("$", @"\$")  // Escape $ in the pattern
                .Replace("(", @"\(")  // Escape ( in the pattern
                .Replace(")", @"\)")  // Escape ) in the pattern
                .Replace("{", @"\{")  // Escape { in the pattern
                .Replace("}", @"\}")  // Escape } in the pattern
                .Replace(".", @"\.")  // Escape . in the pattern
                .Replace("$collection", @"(\w+)")  // Replace $collection with a capture group
                .Replace("$sum", @"(\w+)")  // Replace $sum with a capture group
                .Replace("$i", @"(\w+)")  // Replace $i with a capture group
        
        // Create the replacement pattern
        // Again, this is simplified
        let replacementPattern = 
            rule.Replacement
                .Replace("$collection", "$1")
                .Replace("$sum", "$2")
        
        // Apply the transformation
        Regex.Replace(code, regexPattern, replacementPattern)
    
    // Apply all rules to a file
    let applyRulesToFile (filePath: string) (rules: MetaRule list) : string =
        if File.Exists(filePath) then
            let originalCode = File.ReadAllText(filePath)
            
            // Apply each rule in sequence
            let transformedCode = 
                rules |> List.fold (fun code rule -> applyRule rule code) originalCode
            
            // Add any required namespaces
            let requiredNamespaces = 
                rules |> List.collect (fun r -> r.RequiredNamespaces) |> List.distinct
            
            // Simple namespace insertion (would be more sophisticated in reality)
            let codeWithNamespaces =
                if requiredNamespaces.Length > 0 then
                    let namespaceInsertions = 
                        requiredNamespaces 
                        |> List.map (fun ns -> $"using {ns};") 
                        |> String.concat Environment.NewLine
                    
                    if not (transformedCode.Contains(namespaceInsertions)) then
                        Regex.Replace(transformedCode, 
                                     "^(using .*?;.*?)$", 
                                     "$1" + Environment.NewLine + namespaceInsertions,
                                     RegexOptions.Multiline)
                    else
                        transformedCode
                else
                    transformedCode
            
            codeWithNamespaces
        else
            ""
