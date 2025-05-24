namespace TarsEngine.FSharp.Core.CodeAnalysis

/// Module for generating code analysis reports
module ReportGenerator =
    open System
    open System.IO
    open Types
    open PatternDetector
    
    /// Generates a Markdown report for matches
    let generateMarkdownReport (report: Report) : string =
        let matchesByPattern = groupMatchesByPattern report.Matches
        let matchesByFile = groupMatchesByFile report.Matches
        
        let patternSection =
            matchesByPattern
            |> List.map (fun (pattern, matches) ->
                let matchList =
                    matches
                    |> List.map (fun m ->
                        sprintf "- %s:%d:%d: %s" 
                            (Path.GetFileName(m.FilePath)) 
                            m.LineNumber 
                            m.ColumnNumber 
                            m.Text)
                    |> String.concat "\n"
                
                sprintf "### Pattern: %s\n\n**Description**: %s\n\n**Severity**: %.2f\n\n**Category**: %s\n\n**Matches**:\n\n%s"
                    pattern.Name
                    pattern.Description
                    pattern.Severity
                    pattern.Category
                    matchList)
            |> String.concat "\n\n"
        
        let fileSection =
            matchesByFile
            |> List.map (fun (filePath, matches) ->
                let matchList =
                    matches
                    |> List.map (fun m ->
                        sprintf "- Line %d: %s (%s)" 
                            m.LineNumber 
                            m.Text 
                            m.Pattern.Name)
                    |> String.concat "\n"
                
                sprintf "### File: %s\n\n**Matches**:\n\n%s"
                    filePath
                    matchList)
            |> String.concat "\n\n"
        
        let transformationSection =
            report.Transformations
            |> List.map (fun (transformation, content) ->
                sprintf "### Transformation: %s\n\n**Description**: %s\n\n**Pattern**: %s\n\n**Replacement**: %s\n\n**Language**: %s"
                    transformation.Name
                    transformation.Description
                    transformation.Pattern
                    transformation.Replacement
                    transformation.Language)
            |> String.concat "\n\n"
        
        sprintf "# Code Analysis Report\n\n## Summary\n\n%s\n\n## Score\n\n%.2f\n\n## Patterns\n\n%s\n\n## Files\n\n%s\n\n## Transformations\n\n%s"
            report.Summary
            report.Score
            patternSection
            fileSection
            transformationSection
    
    /// Generates an HTML report for matches
    let generateHtmlReport (report: Report) : string =
        let matchesByPattern = groupMatchesByPattern report.Matches
        let matchesByFile = groupMatchesByFile report.Matches
        
        let patternSection =
            matchesByPattern
            |> List.map (fun (pattern, matches) ->
                let matchList =
                    matches
                    |> List.map (fun m ->
                        sprintf "<li>%s:%d:%d: %s</li>" 
                            (Path.GetFileName(m.FilePath)) 
                            m.LineNumber 
                            m.ColumnNumber 
                            m.Text)
                    |> String.concat "\n"
                
                sprintf "<h3>Pattern: %s</h3>\n<p><strong>Description</strong>: %s</p>\n<p><strong>Severity</strong>: %.2f</p>\n<p><strong>Category</strong>: %s</p>\n<p><strong>Matches</strong>:</p>\n<ul>\n%s\n</ul>"
                    pattern.Name
                    pattern.Description
                    pattern.Severity
                    pattern.Category
                    matchList)
            |> String.concat "\n\n"
        
        let fileSection =
            matchesByFile
            |> List.map (fun (filePath, matches) ->
                let matchList =
                    matches
                    |> List.map (fun m ->
                        sprintf "<li>Line %d: %s (%s)</li>" 
                            m.LineNumber 
                            m.Text 
                            m.Pattern.Name)
                    |> String.concat "\n"
                
                sprintf "<h3>File: %s</h3>\n<p><strong>Matches</strong>:</p>\n<ul>\n%s\n</ul>"
                    filePath
                    matchList)
            |> String.concat "\n\n"
        
        let transformationSection =
            report.Transformations
            |> List.map (fun (transformation, content) ->
                sprintf "<h3>Transformation: %s</h3>\n<p><strong>Description</strong>: %s</p>\n<p><strong>Pattern</strong>: %s</p>\n<p><strong>Replacement</strong>: %s</p>\n<p><strong>Language</strong>: %s</p>"
                    transformation.Name
                    transformation.Description
                    transformation.Pattern
                    transformation.Replacement
                    transformation.Language)
            |> String.concat "\n\n"
        
        sprintf "<!DOCTYPE html>\n<html>\n<head>\n<title>Code Analysis Report</title>\n<style>\nbody { font-family: Arial, sans-serif; margin: 20px; }\nh1 { color: #333; }\nh2 { color: #666; }\nh3 { color: #999; }\nul { list-style-type: disc; }\nli { margin-bottom: 5px; }\n</style>\n</head>\n<body>\n<h1>Code Analysis Report</h1>\n<h2>Summary</h2>\n<pre>%s</pre>\n<h2>Score</h2>\n<p>%.2f</p>\n<h2>Patterns</h2>\n%s\n<h2>Files</h2>\n%s\n<h2>Transformations</h2>\n%s\n</body>\n</html>"
            report.Summary
            report.Score
            patternSection
            fileSection
            transformationSection
    
    /// Generates a JSON report for matches
    let generateJsonReport (report: Report) : string =
        let matchesJson =
            report.Matches
            |> List.map (fun m ->
                sprintf """
                {
                    "pattern": {
                        "name": "%s",
                        "description": "%s",
                        "regex": "%s",
                        "severity": %.2f,
                        "category": "%s",
                        "language": "%s"
                    },
                    "text": "%s",
                    "lineNumber": %d,
                    "columnNumber": %d,
                    "filePath": "%s"
                }"""
                    m.Pattern.Name
                    m.Pattern.Description
                    m.Pattern.Regex
                    m.Pattern.Severity
                    m.Pattern.Category
                    m.Pattern.Language
                    m.Text
                    m.LineNumber
                    m.ColumnNumber
                    m.FilePath)
            |> String.concat ",\n"
        
        let transformationsJson =
            report.Transformations
            |> List.map (fun (t, _) ->
                sprintf """
                {
                    "name": "%s",
                    "description": "%s",
                    "pattern": "%s",
                    "replacement": "%s",
                    "language": "%s"
                }"""
                    t.Name
                    t.Description
                    t.Pattern
                    t.Replacement
                    t.Language)
            |> String.concat ",\n"
        
        sprintf """
        {
            "summary": "%s",
            "score": %.2f,
            "matches": [
                %s
            ],
            "transformations": [
                %s
            ]
        }"""
            (report.Summary.Replace("\n", "\\n").Replace("\"", "\\\""))
            report.Score
            matchesJson
            transformationsJson
    
    /// Saves a report to a file
    let saveReport (report: Report) (filePath: string) (format: string) : unit =
        try
            let content =
                match format.ToLower() with
                | "markdown" | "md" -> generateMarkdownReport report
                | "html" -> generateHtmlReport report
                | "json" -> generateJsonReport report
                | _ -> generateMarkdownReport report
            
            File.WriteAllText(filePath, content)
        with
        | ex -> 
            printfn "Error saving report to %s: %s" filePath ex.Message
