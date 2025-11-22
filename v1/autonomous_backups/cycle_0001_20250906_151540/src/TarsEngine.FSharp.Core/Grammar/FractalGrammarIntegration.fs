namespace Tars.Engine.Grammar

open System
open System.IO
open System.Threading.Tasks
open Tars.Engine.Grammar.FractalGrammar
open Tars.Engine.Grammar.FractalGrammarParser

/// Integration of Fractal Grammars with TARS ecosystem
module FractalGrammarIntegration =

    /// Extended grammar source to include fractal grammars
    type ExtendedGrammarSource =
        | StandardGrammar of GrammarSource.GrammarSource
        | FractalGrammar of FractalGrammar
        | GeneratedFractal of FractalGenerationResult

    /// Fractal grammar execution context
    type FractalExecutionContext = {
        MaxIterations: int
        TimeoutMs: int
        MemoryLimitMB: int
        EnableVisualization: bool
        OutputFormat: FractalOutputFormat
        CacheResults: bool
        ParallelExecution: bool
    }

    and FractalOutputFormat =
        | EBNF
        | ANTLR
        | JSON
        | XML
        | GraphViz
        | SVG

    /// Fractal grammar execution result
    type FractalExecutionResult = {
        Success: bool
        GeneratedGrammar: string
        VisualizationData: string option
        ExecutionTime: TimeSpan
        MemoryUsed: int64
        IterationsCompleted: int
        FractalDimension: float
        ComplexityMetrics: Map<string, obj>
        ErrorMessages: string list
        Warnings: string list
    }

    /// Fractal grammar manager
    type FractalGrammarManager() =
        let engine = FractalGrammarEngine()
        let parser = FractalGrammarParser()
        let service = FractalGrammarService()
        
        /// Create default execution context
        member this.CreateDefaultContext() : FractalExecutionContext =
            {
                MaxIterations = 1000
                TimeoutMs = 30000
                MemoryLimitMB = 100
                EnableVisualization = true
                OutputFormat = EBNF
                CacheResults = true
                ParallelExecution = false
            }

        /// Execute fractal grammar with context
        member this.ExecuteFractalGrammar(fractalGrammar: FractalGrammar, context: FractalExecutionContext) : FractalExecutionResult =
            let startTime = DateTime.UtcNow
            let mutable memoryUsed = 0L
            let errors = ResizeArray<string>()
            let warnings = ResizeArray<string>()
            
            try
                // Generate fractal grammar
                let generationResult = engine.GenerateFractalGrammar(fractalGrammar)
                
                if not generationResult.Success then
                    errors.AddRange(generationResult.ErrorMessages)
                    warnings.AddRange(generationResult.Warnings)
                
                // Format output according to context
                let formattedGrammar = this.FormatGrammar(generationResult.GeneratedGrammar, context.OutputFormat)
                
                // Generate visualization if requested
                let visualizationData = 
                    if context.EnableVisualization then
                        Some (this.GenerateVisualization(generationResult.FractalTree, context.OutputFormat))
                    else
                        None
                
                // Calculate complexity metrics
                let complexityMetrics = engine.AnalyzeFractalComplexity(fractalGrammar)
                
                // Calculate fractal dimension
                let fractalDimension = this.CalculateFractalDimension(generationResult.FractalTree)
                
                memoryUsed <- generationResult.MemoryUsage
                
                {
                    Success = generationResult.Success
                    GeneratedGrammar = formattedGrammar
                    VisualizationData = visualizationData
                    ExecutionTime = DateTime.UtcNow - startTime
                    MemoryUsed = memoryUsed
                    IterationsCompleted = generationResult.IterationsPerformed
                    FractalDimension = fractalDimension
                    ComplexityMetrics = complexityMetrics
                    ErrorMessages = errors |> Seq.toList
                    Warnings = warnings |> Seq.toList
                }
            
            with
            | ex ->
                errors.Add(sprintf "Fractal execution failed: %s" ex.Message)
                {
                    Success = false
                    GeneratedGrammar = ""
                    VisualizationData = None
                    ExecutionTime = DateTime.UtcNow - startTime
                    MemoryUsed = memoryUsed
                    IterationsCompleted = 0
                    FractalDimension = 0.0
                    ComplexityMetrics = Map.empty
                    ErrorMessages = errors |> Seq.toList
                    Warnings = warnings |> Seq.toList
                }

        /// Format grammar according to output format
        member private this.FormatGrammar(grammar: string, format: FractalOutputFormat) : string =
            match format with
            | EBNF -> grammar
            | ANTLR -> this.ConvertToANTLR(grammar)
            | JSON -> this.ConvertToJSON(grammar)
            | XML -> this.ConvertToXML(grammar)
            | GraphViz -> this.ConvertToGraphViz(grammar)
            | SVG -> this.ConvertToSVG(grammar)

        /// Convert EBNF to ANTLR format
        member private this.ConvertToANTLR(ebnf: string) : string =
            ebnf
                .Replace("=", ":")
                .Replace(";", ";")
                .Replace(",", " ")
                .Replace("\"", "'")

        /// Convert grammar to JSON representation
        member private this.ConvertToJSON(grammar: string) : string =
            let lines = grammar.Split('\n') |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
            let rules = 
                lines
                |> Array.map (fun line ->
                    let parts = line.Split('=')
                    if parts.Length >= 2 then
                        sprintf "  \"%s\": \"%s\"" (parts.[0].Trim()) (parts.[1].Trim().TrimEnd(';'))
                    else
                        sprintf "  \"comment\": \"%s\"" (line.Trim()))
                |> String.concat ",\n"
            
            sprintf "{\n  \"fractal_grammar\": {\n%s\n  }\n}" rules

        /// Convert grammar to XML representation
        member private this.ConvertToXML(grammar: string) : string =
            let lines = grammar.Split('\n') |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
            let rules = 
                lines
                |> Array.map (fun line ->
                    let parts = line.Split('=')
                    if parts.Length >= 2 then
                        sprintf "    <rule name=\"%s\">%s</rule>" (parts.[0].Trim()) (parts.[1].Trim().TrimEnd(';'))
                    else
                        sprintf "    <!-- %s -->" (line.Trim()))
                |> String.concat "\n"
            
            sprintf "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<fractal_grammar>\n%s\n</fractal_grammar>" rules

        /// Convert grammar to GraphViz DOT format
        member private this.ConvertToGraphViz(grammar: string) : string =
            let header = "digraph FractalGrammar {\n  rankdir=TB;\n  node [shape=box];\n"
            let footer = "}"
            
            let lines = grammar.Split('\n') |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
            let edges = 
                lines
                |> Array.mapi (fun i line ->
                    let parts = line.Split('=')
                    if parts.Length >= 2 then
                        sprintf "  \"%s\" -> \"rule_%d\" [label=\"%s\"];" (parts.[0].Trim()) i (parts.[1].Trim().TrimEnd(';'))
                    else
                        sprintf "  \"comment_%d\" [label=\"%s\"];" i (line.Trim()))
                |> String.concat "\n"
            
            header + edges + "\n" + footer

        /// Convert grammar to SVG visualization
        member private this.ConvertToSVG(grammar: string) : string =
            let width = 800
            let height = 600
            let header = sprintf "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\">" width height
            let footer = "</svg>"
            
            let lines = grammar.Split('\n') |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
            let elements = 
                lines
                |> Array.mapi (fun i line ->
                    let y = 50 + i * 30
                    sprintf "  <text x=\"20\" y=\"%d\" font-family=\"monospace\" font-size=\"12\">%s</text>" y (line.Trim()))
                |> String.concat "\n"
            
            header + "\n" + elements + "\n" + footer

        /// Generate visualization data for fractal tree
        member private this.GenerateVisualization(tree: FractalNode, format: FractalOutputFormat) : string =
            match format with
            | GraphViz -> this.TreeToGraphViz(tree)
            | SVG -> this.TreeToSVG(tree)
            | JSON -> this.TreeToJSON(tree)
            | XML -> this.TreeToXML(tree)
            | _ -> sprintf "Visualization for %A not implemented" format

        /// Convert fractal tree to GraphViz
        member private this.TreeToGraphViz(tree: FractalNode) : string =
            let header = "digraph FractalTree {\n  rankdir=TB;\n  node [shape=ellipse];\n"
            let footer = "}"
            
            let rec generateNodes (node: FractalNode) =
                let nodeLabel = sprintf "\"%s\" [label=\"%s\\nL%d\"];" node.Id node.Name node.Level
                let childNodes = node.Children |> List.map generateNodes |> String.concat "\n"
                let edges = 
                    node.Children 
                    |> List.map (fun child -> sprintf "  \"%s\" -> \"%s\";" node.Id child.Id)
                    |> String.concat "\n"
                
                nodeLabel + "\n" + childNodes + "\n" + edges
            
            header + generateNodes tree + "\n" + footer

        /// Convert fractal tree to SVG
        member private this.TreeToSVG(tree: FractalNode) : string =
            let width = 1000
            let height = 800
            let header = sprintf "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\">" width height
            let footer = "</svg>"
            
            let rec generateSVGNodes (node: FractalNode) (x: float) (y: float) (level: int) =
                let nodeRadius = 20.0
                let levelHeight = 100.0
                let nodeSpacing = 150.0
                
                let circle = sprintf "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"lightblue\" stroke=\"black\"/>" x y nodeRadius
                let text = sprintf "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"10\">%s</text>" x (y + 3.0) node.Name
                
                let childElements = 
                    node.Children
                    |> List.mapi (fun i child ->
                        let childX = x + (float i - float node.Children.Length / 2.0) * nodeSpacing
                        let childY = y + levelHeight
                        let line = sprintf "  <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"black\"/>" x y childX childY
                        line + "\n" + generateSVGNodes child childX childY (level + 1))
                    |> String.concat "\n"
                
                circle + "\n" + text + "\n" + childElements
            
            header + "\n" + generateSVGNodes tree 500.0 50.0 0 + "\n" + footer

        /// Convert fractal tree to JSON
        member private this.TreeToJSON(tree: FractalNode) : string =
            let rec nodeToJSON (node: FractalNode) =
                let children = 
                    if node.Children.IsEmpty then "[]"
                    else
                        node.Children 
                        |> List.map nodeToJSON 
                        |> String.concat ", "
                        |> sprintf "[%s]"
                
                sprintf """{"id": "%s", "name": "%s", "level": %d, "pattern": "%s", "children": %s}""" 
                    node.Id node.Name node.Level node.Pattern children
            
            nodeToJSON tree

        /// Convert fractal tree to XML
        member private this.TreeToXML(tree: FractalNode) : string =
            let rec nodeToXML (node: FractalNode) (indent: string) =
                let children = 
                    if node.Children.IsEmpty then ""
                    else
                        node.Children 
                        |> List.map (fun child -> nodeToXML child (indent + "  "))
                        |> String.concat "\n"
                        |> sprintf "\n%s\n%s" children indent
                
                sprintf "%s<node id=\"%s\" name=\"%s\" level=\"%d\" pattern=\"%s\">%s</node>" 
                    indent node.Id node.Name node.Level node.Pattern children
            
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + nodeToXML tree ""

        /// Calculate fractal dimension of tree
        member private this.CalculateFractalDimension(tree: FractalNode) : float =
            let rec countNodesAtLevel (node: FractalNode) (targetLevel: int) =
                if node.Level = targetLevel then 1
                else if node.Level < targetLevel then
                    node.Children |> List.sumBy (fun child -> countNodesAtLevel child targetLevel)
                else 0
            
            let maxLevel = this.GetMaxLevel(tree)
            if maxLevel <= 1 then 1.0
            else
                let level1Count = float (countNodesAtLevel tree 1)
                let level2Count = float (countNodesAtLevel tree 2)
                
                if level1Count > 0.0 && level2Count > 0.0 then
                    Math.Log(level2Count) / Math.Log(level1Count)
                else
                    1.0

        /// Get maximum level in fractal tree
        member private this.GetMaxLevel(tree: FractalNode) : int =
            let rec getMaxLevel (node: FractalNode) =
                if node.Children.IsEmpty then node.Level
                else
                    node.Children |> List.map getMaxLevel |> List.max
            
            getMaxLevel tree

        /// Parse fractal grammar from file
        member this.ParseFractalGrammarFile(filePath: string) : FractalParseResult =
            if File.Exists(filePath) then
                let content = File.ReadAllText(filePath)
                parser.ParseFractalGrammar(content)
            else
                {
                    Success = false
                    FractalGrammar = None
                    ParsedRules = []
                    ErrorMessages = [sprintf "File not found: %s" filePath]
                    Warnings = []
                    ParseTime = TimeSpan.Zero
                }

        /// Save fractal grammar to file
        member this.SaveFractalGrammar(fractalGrammar: FractalGrammar, filePath: string, format: FractalOutputFormat) : bool =
            try
                let context = this.CreateDefaultContext()
                let result = this.ExecuteFractalGrammar(fractalGrammar, { context with OutputFormat = format })
                
                if result.Success then
                    File.WriteAllText(filePath, result.GeneratedGrammar)
                    true
                else
                    false
            with
            | _ -> false

    /// Fractal grammar CLI integration
    module FractalCLI =
        
        /// CLI command for fractal grammar operations
        type FractalCommand =
            | Generate of inputFile: string * outputFile: string * format: FractalOutputFormat
            | Parse of inputFile: string
            | Visualize of inputFile: string * outputFile: string
            | Analyze of inputFile: string
            | Examples

        /// Execute fractal CLI command
        let executeFractalCommand (command: FractalCommand) =
            let manager = FractalGrammarManager()
            
            match command with
            | Generate (inputFile, outputFile, format) ->
                let parseResult = manager.ParseFractalGrammarFile(inputFile)
                if parseResult.Success then
                    match parseResult.FractalGrammar with
                    | Some fractalGrammar ->
                        let success = manager.SaveFractalGrammar(fractalGrammar, outputFile, format)
                        if success then
                            printfn "✅ Fractal grammar generated successfully: %s" outputFile
                        else
                            printfn "❌ Failed to generate fractal grammar"
                    | None ->
                        printfn "❌ No fractal grammar parsed"
                else
                    printfn "❌ Parse errors:"
                    parseResult.ErrorMessages |> List.iter (printfn "  %s")
            
            | Parse inputFile ->
                let parseResult = manager.ParseFractalGrammarFile(inputFile)
                if parseResult.Success then
                    printfn "✅ Fractal grammar parsed successfully"
                    printfn "📊 Rules found: %d" parseResult.ParsedRules.Length
                    printfn "⏱️  Parse time: %A" parseResult.ParseTime
                else
                    printfn "❌ Parse errors:"
                    parseResult.ErrorMessages |> List.iter (printfn "  %s")
            
            | Visualize (inputFile, outputFile) ->
                let parseResult = manager.ParseFractalGrammarFile(inputFile)
                if parseResult.Success then
                    match parseResult.FractalGrammar with
                    | Some fractalGrammar ->
                        let context = { manager.CreateDefaultContext() with OutputFormat = SVG }
                        let result = manager.ExecuteFractalGrammar(fractalGrammar, context)
                        
                        if result.Success then
                            match result.VisualizationData with
                            | Some svg ->
                                File.WriteAllText(outputFile, svg)
                                printfn "✅ Visualization saved: %s" outputFile
                            | None ->
                                printfn "❌ No visualization data generated"
                        else
                            printfn "❌ Execution failed"
                    | None ->
                        printfn "❌ No fractal grammar parsed"
                else
                    printfn "❌ Parse failed"
            
            | Analyze inputFile ->
                let parseResult = manager.ParseFractalGrammarFile(inputFile)
                if parseResult.Success then
                    match parseResult.FractalGrammar with
                    | Some fractalGrammar ->
                        let context = manager.CreateDefaultContext()
                        let result = manager.ExecuteFractalGrammar(fractalGrammar, context)
                        
                        printfn "📊 FRACTAL GRAMMAR ANALYSIS"
                        printfn "============================"
                        printfn "Success: %b" result.Success
                        printfn "Execution Time: %A" result.ExecutionTime
                        printfn "Memory Used: %d KB" (result.MemoryUsed / 1024L)
                        printfn "Iterations: %d" result.IterationsCompleted
                        printfn "Fractal Dimension: %.3f" result.FractalDimension
                        
                        printfn "\nComplexity Metrics:"
                        result.ComplexityMetrics |> Map.iter (fun key value ->
                            printfn "  %s: %A" key value)
                    | None ->
                        printfn "❌ No fractal grammar parsed"
                else
                    printfn "❌ Parse failed"
            
            | Examples ->
                printfn "🔄 FRACTAL GRAMMAR EXAMPLES"
                printfn "============================"
                printfn ""
                printfn "1. Sierpinski Triangle:"
                printfn "   fractal sierpinski {"
                printfn "     pattern = \"triangle\""
                printfn "     recursive = \"triangle triangle triangle\""
                printfn "     dimension = 1.585"
                printfn "     depth = 5"
                printfn "     transform scale 0.5"
                printfn "   }"
                printfn ""
                printfn "2. Koch Snowflake:"
                printfn "   fractal koch_curve {"
                printfn "     pattern = \"line\""
                printfn "     recursive = \"line turn60 line turn-120 line turn60 line\""
                printfn "     dimension = 1.261"
                printfn "     depth = 6"
                printfn "     transform scale 0.333"
                printfn "   }"
