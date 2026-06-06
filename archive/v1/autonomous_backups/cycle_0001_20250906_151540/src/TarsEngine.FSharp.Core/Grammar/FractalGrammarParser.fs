namespace Tars.Engine.Grammar

open System
open System.Text.RegularExpressions
open Tars.Engine.Grammar.FractalGrammar

/// Fractal Grammar Parser for TARS
/// Parses fractal grammar specifications and creates executable fractal grammars
module FractalGrammarParser =

    /// Fractal grammar parsing result
    type FractalParseResult = {
        Success: bool
        FractalGrammar: FractalGrammar option
        ParsedRules: FractalRule list
        ErrorMessages: string list
        Warnings: string list
        ParseTime: TimeSpan
    }

    /// Fractal grammar token types
    type FractalToken =
        | FractalKeyword of string
        | Identifier of string
        | Number of float
        | String of string
        | Operator of string
        | Delimiter of char
        | Whitespace
        | Comment of string
        | EOF

    /// Fractal grammar lexer
    type FractalLexer(input: string) =
        let mutable position = 0
        let mutable line = 1
        let mutable column = 1
        
        member this.CurrentChar =
            if position >= input.Length then '\000'
            else input.[position]
        
        member this.Advance() =
            if this.CurrentChar = '\n' then
                line <- line + 1
                column <- 1
            else
                column <- column + 1
            position <- position + 1
        
        member this.PeekChar(offset: int) =
            let pos = position + offset
            if pos >= input.Length then '\000'
            else input.[pos]
        
        member this.ReadWhile(predicate: char -> bool) =
            let start = position
            while predicate this.CurrentChar do
                this.Advance()
            input.Substring(start, position - start)
        
        member this.NextToken() : FractalToken =
            // Skip whitespace
            while Char.IsWhiteSpace(this.CurrentChar) && this.CurrentChar <> '\000' do
                this.Advance()
            
            match this.CurrentChar with
            | '\000' -> EOF
            | '/' when this.PeekChar(1) = '/' ->
                this.Advance(); this.Advance()
                let comment = this.ReadWhile(fun c -> c <> '\n')
                Comment comment
            | '(' | ')' | '{' | '}' | '[' | ']' | ';' | ',' | ':' as delim ->
                this.Advance()
                Delimiter delim
            | '+' | '-' | '*' | '/' | '=' | '<' | '>' | '!' as op ->
                this.Advance()
                Operator (string op)
            | '"' ->
                this.Advance()
                let str = this.ReadWhile(fun c -> c <> '"')
                this.Advance() // Skip closing quote
                String str
            | c when Char.IsDigit(c) ->
                let numStr = this.ReadWhile(fun c -> Char.IsDigit(c) || c = '.')
                Number (Double.Parse(numStr))
            | c when Char.IsLetter(c) || c = '_' ->
                let identifier = this.ReadWhile(fun c -> Char.IsLetterOrDigit(c) || c = '_')
                match identifier.ToLowerInvariant() with
                | "fractal" | "rule" | "transform" | "scale" | "rotate" | "compose" | "recursive" 
                | "dimension" | "depth" | "terminate" | "when" | "if" | "else" -> 
                    FractalKeyword identifier
                | _ -> Identifier identifier
            | _ ->
                this.Advance()
                this.NextToken()

    /// Fractal grammar parser
    type FractalGrammarParser() =
        
        /// Parse fractal grammar from text
        member this.ParseFractalGrammar(input: string) : FractalParseResult =
            let startTime = DateTime.UtcNow
            let errors = ResizeArray<string>()
            let warnings = ResizeArray<string>()
            
            try
                let lexer = FractalLexer(input)
                let rules = this.ParseFractalRules(lexer, errors, warnings)
                
                if errors.Count = 0 then
                    let fractalGrammar = this.BuildFractalGrammar(rules)
                    {
                        Success = true
                        FractalGrammar = Some fractalGrammar
                        ParsedRules = rules
                        ErrorMessages = []
                        Warnings = warnings |> Seq.toList
                        ParseTime = DateTime.UtcNow - startTime
                    }
                else
                    {
                        Success = false
                        FractalGrammar = None
                        ParsedRules = rules
                        ErrorMessages = errors |> Seq.toList
                        Warnings = warnings |> Seq.toList
                        ParseTime = DateTime.UtcNow - startTime
                    }
            with
            | ex ->
                errors.Add(sprintf "Parse error: %s" ex.Message)
                {
                    Success = false
                    FractalGrammar = None
                    ParsedRules = []
                    ErrorMessages = errors |> Seq.toList
                    Warnings = warnings |> Seq.toList
                    ParseTime = DateTime.UtcNow - startTime
                }

        /// Parse fractal rules from token stream
        member private this.ParseFractalRules(lexer: FractalLexer, errors: ResizeArray<string>, warnings: ResizeArray<string>) : FractalRule list =
            let rules = ResizeArray<FractalRule>()
            let mutable token = lexer.NextToken()
            
            while token <> EOF do
                match token with
                | FractalKeyword "fractal" ->
                    match this.ParseFractalRule(lexer, errors, warnings) with
                    | Some rule -> rules.Add(rule)
                    | None -> ()
                | Comment _ -> () // Skip comments
                | Whitespace -> () // Skip whitespace
                | _ -> 
                    warnings.Add(sprintf "Unexpected token: %A" token)
                
                token <- lexer.NextToken()
            
            rules |> Seq.toList

        /// Parse a single fractal rule
        member private this.ParseFractalRule(lexer: FractalLexer, errors: ResizeArray<string>, warnings: ResizeArray<string>) : FractalRule option =
            try
                // Parse: fractal rule_name { ... }
                let mutable token = lexer.NextToken()
                
                let ruleName = 
                    match token with
                    | Identifier name -> name
                    | _ -> 
                        errors.Add("Expected rule name after 'fractal'")
                        "unnamed_rule"
                
                token <- lexer.NextToken()
                if token <> Delimiter '{' then
                    errors.Add("Expected '{' after rule name")
                
                // Parse rule body
                let ruleBody = this.ParseRuleBody(lexer, errors, warnings)
                
                Some {
                    Name = ruleName
                    BasePattern = ruleBody.BasePattern
                    RecursivePattern = ruleBody.RecursivePattern
                    TerminationCondition = ruleBody.TerminationCondition
                    Transformations = ruleBody.Transformations
                    Properties = ruleBody.Properties
                    Dependencies = ruleBody.Dependencies
                }
            with
            | ex ->
                errors.Add(sprintf "Error parsing fractal rule: %s" ex.Message)
                None

        /// Parse rule body content
        member private this.ParseRuleBody(lexer: FractalLexer, errors: ResizeArray<string>, warnings: ResizeArray<string>) =
            let mutable basePattern = ""
            let mutable recursivePattern = None
            let mutable terminationCondition = "max_depth_5"
            let mutable transformations = []
            let mutable properties = FractalGrammarEngine().CreateDefaultProperties()
            let mutable dependencies = []
            
            let mutable token = lexer.NextToken()
            
            while token <> Delimiter '}' && token <> EOF do
                match token with
                | FractalKeyword "pattern" ->
                    token <- lexer.NextToken()
                    if token = Operator "=" then
                        token <- lexer.NextToken()
                        match token with
                        | String pattern -> basePattern <- pattern
                        | _ -> errors.Add("Expected string after 'pattern ='")
                
                | FractalKeyword "recursive" ->
                    token <- lexer.NextToken()
                    if token = Operator "=" then
                        token <- lexer.NextToken()
                        match token with
                        | String pattern -> recursivePattern <- Some pattern
                        | _ -> errors.Add("Expected string after 'recursive ='")
                
                | FractalKeyword "terminate" ->
                    token <- lexer.NextToken()
                    if token = Operator "=" then
                        token <- lexer.NextToken()
                        match token with
                        | String condition -> terminationCondition <- condition
                        | _ -> errors.Add("Expected string after 'terminate ='")
                
                | FractalKeyword "dimension" ->
                    token <- lexer.NextToken()
                    if token = Operator "=" then
                        token <- lexer.NextToken()
                        match token with
                        | Number dim -> properties <- { properties with Dimension = dim }
                        | _ -> errors.Add("Expected number after 'dimension ='")
                
                | FractalKeyword "depth" ->
                    token <- lexer.NextToken()
                    if token = Operator "=" then
                        token <- lexer.NextToken()
                        match token with
                        | Number depth -> properties <- { properties with RecursionLimit = int depth }
                        | _ -> errors.Add("Expected number after 'depth ='")
                
                | FractalKeyword "transform" ->
                    let transform = this.ParseTransformation(lexer, errors, warnings)
                    transformations <- transform :: transformations
                
                | Comment _ -> () // Skip comments
                | _ -> ()
                
                token <- lexer.NextToken()
            
            {|
                BasePattern = basePattern
                RecursivePattern = recursivePattern
                TerminationCondition = terminationCondition
                Transformations = List.rev transformations
                Properties = properties
                Dependencies = dependencies
            |}

        /// Parse transformation specification
        member private this.ParseTransformation(lexer: FractalLexer, errors: ResizeArray<string>, warnings: ResizeArray<string>) : FractalTransformation =
            let mutable token = lexer.NextToken()
            
            match token with
            | FractalKeyword "scale" ->
                token <- lexer.NextToken()
                match token with
                | Number factor -> Scale factor
                | _ -> 
                    errors.Add("Expected number after 'scale'")
                    Scale 1.0
            
            | FractalKeyword "rotate" ->
                token <- lexer.NextToken()
                match token with
                | Number angle -> Rotate angle
                | _ -> 
                    errors.Add("Expected number after 'rotate'")
                    Rotate 0.0
            
            | FractalKeyword "recursive" ->
                token <- lexer.NextToken()
                let depth = 
                    match token with
                    | Number d -> int d
                    | _ -> 3
                
                token <- lexer.NextToken()
                let innerTransform = this.ParseTransformation(lexer, errors, warnings)
                Recursive (depth, innerTransform)
            
            | FractalKeyword "compose" ->
                // Parse list of transformations
                let transforms = ResizeArray<FractalTransformation>()
                token <- lexer.NextToken()
                
                if token = Delimiter '[' then
                    token <- lexer.NextToken()
                    while token <> Delimiter ']' && token <> EOF do
                        let transform = this.ParseTransformation(lexer, errors, warnings)
                        transforms.Add(transform)
                        token <- lexer.NextToken()
                        if token = Delimiter ',' then
                            token <- lexer.NextToken()
                
                Compose (transforms |> Seq.toList)
            
            | _ ->
                warnings.Add(sprintf "Unknown transformation: %A" token)
                Scale 1.0

        /// Build fractal grammar from parsed rules
        member private this.BuildFractalGrammar(rules: FractalRule list) : FractalGrammar =
            let builder = FractalGrammarBuilder()
            
            builder
                .WithName("ParsedFractalGrammar")
                .WithVersion("1.0")
                .WithBaseGrammar(Grammar.createInline "base" "// Parsed fractal grammar")
            
            for rule in rules do
                builder.AddFractalRule(rule) |> ignore
            
            builder.Build()

    /// Fractal grammar DSL for easy specification
    module FractalDSL =
        
        /// Create a fractal grammar using F# computation expression
        type FractalGrammarBuilder() =
            member this.Yield(rule: FractalRule) = [rule]
            member this.Combine(rules1: FractalRule list, rules2: FractalRule list) = rules1 @ rules2
            member this.For(items: 'a seq, f: 'a -> FractalRule list) = 
                items |> Seq.collect f |> Seq.toList
            member this.Zero() = []
            
            member this.Run(rules: FractalRule list) =
                let builder = FractalGrammarBuilder()
                builder
                    .WithName("DSLFractalGrammar")
                    .WithVersion("1.0")
                    .WithBaseGrammar(Grammar.createInline "base" "// DSL fractal grammar")
                
                for rule in rules do
                    builder.AddFractalRule(rule) |> ignore
                
                builder.Build()
        
        let fractalGrammar = FractalGrammarBuilder()
        
        /// Create a simple fractal rule
        let fractalRule name pattern =
            {
                Name = name
                BasePattern = pattern
                RecursivePattern = Some (sprintf "(%s)*" pattern)
                TerminationCondition = "max_depth_5"
                Transformations = [Scale 0.8]
                Properties = FractalGrammarEngine().CreateDefaultProperties()
                Dependencies = []
            }
        
        /// Create a scaling transformation
        let scale factor = Scale factor
        
        /// Create a rotation transformation  
        let rotate angle = Rotate angle
        
        /// Create a recursive transformation
        let recursive depth transform = Recursive (depth, transform)
        
        /// Create a composition of transformations
        let compose transforms = Compose transforms

    /// Example fractal grammar specifications
    module Examples =
        
        /// Sierpinski Triangle Grammar
        let sierpinskiTriangle = 
            let rule = {
                Name = "sierpinski"
                BasePattern = "triangle"
                RecursivePattern = Some "triangle triangle triangle"
                TerminationCondition = "max_depth_5"
                Transformations = [
                    Scale 0.5
                    Recursive (5, Compose [Scale 0.5; Rotate 120.0])
                ]
                Properties = {
                    Dimension = 1.585  // log(3)/log(2)
                    ScalingFactor = 0.5
                    IterationDepth = 5
                    SelfSimilarityRatio = 0.333
                    RecursionLimit = 8
                    CompositionRules = ["triangle_subdivision"]
                }
                Dependencies = []
            }
            
            FractalGrammarBuilder()
                .WithName("SierpinskiTriangle")
                .WithVersion("1.0")
                .AddFractalRule(rule)
                .Build()
        
        /// Koch Snowflake Grammar
        let kochSnowflake =
            let rule = {
                Name = "koch_curve"
                BasePattern = "line"
                RecursivePattern = Some "line turn60 line turn-120 line turn60 line"
                TerminationCondition = "max_depth_6"
                Transformations = [
                    Scale (1.0/3.0)
                    Recursive (6, Scale (1.0/3.0))
                ]
                Properties = {
                    Dimension = 1.261  // log(4)/log(3)
                    ScalingFactor = 1.0/3.0
                    IterationDepth = 6
                    SelfSimilarityRatio = 0.25
                    RecursionLimit = 10
                    CompositionRules = ["line_subdivision"; "angle_preservation"]
                }
                Dependencies = []
            }
            
            FractalGrammarBuilder()
                .WithName("KochSnowflake")
                .WithVersion("1.0")
                .AddFractalRule(rule)
                .Build()
