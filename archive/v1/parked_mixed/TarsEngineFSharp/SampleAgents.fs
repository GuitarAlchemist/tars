namespace TarsEngineFSharp

module SampleAgents =
    open System
    open System.Collections.Generic
    open System.IO
    open System.Text.RegularExpressions
    open System.Threading.Tasks
    open Microsoft.CodeAnalysis
    open Microsoft.CodeAnalysis.CSharp
    open Microsoft.CodeAnalysis.CSharp.Syntax
    open TarsEngine.Interfaces

    // Define the types we need
    type CodeIssue(message: string, description: string, location: string) =
        member val Message = message with get, set
        member val Description = description with get, set
        member val Location = location with get, set

    type CodeFix(original: string, replacement: string, description: string) =
        member val Original = original with get, set
        member val Replacement = replacement with get, set
        member val Description = description with get, set

    type AnalysisResult(issues: IEnumerable<CodeIssue>, fixes: IEnumerable<CodeFix>) =
        member val Issues = issues with get, set
        member val Fixes = fixes with get, set

    // Define the agent role enum
    type AgentRole =
        | Analysis
        | Transformation
        | Validation

    // Define the interfaces
    type ICodeAgent =
        abstract member Name : string
        abstract member Description : string
        abstract member Role : AgentRole

    type IAnalysisAgent =
        inherit ICodeAgent
        abstract member AnalyzeAsync : string -> Task<AnalysisResult>

    type ITransformationAgent =
        inherit ICodeAgent
        abstract member TransformAsync : string -> Task<string>

    /// <summary>
    /// Agent that analyzes code for null reference risks
    /// </summary>
    type NullReferenceAnalysisAgent() =
        interface ICodeAgent with
            member _.Name = "NullReferenceAnalyzer"
            member _.Description = "Analyzes code for potential null reference exceptions"
            member _.Role = AgentRole.Analysis

        interface IAnalysisAgent with
            member _.AnalyzeAsync(code: string) =
                task {
                    // Parse the code
                    let tree = CSharpSyntaxTree.ParseText(code)
                    let root = tree.GetRoot()

                    let issues = new List<CodeIssue>()
                    let fixes = new List<CodeFix>()

                    // Find methods with reference type parameters
                    for node in root.DescendantNodes() do
                        match node with
                        | :? MethodDeclarationSyntax as method ->
                            // Check if method has parameters that could be null
                            let refParams =
                                method.ParameterList.Parameters
                                |> Seq.choose (fun p ->
                                    match p.Type with
                                    | :? IdentifierNameSyntax as typeName ->
                                        // Simple check for reference types
                                        let typeText = typeName.Identifier.Text
                                        if typeText = "string" ||
                                           typeText.EndsWith("[]") ||
                                           (typeText <> "int" &&
                                            typeText <> "double" &&
                                            typeText <> "bool" &&
                                            typeText <> "decimal" &&
                                            typeText <> "float") then
                                            Some p.Identifier.Text
                                        else
                                            None
                                    | _ -> None)
                                |> Seq.toList

                            // Check if method body has null checks for these parameters
                            if refParams.Length > 0 && method.Body <> null then
                                let hasNullChecks =
                                    method.Body.Statements
                                    |> Seq.exists (fun stmt ->
                                        match stmt with
                                        | :? IfStatementSyntax as ifStmt ->
                                            let condition = ifStmt.Condition.ToString()
                                            refParams |> List.exists (fun p ->
                                                condition.Contains($"{p} == null") ||
                                                condition.Contains($"{p} is null"))
                                        | _ -> false)

                                if not hasNullChecks then
                                    // Add an issue
                                    let location = $"Line {method.GetLocation().GetLineSpan().StartLinePosition.Line + 1}"
                                    let message = "Missing null check"
                                    let description = "Method " + method.Identifier.Text + " has reference type parameters " + String.Join(", ", refParams) + " without null checks"

                                    issues.Add(CodeIssue(message, description, location))

                                    // Add a fix
                                    let paramChecks =
                                        refParams
                                        |> List.map (fun p -> $"            if ({p} == null) throw new ArgumentNullException(nameof({p}));")
                                        |> String.concat Environment.NewLine

                                    let original = method.ToString()
                                    let bodyStart = method.Body.OpenBraceToken.Span.End
                                    let bodyStartPos = bodyStart - method.SpanStart + 1  // +1 for the newline after {

                                    let replacement =
                                        original.Substring(0, bodyStartPos) +
                                        Environment.NewLine +
                                        paramChecks +
                                        original.Substring(bodyStartPos)

                                    fixes.Add(CodeFix(original, replacement, "Add null checks for parameters: " + String.Join(", ", refParams)))
                        | _ -> ()

                    return AnalysisResult(issues, fixes)
                }

    /// <summary>
    /// Agent that analyzes code for inefficient loops
    /// </summary>
    type IneffectiveLoopAnalysisAgent() =
        interface ICodeAgent with
            member _.Name = "IneffectiveLoopAnalyzer"
            member _.Description = "Analyzes code for loops that could be replaced with LINQ"
            member _.Role = AgentRole.Analysis

        interface IAnalysisAgent with
            member _.AnalyzeAsync(code: string) =
                task {
                    // Parse the code
                    let tree = CSharpSyntaxTree.ParseText(code)
                    let root = tree.GetRoot()

                    let issues = new List<CodeIssue>()
                    let fixes = new List<CodeFix>()

                    // Find for loops that could be replaced with LINQ
                    for node in root.DescendantNodes() do
                        match node with
                        | :? ForStatementSyntax as forLoop ->
                            // Check if this is a loop that calculates a sum
                            match forLoop.Statement with
                            | :? BlockSyntax as block ->
                                // Look for sum += pattern
                                let hasSumPattern =
                                    block.Statements
                                    |> Seq.exists (fun stmt ->
                                        match stmt with
                                        | :? ExpressionStatementSyntax as exprStmt ->
                                            match exprStmt.Expression with
                                            | :? AssignmentExpressionSyntax as assignment ->
                                                assignment.Kind() = SyntaxKind.AddAssignmentExpression ||
                                                (assignment.Kind() = SyntaxKind.SimpleAssignmentExpression &&
                                                 assignment.Right.ToString().Contains("+"))
                                            | _ -> false
                                        | _ -> false)

                                if hasSumPattern then
                                    // Extract the collection and sum variable (simplified)
                                    let forLoopText = forLoop.ToString()
                                    let collectionMatch = Regex.Match(forLoopText, @"(\w+)\.Count")
                                    let sumMatch = Regex.Match(forLoopText, @"(\w+)\s*[+=]\s*")

                                    let collection = if collectionMatch.Success then collectionMatch.Groups.[1].Value else "collection"
                                    let sumVar = if sumMatch.Success then sumMatch.Groups.[1].Value else "sum"

                                    // Add an issue
                                    let location = $"Line {forLoop.GetLocation().GetLineSpan().StartLinePosition.Line + 1}"
                                    let message = "Inefficient loop"
                                    let description = "Manual loop to calculate sum could be replaced with LINQ"

                                    issues.Add(CodeIssue(message, description, location))

                                    // Add a fix
                                    let original = forLoop.ToString()
                                    let replacement = $"{sumVar} = {collection}.Sum();"

                                    fixes.Add(CodeFix(original, replacement, "Replace manual loop with LINQ Sum()"))
                            | _ -> ()
                        | _ -> ()

                    return AnalysisResult(issues, fixes)
                }

    /// <summary>
    /// Agent that transforms code by applying fixes from analysis
    /// </summary>
    type CodeTransformationAgent() =
        interface ICodeAgent with
            member _.Name = "CodeTransformer"
            member _.Description = "Transforms code by applying fixes from analysis"
            member _.Role = AgentRole.Transformation

        interface ITransformationAgent with
            member _.TransformAsync(code: string) =
                task {
                    // Parse the code
                    let tree = CSharpSyntaxTree.ParseText(code)
                    let root = tree.GetRoot()

                    // Run analysis to get fixes
                    let nullRefAgent = NullReferenceAnalysisAgent() :> IAnalysisAgent
                    let loopAgent = IneffectiveLoopAnalysisAgent() :> IAnalysisAgent

                    let! nullRefResult = nullRefAgent.AnalyzeAsync(code)
                    let! loopResult = loopAgent.AnalyzeAsync(code)

                    // Combine all fixes
                    let allFixes =
                        Seq.append
                            (nullRefResult.Fixes |> Seq.cast<CodeFix>)
                            (loopResult.Fixes |> Seq.cast<CodeFix>)
                        |> Seq.toList

                    // Apply fixes
                    let mutable transformedCode = code

                    for fix in allFixes do
                        if not (String.IsNullOrEmpty(fix.Original)) then
                            transformedCode <- transformedCode.Replace(fix.Original, fix.Replacement)

                    return transformedCode
                }

    /// <summary>
    /// Agent that transforms code to add null checks
    /// </summary>
    type NullCheckTransformationAgent() =
        interface ICodeAgent with
            member _.Name = "NullCheckTransformer"
            member _.Description = "Transforms code to add null checks"
            member _.Role = AgentRole.Transformation

        interface ITransformationAgent with
            member _.TransformAsync(code: string) =
                task {
                    // Parse the code
                    let tree = CSharpSyntaxTree.ParseText(code)
                    let root = tree.GetRoot()

                    // Find methods with reference type parameters
                    let methodsWithRefParams =
                        root.DescendantNodes()
                        |> Seq.choose (fun node ->
                            match node with
                            | :? MethodDeclarationSyntax as method ->
                                // Find reference type parameters
                                let refParams =
                                    method.ParameterList.Parameters
                                    |> Seq.choose (fun p ->
                                        match p.Type with
                                        | :? IdentifierNameSyntax as typeName ->
                                            // Simple check for reference types
                                            let typeText = typeName.Identifier.Text
                                            if typeText = "string" ||
                                               typeText.EndsWith("[]") ||
                                               (typeText <> "int" &&
                                                typeText <> "double" &&
                                                typeText <> "bool" &&
                                                typeText <> "decimal" &&
                                                typeText <> "float") then
                                                Some p.Identifier.Text
                                            else
                                                None
                                        | _ -> None)
                                    |> Seq.toList

                                // Check if method body has null checks for these parameters
                                if refParams.Length > 0 && method.Body <> null then
                                    let hasNullChecks =
                                        method.Body.Statements
                                        |> Seq.exists (fun stmt ->
                                            match stmt with
                                            | :? IfStatementSyntax as ifStmt ->
                                                let condition = ifStmt.Condition.ToString()
                                                refParams |> List.exists (fun p ->
                                                    condition.Contains($"{p} == null") ||
                                                    condition.Contains($"{p} is null"))
                                            | _ -> false)

                                    if not hasNullChecks then
                                        Some (method, refParams)
                                    else
                                        None
                                else
                                    None
                            | _ -> None)
                        |> Seq.toList

                    // Transform the code
                    let mutable transformedCode = code

                    for (method, refParams) in methodsWithRefParams do
                        let paramChecks =
                            refParams
                            |> List.map (fun p -> $"            if ({p} == null) throw new ArgumentNullException(nameof({p}));")
                            |> String.concat Environment.NewLine

                        let original = method.ToString()
                        let bodyStart = method.Body.OpenBraceToken.Span.End
                        let bodyStartPos = bodyStart - method.SpanStart + 1  // +1 for the newline after {

                        let replacement =
                            original.Substring(0, bodyStartPos) +
                            Environment.NewLine +
                            paramChecks +
                            original.Substring(bodyStartPos)

                        transformedCode <- transformedCode.Replace(original, replacement)

                    return transformedCode
                }

    /// <summary>
    /// Agent that transforms code to replace loops with LINQ
    /// </summary>
    type LinqTransformationAgent() =
        interface ICodeAgent with
            member _.Name = "LinqTransformer"
            member _.Description = "Transforms code to replace loops with LINQ"
            member _.Role = AgentRole.Transformation

        interface ITransformationAgent with
            member _.TransformAsync(code: string) =
                task {
                    // Parse the code
                    let tree = CSharpSyntaxTree.ParseText(code)
                    let root = tree.GetRoot()

                    // Find for loops that could be replaced with LINQ
                    let loopsToReplace =
                        root.DescendantNodes()
                        |> Seq.choose (fun node ->
                            match node with
                            | :? ForStatementSyntax as forLoop ->
                                // Check if this is a loop that calculates a sum
                                match forLoop.Statement with
                                | :? BlockSyntax as block ->
                                    // Look for sum += pattern
                                    let hasSumPattern =
                                        block.Statements
                                        |> Seq.exists (fun stmt ->
                                            match stmt with
                                            | :? ExpressionStatementSyntax as exprStmt ->
                                                match exprStmt.Expression with
                                                | :? AssignmentExpressionSyntax as assignment ->
                                                    assignment.Kind() = SyntaxKind.AddAssignmentExpression ||
                                                    (assignment.Kind() = SyntaxKind.SimpleAssignmentExpression &&
                                                     assignment.Right.ToString().Contains("+"))
                                                | _ -> false
                                            | _ -> false)

                                    if hasSumPattern then
                                        // Extract the collection and sum variable (simplified)
                                        let forLoopText = forLoop.ToString()
                                        let collectionMatch = Regex.Match(forLoopText, @"(\w+)\.Count")
                                        let sumMatch = Regex.Match(forLoopText, @"(\w+)\s*[+=]\s*")

                                        let collection = if collectionMatch.Success then collectionMatch.Groups.[1].Value else "collection"
                                        let sumVar = if sumMatch.Success then sumMatch.Groups.[1].Value else "sum"

                                        Some (forLoop, collection, sumVar)
                                    else
                                        None
                                | _ -> None
                            | _ -> None)
                        |> Seq.toList

                    // Transform the code
                    let mutable transformedCode = code

                    for (forLoop, collection, sumVar) in loopsToReplace do
                        let original = forLoop.ToString()
                        let replacement = $"{sumVar} = {collection}.Sum();"

                        transformedCode <- transformedCode.Replace(original, replacement)

                    return transformedCode
                }
