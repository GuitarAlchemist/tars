namespace TarsEngineFSharp

module RetroactionAnalysis =
    open System
    open System.IO
    open System.Text.RegularExpressions
    open Microsoft.CodeAnalysis
    open Microsoft.CodeAnalysis.CSharp
    open Microsoft.CodeAnalysis.CSharp.Syntax

    // Simple types for code issues and fixes
    type CodeIssueType =
        | MissingNullCheck
        | IneffectiveLoop
        | DivideByZeroRisk
        | MissingDocumentation

    type CodeIssue = {
        Type: CodeIssueType
        Location: string
        Description: string
    }

    type CodeFix = {
        Original: string
        Replacement: string
        Description: string
    }

    type AnalysisResult = {
        FilePath: string
        Issues: CodeIssue list
        Fixes: CodeFix list
    }

    // Pattern detection functions
    let detectMissingNullChecks (root: SyntaxNode) =
        // Find methods with reference type parameters that don't have null checks
        root.DescendantNodes()
        |> Seq.choose (fun node ->
            match node with
            | :? MethodDeclarationSyntax as method ->
                // Find reference type parameters
                let refParams =
                    method.ParameterList.Parameters
                    |> Seq.filter (fun p ->
                        match p.Type with
                        | :? IdentifierNameSyntax as typeName ->
                            // Simple check for reference types
                            let typeText = typeName.Identifier.Text
                            typeText = "string" ||
                            typeText.EndsWith("[]") ||
                            (typeText <> "int" &&
                             typeText <> "double" &&
                             typeText <> "bool" &&
                             typeText <> "decimal" &&
                             typeText <> "float")
                        | _ -> false)
                    |> Seq.map (fun p -> p.Identifier.Text)
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
                    else None
                else None
            | _ -> None)
        |> Seq.map (fun (method, parameters) ->
            let issue = {
                Type = MissingNullCheck
                Location = sprintf "Line %d" (method.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                Description = sprintf "Method %s has reference type parameters %s without null checks" method.Identifier.Text (String.Join(", ", parameters))
            }

            let paramChecks =
                parameters
                |> List.map (fun p -> $"            if ({p} == null) throw new ArgumentNullException(nameof({p}));")
                |> String.concat Environment.NewLine

            // Create a fix by adding null checks at the beginning of the method body
            let original = method.ToString()
            let bodyStart = method.Body.OpenBraceToken.Span.End
            let bodyStartPos = bodyStart - method.SpanStart + 1  // +1 for the newline after {

            let replacement =
                original.Substring(0, bodyStartPos) +
                Environment.NewLine +
                paramChecks +
                original.Substring(bodyStartPos)

            let fix = {
                Original = original
                Replacement = replacement
                Description = sprintf "Add null checks for parameters: %s" (String.Join(", ", parameters))
            }

            (issue, fix))
        |> Seq.toList

    let detectIneffectiveLoops (root: SyntaxNode) =
        // Find for loops that could be replaced with LINQ
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

                    if hasSumPattern then Some forLoop else None
                | _ -> None
            | _ -> None)
        |> Seq.map (fun forLoop ->
            // Extract the collection and sum variable (simplified)
            let forLoopText = forLoop.ToString()
            let collectionMatch = Regex.Match(forLoopText, @"(\w+)\.Count")
            let sumMatch = Regex.Match(forLoopText, @"(\w+)\s*[+=]\s*")

            let collection = if collectionMatch.Success then collectionMatch.Groups.[1].Value else "collection"
            let sumVar = if sumMatch.Success then sumMatch.Groups.[1].Value else "sum"

            let issue = {
                Type = IneffectiveLoop
                Location = sprintf "Line %d" (forLoop.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                Description = "Manual loop to calculate sum could be replaced with LINQ"
            }

            let fix = {
                Original = forLoop.ToString()
                Replacement = $"{sumVar} = {collection}.Sum();"
                Description = "Replace manual loop with LINQ Sum()"
            }

            (issue, fix))
        |> Seq.toList

    let detectDivideByZeroRisks (root: SyntaxNode) =
        // Find division operations without zero checks
        root.DescendantNodes()
        |> Seq.choose (fun node ->
            match node with
            | :? BinaryExpressionSyntax as binary when binary.Kind() = SyntaxKind.DivideExpression ->
                // Get the method containing this division
                let method =
                    binary.Ancestors()
                    |> Seq.tryPick (function
                        | :? MethodDeclarationSyntax as m -> Some m
                        | _ -> None)

                match method with
                | Some m ->
                    let divisor = binary.Right.ToString()

                    // Check if there's a zero check for this divisor
                    let hasZeroCheck =
                        m.DescendantNodes()
                        |> Seq.exists (fun n ->
                            match n with
                            | :? IfStatementSyntax as ifStmt ->
                                let condition = ifStmt.Condition.ToString()
                                condition.Contains($"{divisor} == 0") ||
                                condition.Contains($"{divisor} <= 0") ||
                                condition.Contains($"{divisor} < 1")
                            | _ -> false)

                    if not hasZeroCheck && not (divisor = "0" || divisor = "1") then
                        Some (binary, divisor)
                    else None
                | None -> None
            | _ -> None)
        |> Seq.map (fun (binary, divisor) ->
            let issue = {
                Type = DivideByZeroRisk
                Location = sprintf "Line %d" (binary.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                Description = $"Division by {divisor} without zero check"
            }

            // Find the statement containing the division
            let statement =
                binary.Ancestors()
                |> Seq.tryPick (function
                    | :? StatementSyntax as s -> Some s
                    | _ -> None)

            match statement with
            | Some stmt ->
                let original = stmt.ToString()
                let zeroCheck = $"if ({divisor} == 0) throw new DivideByZeroException(\"Cannot divide by zero\");"

                let fix = {
                    Original = original
                    Replacement = $"{zeroCheck}{Environment.NewLine}            {original}"
                    Description = $"Add zero check for divisor {divisor}"
                }

                (issue, fix)
            | None ->
                let original = binary.ToString()

                let fix = {
                    Original = original
                    Replacement = $"({divisor} == 0 ? throw new DivideByZeroException(\"Cannot divide by zero\") : {original})"
                    Description = $"Add inline zero check for divisor {divisor}"
                }

                (issue, fix))
        |> Seq.toList

    // Main analysis function
    let analyzeFile (filePath: string) : AnalysisResult =
        try
            // Read the file and parse it with Roslyn
            let code = File.ReadAllText(filePath)
            let tree = CSharpSyntaxTree.ParseText(code)
            let root = tree.GetRoot()

            // Detect issues
            let nullCheckIssues = detectMissingNullChecks root
            let loopIssues = detectIneffectiveLoops root
            let divideByZeroIssues = detectDivideByZeroRisks root

            // Combine all issues and fixes
            {
                FilePath = filePath
                Issues =
                    (nullCheckIssues |> List.map fst) @
                    (loopIssues |> List.map fst) @
                    (divideByZeroIssues |> List.map fst)
                Fixes =
                    (nullCheckIssues |> List.map snd) @
                    (loopIssues |> List.map snd) @
                    (divideByZeroIssues |> List.map snd)
            }
        with
        | ex ->
            printfn $"Error analyzing file %s{filePath}: %s{ex.Message}"
            { FilePath = filePath; Issues = []; Fixes = [] }

    // Analyze multiple files
    let analyzeProject (projectPath: string) (maxFiles: int) : AnalysisResult list =
        if Directory.Exists(projectPath) then
            Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories)
            |> Seq.truncate maxFiles
            |> Seq.map analyzeFile
            |> Seq.toList
        else
            []

    // Apply fixes to a file
    let applyFixesToFile (filePath: string) (fixes: CodeFix list) : string =
        try
            let code = File.ReadAllText(filePath)

            // Apply each fix
            // Note: This is a simplified approach that doesn't handle overlapping fixes
            let transformedCode =
                fixes |> List.fold (fun (acc: string) (fix: CodeFix) ->
                    acc.Replace(fix.Original, fix.Replacement)) code

            transformedCode
        with
        | ex ->
            printfn $"Error applying fixes to %s{filePath}: %s{ex.Message}"
            ""

    // Analyze code string directly
    let analyzeCode (code: string) : AnalysisResult =
        try
            // Parse the code with Roslyn
            let tree = CSharpSyntaxTree.ParseText(code)
            let root = tree.GetRoot()

            // Detect issues
            let nullCheckIssues = detectMissingNullChecks root
            let loopIssues = detectIneffectiveLoops root
            let divideByZeroIssues = detectDivideByZeroRisks root

            // Combine all issues and fixes
            {
                FilePath = "<memory>"
                Issues =
                    (nullCheckIssues |> List.map fst) @
                    (loopIssues |> List.map fst) @
                    (divideByZeroIssues |> List.map fst)
                Fixes =
                    (nullCheckIssues |> List.map snd) @
                    (loopIssues |> List.map snd) @
                    (divideByZeroIssues |> List.map snd)
            }
        with
        | ex ->
            printfn $"Error analyzing code: %s{ex.Message}"
            { FilePath = "<memory>"; Issues = []; Fixes = [] }

    // Apply fixes to a code string
    let applyFixes (code: string) (fixes: CodeFix list) : string =
        try
            // Apply each fix
            // Note: This is a simplified approach that doesn't handle overlapping fixes
            let transformedCode =
                fixes |> List.fold (fun (acc: string) (fix: CodeFix) ->
                    acc.Replace(fix.Original, fix.Replacement)) code

            transformedCode
        with
        | ex ->
            printfn $"Error applying fixes to code: %s{ex.Message}"
            code
