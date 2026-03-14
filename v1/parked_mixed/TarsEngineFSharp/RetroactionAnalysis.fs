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
        | UnusedVariable
        | MissingExceptionHandling
        | IneffectiveStringConcatenation
        | PerformanceIssue
        | SecurityVulnerability
        | AccessibilityIssue

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

    // Detect missing null checks
    let detectMissingNullChecks (root: SyntaxNode) =
        // Find method parameters that might need null checks
        root.DescendantNodes()
        |> Seq.choose (fun node ->
            match node with
            | :? MethodDeclarationSyntax as method ->
                // Check if the method has parameters
                if method.ParameterList.Parameters.Count > 0 then
                    // Get parameters that are reference types (could be null)
                    let refParams =
                        method.ParameterList.Parameters
                        |> Seq.filter (fun p ->
                            let typeName = p.Type.ToString()
                            not (typeName = "int" || typeName = "double" || typeName = "bool" ||
                                 typeName = "float" || typeName = "decimal" || typeName = "char" ||
                                 typeName = "byte" || typeName = "short" || typeName = "long" ||
                                 typeName = "uint" || typeName = "ushort" || typeName = "ulong" ||
                                 typeName = "sbyte" || typeName.EndsWith("?") || typeName.StartsWith("Nullable<")))
                        |> Seq.toList

                    if refParams.Length > 0 then
                        // Check if there are null checks for these parameters
                        let methodBody = method.Body

                        if methodBody <> null then
                            let missingNullChecks =
                                refParams
                                |> List.filter (fun p ->
                                    let paramName = p.Identifier.Text

                                    // Look for null checks in the method body
                                    let hasNullCheck =
                                        methodBody.DescendantNodes()
                                        |> Seq.exists (fun n ->
                                            match n with
                                            | :? BinaryExpressionSyntax as binary ->
                                                // Check for param == null or param != null
                                                (binary.Kind() = SyntaxKind.EqualsExpression || binary.Kind() = SyntaxKind.NotEqualsExpression) &&
                                                ((binary.Left.ToString() = paramName && binary.Right.ToString() = "null") ||
                                                 (binary.Right.ToString() = paramName && binary.Left.ToString() = "null"))
                                            | :? IfStatementSyntax as ifStmt ->
                                                // Check for if (param == null) or if (param != null)
                                                match ifStmt.Condition with
                                                | :? BinaryExpressionSyntax as binary ->
                                                    (binary.Kind() = SyntaxKind.EqualsExpression || binary.Kind() = SyntaxKind.NotEqualsExpression) &&
                                                    ((binary.Left.ToString() = paramName && binary.Right.ToString() = "null") ||
                                                     (binary.Right.ToString() = paramName && binary.Left.ToString() = "null"))
                                                | _ -> false
                                            | _ -> false)

                                    not hasNullCheck)

                            if missingNullChecks.Length > 0 then
                                Some (method, missingNullChecks)
                            else
                                None
                        else
                            None
                    else
                        None
                else
                    None
            | _ -> None)
        |> Seq.collect (fun (method, missingNullChecks) ->
            missingNullChecks
            |> List.map (fun param ->
                let paramName = param.Identifier.Text

                let issue = {
                    Type = MissingNullCheck
                    Location = sprintf "Line %d" (method.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                    Description = sprintf "Parameter '%s' in method '%s' is not checked for null" paramName method.Identifier.Text
                }

                // Create a fix by adding a null check at the beginning of the method
                let methodBody = method.Body
                let original = methodBody.ToString()

                // Find the position to insert the null check (after the opening brace)
                let insertPos = original.IndexOf('{') + 1
                let indent = String.replicate 4 " " // 4 spaces for indentation

                let nullCheck = sprintf "\n%sif (%s == null)\n%s{\n%s    throw new ArgumentNullException(nameof(%s));\n%s}" indent paramName indent indent paramName indent

                let replacement = original.Insert(insertPos, nullCheck)

                let fix = {
                    Original = original
                    Replacement = replacement
                    Description = sprintf "Add null check for parameter '%s'" paramName
                }

                (issue, fix)))
        |> Seq.toList

    // Detect ineffective loops
    let detectIneffectiveLoops (root: SyntaxNode) =
        // Find for loops that could be replaced with LINQ
        root.DescendantNodes()
        |> Seq.choose (fun node ->
            match node with
            | :? ForStatementSyntax as forLoop ->
                // Check if this is a simple loop that calculates a sum
                let isSum =
                    // Look for a pattern like: for (int i = 0; i < collection.Length; i++) { sum += collection[i]; }
                    let body = forLoop.Statement.ToString()

                    // Check if the body contains a sum operation
                    let sumMatch = Regex.Match(body, @"(\w+)\s*\+=\s*(\w+)\[")

                    if sumMatch.Success then
                        let sumVar = sumMatch.Groups.[1].Value
                        let collection = sumMatch.Groups.[2].Value
                        Some (sumVar, collection)
                    else
                        None

                match isSum with
                | Some (sumVar, collection) ->
                    Some (forLoop, sumVar, collection)
                | None -> None
            | _ -> None)
        |> Seq.map (fun (forLoop, sumVar, collection) ->
            let issue = {
                Type = IneffectiveLoop
                Location = sprintf "Line %d" (forLoop.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                Description = "Manual loop to calculate sum could be replaced with LINQ"
            }

            let fix = {
                Original = forLoop.ToString()
                Replacement = sprintf "%s = %s.Sum();" sumVar collection
                Description = "Replace manual loop with LINQ Sum()"
            }

            (issue, fix))
        |> Seq.toList

    // Detect potential divide by zero risks
    let detectDivideByZeroRisks (root: SyntaxNode) =
        // Find division operations
        root.DescendantNodes()
        |> Seq.choose (fun node ->
            match node with
            | :? BinaryExpressionSyntax as binary when binary.Kind() = SyntaxKind.DivideExpression ->
                // Check if the divisor is a variable or a complex expression
                match binary.Right with
                | :? LiteralExpressionSyntax ->
                    // If it's a literal, we can check if it's zero
                    if binary.Right.ToString() = "0" then
                        Some binary
                    else
                        None
                | _ ->
                    // If it's a variable or expression, check if there's a null check before the division
                    let divisor = binary.Right.ToString()

                    // Find the containing statement
                    let containingStatement =
                        binary.Ancestors()
                        |> Seq.tryPick (function
                            | :? StatementSyntax as stmt -> Some stmt
                            | _ -> None)

                    match containingStatement with
                    | Some stmt ->
                        // Check if there's an if statement checking the divisor before this statement
                        let hasDivisorCheck =
                            stmt.Ancestors()
                            |> Seq.exists (fun ancestor ->
                                match ancestor with
                                | :? BlockSyntax as block ->
                                    // Check if there's an if statement checking the divisor
                                    block.Statements
                                    |> Seq.takeWhile (fun s -> not (Object.ReferenceEquals(s, stmt)))
                                    |> Seq.exists (fun s ->
                                        match s with
                                        | :? IfStatementSyntax as ifStmt ->
                                            // Check if the condition checks the divisor for zero
                                            let condition = ifStmt.Condition.ToString()
                                            condition.Contains(divisor) &&
                                            (condition.Contains("== 0") || condition.Contains("!= 0") ||
                                             condition.Contains("== 0.0") || condition.Contains("!= 0.0") ||
                                             condition.Contains("== 0d") || condition.Contains("!= 0d"))
                                        | _ -> false)
                                | _ -> false)

                        if not hasDivisorCheck then
                            Some binary
                        else
                            None
                    | None -> Some binary // If we can't find the containing statement, flag it to be safe
            | _ -> None)
        |> Seq.map (fun binary ->
            let issue = {
                Type = DivideByZeroRisk
                Location = sprintf "Line %d" (binary.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                Description = sprintf "Potential divide by zero risk with divisor '%s'" (binary.Right.ToString())
            }

            // Create a fix by adding a check before the division
            let divisor = binary.Right.ToString()

            // Find the containing statement
            let containingStatement =
                binary.Ancestors()
                |> Seq.tryPick (function
                    | :? StatementSyntax as stmt -> Some stmt
                    | _ -> None)

            let fix =
                match containingStatement with
                | Some stmt ->
                    let original = stmt.ToString()
                    let indent = String.replicate 4 " " // 4 spaces for indentation

                    let replacement =
                        sprintf "if (%s == 0)\n%s{\n%s    throw new DivideByZeroException();\n%s}\n%s"
                            divisor indent indent indent original

                    {
                        Original = original
                        Replacement = replacement
                        Description = sprintf "Add check for zero divisor '%s'" divisor
                    }
                | None ->
                    // If we can't find the containing statement, provide a generic fix
                    {
                        Original = binary.ToString()
                        Replacement = sprintf "(%s == 0 ? throw new DivideByZeroException() : %s)" divisor (binary.ToString())
                        Description = sprintf "Add inline check for zero divisor '%s'" divisor
                    }

            (issue, fix))
        |> Seq.toList

    // Detect unused variables
    let detectUnusedVariables (root: SyntaxNode) =
        // Find all variable declarations
        let variableDeclarations =
            root.DescendantNodes()
            |> Seq.choose (fun node ->
                match node with
                | :? VariableDeclarationSyntax as varDecl ->
                    // Get the parent (could be a field, local declaration, etc.)
                    match varDecl.Parent with
                    | :? LocalDeclarationStatementSyntax ->
                        // Only consider local variables
                        Some (varDecl, varDecl.Variables |> Seq.map (fun v -> v.Identifier.Text) |> Seq.toList)
                    | _ -> None
                | _ -> None)
            |> Seq.toList

        // For each variable, check if it's used elsewhere in the method
        variableDeclarations
        |> List.collect (fun (varDecl, varNames) ->
            // Get the containing method
            let method =
                varDecl.Ancestors()
                |> Seq.tryPick (function
                    | :? MethodDeclarationSyntax as m -> Some m
                    | _ -> None)

            match method with
            | Some m ->
                // For each variable, check if it's used in the method body
                varNames
                |> List.choose (fun varName ->
                    // Count references to this variable
                    let references =
                        m.DescendantNodes()
                        |> Seq.choose (fun n ->
                            match n with
                            | :? IdentifierNameSyntax as id when id.Identifier.Text = varName ->
                                // Make sure this isn't the declaration itself
                                if not (varDecl.Span.Contains(id.Span)) then
                                    Some id
                                else
                                    None
                            | _ -> None)
                        |> Seq.length

                    // If no references, it's unused
                    if references = 0 then
                        let issue = {
                            Type = UnusedVariable
                            Location = sprintf "Line %d" (varDecl.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                            Description = sprintf "Variable '%s' is declared but never used" varName
                        }

                        // Create a fix to remove the variable
                        let original = varDecl.Parent.ToString()

                        // If there's only one variable in the declaration, remove the whole statement
                        let replacement =
                            if varDecl.Variables.Count = 1 then
                                "" // Remove the entire statement
                            else
                                // This is a simplified approach - in a real implementation, we'd need to be more careful
                                // about removing just the specific variable while preserving others
                                original.Replace(varName, "/* removed unused variable */")

                        let fix = {
                            Original = original
                            Replacement = replacement
                            Description = sprintf "Remove unused variable '%s'" varName
                        }

                        Some (issue, fix)
                    else
                        None)
            | None -> [])
        |> List.toArray
        |> Array.toList

    // Detect missing exception handling
    let detectMissingExceptionHandling (root: SyntaxNode) =
        // Find methods that call methods that could throw exceptions but don't have try/catch
        root.DescendantNodes()
        |> Seq.choose (fun node ->
            match node with
            | :? MethodDeclarationSyntax as method ->
                // Check if the method has any try/catch blocks
                let hasTryCatch =
                    method.DescendantNodes()
                    |> Seq.exists (fun n -> n :? TryStatementSyntax)

                // Check if the method calls any methods that typically throw exceptions
                let riskyCalls =
                    method.DescendantNodes()
                    |> Seq.choose (fun n ->
                        match n with
                        | :? InvocationExpressionSyntax as invoke ->
                            let methodName = invoke.Expression.ToString()
                            // List of methods that commonly throw exceptions
                            if methodName.Contains(".Open") ||
                               methodName.Contains(".Read") ||
                               methodName.Contains(".Write") ||
                               methodName.Contains(".Parse") ||
                               methodName.Contains(".Convert") ||
                               methodName.EndsWith(".First") ||
                               methodName.EndsWith(".Single") then
                                Some methodName
                            else
                                None
                        | _ -> None)
                    |> Seq.toList

                if not hasTryCatch && riskyCalls.Length > 0 then
                    Some (method, riskyCalls)
                else
                    None
            | _ -> None)
        |> Seq.map (fun (method, riskyCalls) ->
            let issue = {
                Type = MissingExceptionHandling
                Location = sprintf "Line %d" (method.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                Description = sprintf "Method '%s' calls potentially throwing methods (%s) without exception handling" method.Identifier.Text (String.Join(", ", riskyCalls))
            }

            // Create a fix by wrapping the method body in a try/catch
            let original = method.Body.ToString()

            // Remove the braces from the original body
            let bodyContent = original.Substring(1, original.Length - 2).Trim()

            let replacement =
                "{\n    try\n    {\n" + bodyContent + "\n    }\n    catch (Exception ex)\n    {\n        // TODO: Handle or rethrow exception appropriately\n        throw;\n    }\n}"

            let fix = {
                Original = original
                Replacement = replacement
                Description = sprintf "Add try/catch block to method '%s'" method.Identifier.Text
            }

            (issue, fix))
        |> Seq.toList

    // Detect string concatenation that could be replaced with string interpolation
    let detectIneffectiveStringConcatenation (root: SyntaxNode) =
        // Find string concatenation expressions
        root.DescendantNodes()
        |> Seq.choose (fun node ->
            match node with
            | :? BinaryExpressionSyntax as binary when binary.Kind() = SyntaxKind.AddExpression ->
                // Check if both sides involve strings
                let isStringConcat =
                    (binary.Left.ToString().Contains("\"") || binary.Left.ToString().StartsWith("string")) &&
                    (binary.Right.ToString().Contains("\"") || binary.Right.ToString().StartsWith("string"))

                if isStringConcat then Some binary else None
            | _ -> None)
        |> Seq.map (fun binary ->
            let issue = {
                Type = IneffectiveStringConcatenation
                Location = sprintf "Line %d" (binary.GetLocation().GetLineSpan().StartLinePosition.Line + 1)
                Description = "String concatenation could be replaced with string interpolation"
            }

            // Create a fix to replace with string interpolation
            let original = binary.ToString()

            // This is a simplified approach - in a real implementation, we'd need to be more careful
            // about constructing the interpolated string
            let replacement =
                if original.Contains("+") then
                    let parts = original.Split([|'+' |], StringSplitOptions.RemoveEmptyEntries)
                                |> Array.map (fun p -> p.Trim())

                    // Check if we have string literals and variables
                    let hasStringLiteral = parts |> Array.exists (fun p -> p.StartsWith("\""))
                    let hasVariable = parts |> Array.exists (fun p -> not (p.StartsWith("\"")))

                    if hasStringLiteral && hasVariable then
                        // Replace with interpolated string
                        let interpolated =
                            parts
                            |> Array.map (fun p ->
                                if p.StartsWith("\"") then
                                    // Remove quotes from string literals
                                    p.Trim('"')
                                else
                                    // Wrap variables in {}
                                    "{" + p + "}"
                            )
                            |> String.concat ""

                        "$\"" + interpolated + "\""
                    else
                        original
                else
                    original

            let fix = {
                Original = original
                Replacement = replacement
                Description = "Replace string concatenation with string interpolation"
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
            let unusedVarIssues = detectUnusedVariables root
            let exceptionHandlingIssues = detectMissingExceptionHandling root
            let stringConcatIssues = detectIneffectiveStringConcatenation root

            // Combine all issues and fixes
            {
                FilePath = filePath
                Issues =
                    (nullCheckIssues |> List.map fst) @
                    (loopIssues |> List.map fst) @
                    (divideByZeroIssues |> List.map fst) @
                    (unusedVarIssues |> List.map fst) @
                    (exceptionHandlingIssues |> List.map fst) @
                    (stringConcatIssues |> List.map fst)
                Fixes =
                    (nullCheckIssues |> List.map snd) @
                    (loopIssues |> List.map snd) @
                    (divideByZeroIssues |> List.map snd) @
                    (unusedVarIssues |> List.map snd) @
                    (exceptionHandlingIssues |> List.map snd) @
                    (stringConcatIssues |> List.map snd)
            }
        with
        | ex ->
            printfn $"Error analyzing file %s{filePath}: %s{ex.Message}"
            { FilePath = filePath; Issues = []; Fixes = [] }

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
            let unusedVarIssues = detectUnusedVariables root
            let exceptionHandlingIssues = detectMissingExceptionHandling root
            let stringConcatIssues = detectIneffectiveStringConcatenation root

            // Combine all issues and fixes
            {
                FilePath = "<memory>"
                Issues =
                    (nullCheckIssues |> List.map fst) @
                    (loopIssues |> List.map fst) @
                    (divideByZeroIssues |> List.map fst) @
                    (unusedVarIssues |> List.map fst) @
                    (exceptionHandlingIssues |> List.map fst) @
                    (stringConcatIssues |> List.map fst)
                Fixes =
                    (nullCheckIssues |> List.map snd) @
                    (loopIssues |> List.map snd) @
                    (divideByZeroIssues |> List.map snd) @
                    (unusedVarIssues |> List.map snd) @
                    (exceptionHandlingIssues |> List.map snd) @
                    (stringConcatIssues |> List.map snd)
            }
        with
        | ex ->
            printfn $"Error analyzing code: %s{ex.Message}"
            { FilePath = "<memory>"; Issues = []; Fixes = [] }

    // Apply fixes to a file
    let applyFixes (filePath: string) (fixes: CodeFix list) : unit =
        try
            // Read the file
            let code = File.ReadAllText(filePath)

            // Apply each fix
            // Note: This is a simplified approach that doesn't handle overlapping fixes
            let transformedCode =
                fixes |> List.fold (fun (acc: string) (fix: CodeFix) ->
                    acc.Replace(fix.Original, fix.Replacement)) code

            // Write the transformed code back to the file
            File.WriteAllText(filePath, transformedCode)
        with
        | ex ->
            printfn $"Error applying fixes to file %s{filePath}: %s{ex.Message}"

    // Apply fixes to a code string
    let applyFixesToCode (code: string) (fixes: CodeFix list) : string =
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
