namespace TarsEngineFSharp

module EnhancedCodeAnalysis =
    open System
    open System.IO
    open System.Text.RegularExpressions
    open Microsoft.CodeAnalysis
    open Microsoft.CodeAnalysis.CSharp
    open Microsoft.CodeAnalysis.CSharp.Syntax

    // Define type for code patterns
    type CodeIssueType =
        | DivideByZero
        | ManualLoop
        | MissingNullCheck
        | MissingDocumentation

    // Define a type for code issues
    type CodeIssue = {
        Message: string
        Description: string
        Location: string
        IssueType: CodeIssueType
    }

    // Define a type for code fixes
    type CodeFix = {
        Original: string
        Replacement: string
        Description: string
    }

    // Helper function to find method declarations
    let findMethodDeclarations (root: SyntaxNode) =
        root.DescendantNodes()
        |> Seq.choose (function
            | :? MethodDeclarationSyntax as method -> Some method
            | _ -> None)

    // Helper function to find binary expressions
    let findBinaryExpressions (root: SyntaxNode) =
        root.DescendantNodes()
        |> Seq.choose (function
            | :? BinaryExpressionSyntax as binary -> Some binary
            | _ -> None)

    // Helper function to find for statements
    let findForStatements (root: SyntaxNode) =
        root.DescendantNodes()
        |> Seq.choose (function
            | :? ForStatementSyntax as forStmt -> Some forStmt
            | _ -> None)

    // Pattern for missing null checks
    let detectMissingNullChecks (root: SyntaxNode) =
        findMethodDeclarations root
        |> Seq.choose (fun method ->
            // Check if method has parameters that could be null
            let hasReferenceTypeParams =
                if method.ParameterList <> null then
                    method.ParameterList.Parameters
                    |> Seq.exists (fun p ->
                        if p.Type <> null then
                            match p.Type with
                            | :? IdentifierNameSyntax as typeName ->
                                // Simple check - could be enhanced to check actual type
                                typeName.Identifier.Text <> "int" &&
                                typeName.Identifier.Text <> "double" &&
                                typeName.Identifier.Text <> "bool"
                            | _ -> false
                        else
                            false)
                else
                    false

            // Check if method body has null checks
            let hasNullChecks =
                if method.Body <> null then
                    method.Body.Statements
                    |> Seq.exists (fun stmt ->
                        match stmt with
                        | :? IfStatementSyntax as ifStmt ->
                            // Simple check for null comparison
                            let condition = ifStmt.Condition.ToString()
                            condition.Contains("null")
                        | _ -> false)
                else
                    false

            if hasReferenceTypeParams && not hasNullChecks then
                Some (method, MissingNullCheck)
            else
                None)
        |> Seq.toList

    // Pattern for divide by zero risks
    let detectDivideByZeroRisks (root: SyntaxNode) =
        findBinaryExpressions root
        |> Seq.choose (fun binary ->
            if binary.Kind() = SyntaxKind.DivideExpression then
                let divisor = binary.Right.ToString()
                
                // Check if there's a zero check for this divisor
                let method = 
                    binary.Ancestors()
                    |> Seq.tryPick (function
                        | :? MethodDeclarationSyntax as m -> Some m
                        | _ -> None)
                
                match method with
                | Some m when m.Body <> null ->
                    let hasZeroCheck =
                        m.Body.Statements
                        |> Seq.exists (fun stmt ->
                            match stmt with
                            | :? IfStatementSyntax as ifStmt ->
                                let condition = ifStmt.Condition.ToString()
                                condition.Contains($"{divisor} == 0") ||
                                condition.Contains($"{divisor} <= 0") ||
                                condition.Contains($"{divisor} < 1")
                            | _ -> false)
                    
                    if not hasZeroCheck && not (divisor = "0" || divisor = "1") then
                        Some (binary, DivideByZero)
                    else
                        None
                | _ -> None
            else
                None)
        |> Seq.toList

    // Pattern for inefficient loops
    let detectIneffectiveLoops (root: SyntaxNode) =
        findForStatements root
        |> Seq.choose (fun forStmt ->
            // Check if this is a loop that calculates a sum
            match forStmt.Statement with
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
                    Some (forStmt, ManualLoop)
                else
                    None
            | _ -> None)
        |> Seq.toList

    // Convert pattern matches to code issues
    let createIssue (node: SyntaxNode, issueType: CodeIssueType) =
        let location = node.GetLocation().GetLineSpan().StartLinePosition.Line + 1
        
        match issueType with
        | DivideByZero ->
            let binary = node :?> BinaryExpressionSyntax
            let divisor = binary.Right.ToString()
            {
                Message = "Potential divide by zero"
                Description = "Division by " + divisor + " without zero check"
                Location = "Line " + location.ToString()
                IssueType = DivideByZero
            }
        
        | ManualLoop ->
            let forStmt = node :?> ForStatementSyntax
            {
                Message = "Inefficient loop"
                Description = "Manual loop could be replaced with LINQ"
                Location = "Line " + location.ToString()
                IssueType = ManualLoop
            }
        
        | MissingNullCheck ->
            let method = node :?> MethodDeclarationSyntax
            let paramNames = 
                method.ParameterList.Parameters
                |> Seq.map (fun p -> p.Identifier.Text)
                |> String.concat ", "
            {
                Message = "Missing null check"
                Description = "Method " + method.Identifier.Text + " has reference type parameters (" + paramNames + ") without null checks"
                Location = "Line " + location.ToString()
                IssueType = MissingNullCheck
            }
        
        | MissingDocumentation ->
            let method = node :?> MethodDeclarationSyntax
            {
                Message = "Missing documentation"
                Description = "Public method " + method.Identifier.Text + " is missing XML documentation"
                Location = "Line " + location.ToString()
                IssueType = MissingDocumentation
            }

    // Create code fixes for issues
    let createFix (issue: CodeIssue) =
        match issue.IssueType with
        | DivideByZero ->
            // Extract the divisor from the description
            let divisorMatch = Regex.Match(issue.Description, @"Division by (\w+)")
            if divisorMatch.Success then
                let divisor = divisorMatch.Groups.[1].Value
                {
                    Original = "/ " + divisor
                    Replacement = "/ (" + divisor + " != 0 ? " + divisor + " : throw new DivideByZeroException())"
                    Description = "Add zero check for divisor " + divisor
                }
            else
                {
                    Original = ""
                    Replacement = ""
                    Description = "Could not determine fix"
                }
        
        | ManualLoop ->
            {
                Original = "for (int i = 0; i < items.Count; i++) { sum += items[i]; }"
                Replacement = "sum = items.Sum();"
                Description = "Replace manual loop with LINQ Sum()"
            }
        
        | MissingNullCheck ->
            // Extract method name and parameters from description
            let methodMatch = Regex.Match(issue.Description, @"Method (\w+) has reference type parameters \((.*?)\)")
            if methodMatch.Success then
                let methodName = methodMatch.Groups.[1].Value
                let parameters = methodMatch.Groups.[2].Value.Split([|", "|], StringSplitOptions.None)
                
                let nullChecks = 
                    parameters 
                    |> Array.map (fun p -> "    if (" + p + " == null) throw new ArgumentNullException(nameof(" + p + "));")
                    |> String.concat Environment.NewLine
                
                {
                    Original = "public void " + methodName + "("
                    Replacement = "public void " + methodName + "(" + Environment.NewLine + nullChecks
                    Description = "Add null checks for parameters: " + String.Join(", ", parameters)
                }
            else
                {
                    Original = ""
                    Replacement = ""
                    Description = "Could not determine fix"
                }
        
        | MissingDocumentation ->
            // Extract method name from description
            let methodMatch = Regex.Match(issue.Description, @"Public method (\w+)")
            if methodMatch.Success then
                let methodName = methodMatch.Groups.[1].Value
                {
                    Original = "public void " + methodName
                    Replacement = "/// <summary>" + Environment.NewLine + "/// Description for " + methodName + Environment.NewLine + "/// </summary>" + Environment.NewLine + "public void " + methodName
                    Description = "Add XML documentation for method " + methodName
                }
            else
                {
                    Original = ""
                    Replacement = ""
                    Description = "Could not determine fix"
                }

    // Analyze code and return issues and fixes
    let analyzeCode (code: string) =
        let tree = CSharpSyntaxTree.ParseText(code)
        let root = tree.GetRoot()
        
        // Detect all issues
        let nullCheckIssues = detectMissingNullChecks root |> List.map createIssue
        let divideByZeroIssues = detectDivideByZeroRisks root |> List.map createIssue
        let loopIssues = detectIneffectiveLoops root |> List.map createIssue
        
        // Combine all issues
        let allIssues = nullCheckIssues @ divideByZeroIssues @ loopIssues
        
        // Create fixes for all issues
        let fixes = allIssues |> List.map createFix
        
        (allIssues, fixes)
