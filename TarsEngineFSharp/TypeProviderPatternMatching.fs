namespace TarsEngineFSharp

module TypeProviderPatternMatching =
    open System
    open System.IO
    open System.Collections.Generic
    open System.Reflection
    open Microsoft.CodeAnalysis
    open Microsoft.CodeAnalysis.CSharp
    open Microsoft.CodeAnalysis.CSharp.Syntax
    open FSharp.Quotations
    open FSharp.Data
    
    /// <summary>
    /// Type provider for code patterns
    /// </summary>
    type CodePatternProvider = JsonProvider<"""
    {
        "patterns": [
            {
                "name": "NullCheck",
                "description": "Pattern for null checks",
                "language": "CSharp",
                "syntaxKind": "IfStatement",
                "structure": {
                    "condition": {
                        "kind": "BinaryExpression",
                        "operator": "Equals",
                        "left": {
                            "kind": "IdentifierName",
                            "value": "variable"
                        },
                        "right": {
                            "kind": "LiteralExpression",
                            "value": "null"
                        }
                    },
                    "statement": {
                        "kind": "Block",
                        "statements": [
                            {
                                "kind": "ThrowStatement",
                                "expression": {
                                    "kind": "ObjectCreationExpression",
                                    "type": "ArgumentNullException",
                                    "arguments": [
                                        {
                                            "kind": "Argument",
                                            "expression": {
                                                "kind": "InvocationExpression",
                                                "expression": {
                                                    "kind": "IdentifierName",
                                                    "value": "nameof"
                                                },
                                                "arguments": [
                                                    {
                                                        "kind": "Argument",
                                                        "expression": {
                                                            "kind": "IdentifierName",
                                                            "value": "variable"
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        ]
    }
    """>

    // Simple pattern matching function
    let matchPattern (pattern: CodePatternProvider.Pattern) (node: SyntaxNode) =
        match pattern.SyntaxKind with
        | "IfStatement" ->
            match node with
            | :? IfStatementSyntax as ifStmt ->
                // Check if this is a null check
                let condition = ifStmt.Condition.ToString()
                condition.Contains("null")
            | _ -> false
        | "BinaryExpression" ->
            match node with
            | :? BinaryExpressionSyntax as binary ->
                // Check if this is a comparison
                binary.Kind() = SyntaxKind.EqualsExpression
            | _ -> false
        | _ -> false

    // Load patterns from a JSON file
    let loadPatterns (filePath: string) =
        if File.Exists(filePath) then
            let json = File.ReadAllText(filePath)
            let patterns = CodePatternProvider.Parse(json)
            patterns.Patterns
        else
            [||]

    // Find nodes matching a pattern
    let findMatches (pattern: CodePatternProvider.Pattern) (root: SyntaxNode) =
        root.DescendantNodes()
        |> Seq.filter (matchPattern pattern)
        |> Seq.toList
