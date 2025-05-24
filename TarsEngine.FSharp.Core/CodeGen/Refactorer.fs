namespace TarsEngine.FSharp.Core.CodeGen

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of IRefactorer for various languages.
/// </summary>
type Refactorer(logger: ILogger<Refactorer>, refactorings: CodeRefactoring list) =
    
    /// <summary>
    /// Gets the language supported by this refactorer.
    /// </summary>
    member _.Language = "csharp"
    
    /// <summary>
    /// Gets all available refactorings.
    /// </summary>
    /// <returns>The list of available refactorings.</returns>
    member _.GetAvailableRefactorings() =
        task {
            return refactorings
        }
    
    /// <summary>
    /// Gets a refactoring by name.
    /// </summary>
    /// <param name="refactoringName">The name of the refactoring to get.</param>
    /// <returns>The refactoring, if found.</returns>
    member _.GetRefactoringByName(refactoringName: string) =
        task {
            return refactorings |> List.tryFind (fun r -> r.Name.Equals(refactoringName, StringComparison.OrdinalIgnoreCase))
        }
    
    /// <summary>
    /// Gets refactorings by category.
    /// </summary>
    /// <param name="category">The category of refactorings to get.</param>
    /// <returns>The list of refactorings in the category.</returns>
    member _.GetRefactoringsByCategory(category: string) =
        task {
            return refactorings |> List.filter (fun r -> r.Category.Equals(category, StringComparison.OrdinalIgnoreCase))
        }
    
    /// <summary>
    /// Gets refactorings by tag.
    /// </summary>
    /// <param name="tag">The tag of refactorings to get.</param>
    /// <returns>The list of refactorings with the tag.</returns>
    member _.GetRefactoringsByTag(tag: string) =
        task {
            return refactorings |> List.filter (fun r -> r.Tags |> List.exists (fun t -> t.Equals(tag, StringComparison.OrdinalIgnoreCase)))
        }
    
    /// <summary>
    /// Refactors code using a specific refactoring.
    /// </summary>
    /// <param name="code">The code to refactor.</param>
    /// <param name="refactoring">The refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    member _.RefactorCode(code: string, refactoring: CodeRefactoring) =
        try
            logger.LogInformation("Refactoring code using: {RefactoringName}", refactoring.Name)
            
            // Create a regex pattern from the before code
            let pattern = Regex.Escape(refactoring.BeforeCode)
                            .Replace("\\{", "{")
                            .Replace("\\}", "}")
                            .Replace("\\(", "(")
                            .Replace("\\)", ")")
                            .Replace("\\[", "[")
                            .Replace("\\]", "]")
                            .Replace("\\*", "*")
                            .Replace("\\+", "+")
                            .Replace("\\?", "?")
                            .Replace("\\.", ".")
                            .Replace("\\^", "^")
                            .Replace("\\$", "$")
                            .Replace("\\|", "|")
                            .Replace("\\\\", "\\")
            
            // Replace the pattern with the after code
            let refactoredCode = Regex.Replace(code, pattern, refactoring.AfterCode)
            
            // Create the result
            {
                OriginalCode = code
                RefactoredCode = refactoredCode
                Refactoring = refactoring
                Explanation = refactoring.Explanation
                AdditionalInfo = Map.empty
            }
        with
        | ex ->
            logger.LogError(ex, "Error refactoring code using: {RefactoringName}", refactoring.Name)
            {
                OriginalCode = code
                RefactoredCode = code
                Refactoring = refactoring
                Explanation = $"Error refactoring code: {ex.Message}"
                AdditionalInfo = Map.empty
            }
    
    /// <summary>
    /// Refactors code using a refactoring name.
    /// </summary>
    /// <param name="code">The code to refactor.</param>
    /// <param name="refactoringName">The name of the refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    member this.RefactorCodeWithName(code: string, refactoringName: string) =
        task {
            try
                logger.LogInformation("Refactoring code using refactoring name: {RefactoringName}", refactoringName)
                
                // Get the refactoring
                let! refactoring = this.GetRefactoringByName(refactoringName)
                
                // Refactor the code
                match refactoring with
                | Some r ->
                    return this.RefactorCode(code, r)
                | None ->
                    logger.LogError("Refactoring not found: {RefactoringName}", refactoringName)
                    return {
                        OriginalCode = code
                        RefactoredCode = code
                        Refactoring = {
                            Name = refactoringName
                            Description = "Refactoring not found"
                            Language = ""
                            Category = ""
                            Tags = []
                            BeforeCode = ""
                            AfterCode = ""
                            Explanation = $"Refactoring not found: {refactoringName}"
                            AdditionalInfo = Map.empty
                        }
                        Explanation = $"Refactoring not found: {refactoringName}"
                        AdditionalInfo = Map.empty
                    }
            with
            | ex ->
                logger.LogError(ex, "Error refactoring code using refactoring name: {RefactoringName}", refactoringName)
                return {
                    OriginalCode = code
                    RefactoredCode = code
                    Refactoring = {
                        Name = refactoringName
                        Description = "Error refactoring code"
                        Language = ""
                        Category = ""
                        Tags = []
                        BeforeCode = ""
                        AfterCode = ""
                        Explanation = $"Error refactoring code: {ex.Message}"
                        AdditionalInfo = Map.empty
                    }
                    Explanation = $"Error refactoring code: {ex.Message}"
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Refactors a file using a specific refactoring.
    /// </summary>
    /// <param name="filePath">The path to the file to refactor.</param>
    /// <param name="refactoring">The refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    member this.RefactorFile(filePath: string, refactoring: CodeRefactoring) =
        task {
            try
                logger.LogInformation("Refactoring file: {FilePath} using: {RefactoringName}", filePath, refactoring.Name)
                
                // Read the file content
                let code = File.ReadAllText(filePath)
                
                // Refactor the code
                let result = this.RefactorCode(code, refactoring)
                
                // Write the refactored code back to the file
                if result.OriginalCode <> result.RefactoredCode then
                    File.WriteAllText(filePath, result.RefactoredCode)
                
                return result
            with
            | ex ->
                logger.LogError(ex, "Error refactoring file: {FilePath} using: {RefactoringName}", filePath, refactoring.Name)
                return {
                    OriginalCode = ""
                    RefactoredCode = ""
                    Refactoring = refactoring
                    Explanation = $"Error refactoring file: {ex.Message}"
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Refactors a file using a refactoring name.
    /// </summary>
    /// <param name="filePath">The path to the file to refactor.</param>
    /// <param name="refactoringName">The name of the refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    member this.RefactorFileWithName(filePath: string, refactoringName: string) =
        task {
            try
                logger.LogInformation("Refactoring file: {FilePath} using refactoring name: {RefactoringName}", filePath, refactoringName)
                
                // Get the refactoring
                let! refactoring = this.GetRefactoringByName(refactoringName)
                
                // Refactor the file
                match refactoring with
                | Some r ->
                    return! this.RefactorFile(filePath, r)
                | None ->
                    logger.LogError("Refactoring not found: {RefactoringName}", refactoringName)
                    return {
                        OriginalCode = ""
                        RefactoredCode = ""
                        Refactoring = {
                            Name = refactoringName
                            Description = "Refactoring not found"
                            Language = ""
                            Category = ""
                            Tags = []
                            BeforeCode = ""
                            AfterCode = ""
                            Explanation = $"Refactoring not found: {refactoringName}"
                            AdditionalInfo = Map.empty
                        }
                        Explanation = $"Refactoring not found: {refactoringName}"
                        AdditionalInfo = Map.empty
                    }
            with
            | ex ->
                logger.LogError(ex, "Error refactoring file: {FilePath} using refactoring name: {RefactoringName}", filePath, refactoringName)
                return {
                    OriginalCode = ""
                    RefactoredCode = ""
                    Refactoring = {
                        Name = refactoringName
                        Description = "Error refactoring file"
                        Language = ""
                        Category = ""
                        Tags = []
                        BeforeCode = ""
                        AfterCode = ""
                        Explanation = $"Error refactoring file: {ex.Message}"
                        AdditionalInfo = Map.empty
                    }
                    Explanation = $"Error refactoring file: {ex.Message}"
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Suggests refactorings for code.
    /// </summary>
    /// <param name="code">The code to suggest refactorings for.</param>
    /// <returns>The list of suggested refactorings.</returns>
    member this.SuggestRefactorings(code: string) =
        task {
            try
                logger.LogInformation("Suggesting refactorings for code")
                
                // Get all refactorings for the language
                let languageRefactorings = refactorings |> List.filter (fun r -> r.Language.Equals(this.Language, StringComparison.OrdinalIgnoreCase))
                
                // Find refactorings that match the code
                let suggestedRefactorings = ResizeArray<CodeRefactoring>()
                
                for refactoring in languageRefactorings do
                    // Create a regex pattern from the before code
                    let pattern = Regex.Escape(refactoring.BeforeCode)
                                    .Replace("\\{", "{")
                                    .Replace("\\}", "}")
                                    .Replace("\\(", "(")
                                    .Replace("\\)", ")")
                                    .Replace("\\[", "[")
                                    .Replace("\\]", "]")
                                    .Replace("\\*", "*")
                                    .Replace("\\+", "+")
                                    .Replace("\\?", "?")
                                    .Replace("\\.", ".")
                                    .Replace("\\^", "^")
                                    .Replace("\\$", "$")
                                    .Replace("\\|", "|")
                                    .Replace("\\\\", "\\")
                    
                    // Check if the pattern matches the code
                    if Regex.IsMatch(code, pattern) then
                        suggestedRefactorings.Add(refactoring)
                
                return suggestedRefactorings |> Seq.toList
            with
            | ex ->
                logger.LogError(ex, "Error suggesting refactorings for code")
                return []
        }
    
    /// <summary>
    /// Suggests refactorings for a file.
    /// </summary>
    /// <param name="filePath">The path to the file to suggest refactorings for.</param>
    /// <returns>The list of suggested refactorings.</returns>
    member this.SuggestRefactoringsForFile(filePath: string) =
        task {
            try
                logger.LogInformation("Suggesting refactorings for file: {FilePath}", filePath)
                
                // Read the file content
                let code = File.ReadAllText(filePath)
                
                // Suggest refactorings for the code
                return! this.SuggestRefactorings(code)
            with
            | ex ->
                logger.LogError(ex, "Error suggesting refactorings for file: {FilePath}", filePath)
                return []
        }
    
    interface IRefactorer with
        member this.Language = this.Language
        member this.RefactorCode(code, refactoring) = this.RefactorCode(code, refactoring)
        member this.RefactorCodeWithName(code, refactoringName) = this.RefactorCodeWithName(code, refactoringName)
        member this.RefactorFile(filePath, refactoring) = this.RefactorFile(filePath, refactoring)
        member this.RefactorFileWithName(filePath, refactoringName) = this.RefactorFileWithName(filePath, refactoringName)
        member this.GetAvailableRefactorings() = this.GetAvailableRefactorings()
        member this.GetRefactoringByName(refactoringName) = this.GetRefactoringByName(refactoringName)
        member this.GetRefactoringsByCategory(category) = this.GetRefactoringsByCategory(category)
        member this.GetRefactoringsByTag(tag) = this.GetRefactoringsByTag(tag)
        member this.SuggestRefactorings(code) = this.SuggestRefactorings(code)
        member this.SuggestRefactoringsForFile(filePath) = this.SuggestRefactoringsForFile(filePath)
