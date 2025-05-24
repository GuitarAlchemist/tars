namespace TarsEngine.FSharp.Core.CodeGen

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of ICodeGenerator for various languages.
/// </summary>
type CodeGenerator(logger: ILogger<CodeGenerator>, templates: CodeGenerationTemplate list) =
    
    /// <summary>
    /// Gets the language supported by this generator.
    /// </summary>
    member _.Language = "csharp"
    
    /// <summary>
    /// Gets all available templates.
    /// </summary>
    /// <returns>The list of available templates.</returns>
    member _.GetAvailableTemplates() =
        task {
            return templates
        }
    
    /// <summary>
    /// Gets a template by name.
    /// </summary>
    /// <param name="templateName">The name of the template to get.</param>
    /// <returns>The template, if found.</returns>
    member _.GetTemplateByName(templateName: string) =
        task {
            return templates |> List.tryFind (fun t -> t.Name.Equals(templateName, StringComparison.OrdinalIgnoreCase))
        }
    
    /// <summary>
    /// Gets templates by category.
    /// </summary>
    /// <param name="category">The category of templates to get.</param>
    /// <returns>The list of templates in the category.</returns>
    member _.GetTemplatesByCategory(category: string) =
        task {
            return templates |> List.filter (fun t -> t.Category.Equals(category, StringComparison.OrdinalIgnoreCase))
        }
    
    /// <summary>
    /// Gets templates by tag.
    /// </summary>
    /// <param name="tag">The tag of templates to get.</param>
    /// <returns>The list of templates with the tag.</returns>
    member _.GetTemplatesByTag(tag: string) =
        task {
            return templates |> List.filter (fun t -> t.Tags |> List.exists (fun tg -> tg.Equals(tag, StringComparison.OrdinalIgnoreCase)))
        }
    
    /// <summary>
    /// Generates code from a template.
    /// </summary>
    /// <param name="template">The template to use.</param>
    /// <param name="placeholderValues">The values for placeholders.</param>
    /// <returns>The code generation result.</returns>
    member _.GenerateCode(template: CodeGenerationTemplate, placeholderValues: Map<string, string>) =
        try
            logger.LogInformation("Generating code from template: {TemplateName}", template.Name)
            
            // Replace placeholders in the template
            let mutable generatedCode = template.Content
            
            for placeholder in template.Placeholders do
                let placeholderName = placeholder.Key
                let placeholderDescription = placeholder.Value
                
                // Get the value for the placeholder
                match placeholderValues.TryFind placeholderName with
                | Some value ->
                    // Replace the placeholder with the value
                    generatedCode <- generatedCode.Replace($"${{{placeholderName}}}", value)
                | None ->
                    logger.LogWarning("No value provided for placeholder: {PlaceholderName}", placeholderName)
            
            // Create the result
            {
                GeneratedCode = generatedCode
                Template = template
                PlaceholderValues = placeholderValues
                AdditionalInfo = Map.empty
            }
        with
        | ex ->
            logger.LogError(ex, "Error generating code from template: {TemplateName}", template.Name)
            {
                GeneratedCode = $"Error generating code: {ex.Message}"
                Template = template
                PlaceholderValues = placeholderValues
                AdditionalInfo = Map.empty
            }
    
    /// <summary>
    /// Generates code from a template name.
    /// </summary>
    /// <param name="templateName">The name of the template to use.</param>
    /// <param name="placeholderValues">The values for placeholders.</param>
    /// <returns>The code generation result.</returns>
    member this.GenerateCodeFromTemplate(templateName: string, placeholderValues: Map<string, string>) =
        task {
            try
                logger.LogInformation("Generating code from template name: {TemplateName}", templateName)
                
                // Get the template
                let! template = this.GetTemplateByName(templateName)
                
                // Generate code from the template
                match template with
                | Some t ->
                    return this.GenerateCode(t, placeholderValues)
                | None ->
                    logger.LogError("Template not found: {TemplateName}", templateName)
                    return {
                        GeneratedCode = $"Error generating code: Template not found: {templateName}"
                        Template = {
                            Name = templateName
                            Description = "Template not found"
                            Content = ""
                            Language = ""
                            Category = ""
                            Tags = []
                            Placeholders = Map.empty
                            AdditionalInfo = Map.empty
                        }
                        PlaceholderValues = placeholderValues
                        AdditionalInfo = Map.empty
                    }
            with
            | ex ->
                logger.LogError(ex, "Error generating code from template name: {TemplateName}", templateName)
                return {
                    GeneratedCode = $"Error generating code: {ex.Message}"
                    Template = {
                        Name = templateName
                        Description = "Error generating code"
                        Content = ""
                        Language = ""
                        Category = ""
                        Tags = []
                        Placeholders = Map.empty
                        AdditionalInfo = Map.empty
                    }
                    PlaceholderValues = placeholderValues
                    AdditionalInfo = Map.empty
                }
        }
    
    interface ICodeGenerator with
        member this.Language = this.Language
        member this.GenerateCode(template, placeholderValues) = this.GenerateCode(template, placeholderValues)
        member this.GenerateCodeFromTemplate(templateName, placeholderValues) = this.GenerateCodeFromTemplate(templateName, placeholderValues)
        member this.GetAvailableTemplates() = this.GetAvailableTemplates()
        member this.GetTemplateByName(templateName) = this.GetTemplateByName(templateName)
        member this.GetTemplatesByCategory(category) = this.GetTemplatesByCategory(category)
        member this.GetTemplatesByTag(tag) = this.GetTemplatesByTag(tag)
