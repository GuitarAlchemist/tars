namespace TarsEngine.FSharp.Core.Services

open System
open System.Collections.Generic
open System.IO
open System.Threading
open System.Threading.Tasks
open TarsEngine.DSL.Ast

/// <summary>
/// Interface for the TARS engine service.
/// </summary>
type ITarsEngineService =
    /// <summary>
    /// Generates an improvement.
    /// </summary>
    /// <param name="cancellationToken">The cancellation token.</param>
    /// <returns>The improvement result.</returns>
    abstract member GenerateImprovement : cancellationToken:CancellationToken -> Task<ImprovementResult>
    
    /// <summary>
    /// Loads a checkpoint.
    /// </summary>
    /// <returns>Whether the checkpoint was loaded successfully.</returns>
    abstract member LoadCheckpoint : unit -> Task<bool>
    
    /// <summary>
    /// Saves a checkpoint.
    /// </summary>
    /// <returns>A task representing the asynchronous operation.</returns>
    abstract member SaveCheckpoint : unit -> Task
    
    /// <summary>
    /// Processes an uploaded file.
    /// </summary>
    /// <param name="fileStream">The file stream.</param>
    /// <param name="fileName">The file name.</param>
    /// <returns>The result of processing the file.</returns>
    abstract member ProcessUploadedFile : fileStream:Stream * fileName:string -> Task<string>
    
    /// <summary>
    /// Processes a prompt.
    /// </summary>
    /// <param name="prompt">The prompt to process.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    abstract member ProcessPrompt : prompt:string -> Task

/// <summary>
/// Interface for the TARS engine.
/// </summary>
type ITarsEngine =
    /// <summary>
    /// Generates an improvement.
    /// </summary>
    /// <returns>The time, capability, and confidence of the improvement.</returns>
    abstract member GenerateImprovement : unit -> Task<DateTime * string * float>
    
    /// <summary>
    /// Resumes the last session.
    /// </summary>
    /// <returns>A task representing the asynchronous operation.</returns>
    abstract member ResumeLastSession : unit -> Task

/// <summary>
/// Interface for the metascript service.
/// </summary>
type IMetascriptService =
    /// <summary>
    /// Executes a metascript asynchronously.
    /// </summary>
    /// <param name="metascript">The metascript to execute.</param>
    /// <param name="context">The context for the metascript execution.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the execution result.</returns>
    abstract member ExecuteMetascriptAsync : metascript:string * ?context:MetascriptContext -> Task<MetascriptExecutionResult<PropertyValue>>
    
    /// <summary>
    /// Parses a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to parse.</param>
    /// <returns>The parsed TARS program.</returns>
    abstract member ParseMetascript : metascript:string -> TarsProgram
    
    /// <summary>
    /// Executes a TARS program asynchronously.
    /// </summary>
    /// <param name="program">The TARS program to execute.</param>
    /// <param name="context">The context for the program execution.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the execution result.</returns>
    abstract member ExecuteTarsProgramAsync : program:TarsProgram * ?context:MetascriptContext -> Task<MetascriptExecutionResult<PropertyValue>>
    
    /// <summary>
    /// Gets the available DSL functions.
    /// </summary>
    /// <returns>A dictionary of function names to descriptions.</returns>
    abstract member GetAvailableDslFunctions : unit -> IDictionary<string, string>
    
    /// <summary>
    /// Validates a metascript without executing it.
    /// </summary>
    /// <param name="metascript">The metascript to validate.</param>
    /// <returns>A result indicating whether the metascript is valid, with error messages if not.</returns>
    abstract member ValidateMetascript : metascript:string -> bool

/// <summary>
/// Interface for the metascript generator service.
/// </summary>
type IMetascriptGeneratorService =
    /// <summary>
    /// Generates a metascript.
    /// </summary>
    /// <param name="templateName">The name of the template to use.</param>
    /// <param name="parameters">The parameters for the template.</param>
    /// <returns>The generated metascript.</returns>
    abstract member GenerateMetascriptAsync : templateName:string * parameters:Map<string, string> -> Task<GeneratedMetascript>
    
    /// <summary>
    /// Executes a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to execute.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>The execution result.</returns>
    abstract member ExecuteMetascriptAsync : metascript:GeneratedMetascript * ?options:Map<string, string> -> Task<MetascriptExecutionResult<obj>>
    
    /// <summary>
    /// Gets all available templates.
    /// </summary>
    /// <param name="language">Optional language filter.</param>
    /// <returns>The list of available templates.</returns>
    abstract member GetTemplatesAsync : ?language:string -> Task<MetascriptTemplate list>
    
    /// <summary>
    /// Gets a template by name.
    /// </summary>
    /// <param name="templateName">The name of the template to get.</param>
    /// <returns>The template, if found.</returns>
    abstract member GetTemplateByNameAsync : templateName:string -> Task<MetascriptTemplate option>
    
    /// <summary>
    /// Creates a new template.
    /// </summary>
    /// <param name="template">The template to create.</param>
    /// <returns>Whether the template was created successfully.</returns>
    abstract member CreateTemplateAsync : template:MetascriptTemplate -> Task<bool>
    
    /// <summary>
    /// Updates a template.
    /// </summary>
    /// <param name="template">The template to update.</param>
    /// <returns>Whether the template was updated successfully.</returns>
    abstract member UpdateTemplateAsync : template:MetascriptTemplate -> Task<bool>
    
    /// <summary>
    /// Deletes a template.
    /// </summary>
    /// <param name="templateName">The name of the template to delete.</param>
    /// <returns>Whether the template was deleted successfully.</returns>
    abstract member DeleteTemplateAsync : templateName:string -> Task<bool>
    
    /// <summary>
    /// Gets templates by category.
    /// </summary>
    /// <param name="category">The category of templates to get.</param>
    /// <returns>The list of templates in the category.</returns>
    abstract member GetTemplatesByCategoryAsync : category:string -> Task<MetascriptTemplate list>
    
    /// <summary>
    /// Gets templates by tag.
    /// </summary>
    /// <param name="tag">The tag of templates to get.</param>
    /// <returns>The list of templates with the tag.</returns>
    abstract member GetTemplatesByTagAsync : tag:string -> Task<MetascriptTemplate list>
