/// AI Persona System - Core Types
/// Enables structured role-based prompting using RTFD pattern
module Tars.Core.Persona

open System
open System.Text.Json.Serialization

// ============================================================================
// Output Format
// ============================================================================

/// Defines how LLM output should be structured
[<JsonConverter(typeof<JsonStringEnumConverter>)>]
type OutputFormat =
    | Markdown
    | JSON
    | Table
    | BulletPoints
    | Prose
    | Custom of string

// ============================================================================
// Few-Shot Examples
// ============================================================================

/// Example input/output pair for few-shot prompting
[<CLIMutable>]
type FewShotExample = {
    [<JsonPropertyName("input")>]
    Input: string
    [<JsonPropertyName("output")>]
    Output: string
}

// ============================================================================
// Persona Definition
// ============================================================================

/// An AI persona with role definition and behavioral constraints
[<CLIMutable>]
type Persona = {
    [<JsonPropertyName("id")>]
    Id: string
    
    [<JsonPropertyName("name")>]
    Name: string
    
    [<JsonPropertyName("role")>]
    Role: string  // "Act as an expert..."
    
    [<JsonPropertyName("description")>]
    Description: string option
    
    [<JsonPropertyName("defaultFormat")>]
    DefaultFormat: OutputFormat
    
    [<JsonPropertyName("constraints")>]
    Constraints: string list
    
    [<JsonPropertyName("temperature")>]
    Temperature: float option
    
    [<JsonPropertyName("examples")>]
    Examples: FewShotExample list
    
    [<JsonPropertyName("tags")>]
    Tags: string list
}

// ============================================================================
// RTFD Prompt Structure
// ============================================================================

/// Role-Task-Format-Details prompt structure
type RtfdPrompt = {
    Persona: Persona
    Task: string
    Format: OutputFormat option     // Override persona default
    Details: string option          // Additional context/constraints
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a minimal persona with just name and role
let createPersona id name role =
    { Id = id
      Name = name
      Role = role
      Description = None
      DefaultFormat = Markdown
      Constraints = []
      Temperature = None
      Examples = []
      Tags = [] }

/// Generate format instruction string
let formatInstruction (format: OutputFormat) =
    match format with
    | Markdown -> "Format your response in clean markdown with headers and code blocks where appropriate."
    | JSON -> "Return your response as valid JSON only. No additional text or explanation."
    | Table -> "Present the output as a markdown table with clear column headers."
    | BulletPoints -> "Structure your response using bullet points for clarity."
    | Prose -> ""
    | Custom s -> s

/// Build complete prompt from RTFD structure
let buildRtfdPrompt (rtfd: RtfdPrompt) : string =
    let format = rtfd.Format |> Option.defaultValue rtfd.Persona.DefaultFormat
    let formatInstr = formatInstruction format
    let constraintsSection =
        match rtfd.Persona.Constraints with
        | [] -> ""
        | cs -> "\n\nConstraints:\n" + (cs |> List.map (sprintf "- %s") |> String.concat "\n")
    let examplesSection =
        match rtfd.Persona.Examples with
        | [] -> ""
        | exs ->
            let exampleText =
                exs
                |> List.mapi (fun i ex -> $"Example {i+1}:\nInput: {ex.Input}\nOutput: {ex.Output}")
                |> String.concat "\n\n"
            $"\n\nExamples:\n{exampleText}"
    let detailsSection =
        rtfd.Details |> Option.map (sprintf "\n\nAdditional Context:\n%s") |> Option.defaultValue ""
    
    $"""{rtfd.Persona.Role}

Task: {rtfd.Task}
{formatInstr}{constraintsSection}{examplesSection}{detailsSection}"""

// ============================================================================
// Built-in Personas
// ============================================================================

module BuiltIn =
    let codeReviewer =
        { Id = "code-reviewer"
          Name = "Code Reviewer"
          Role = "Act as an expert code reviewer with a focus on security, performance, and maintainability. You identify bugs, suggest improvements, and ensure code follows best practices."
          Description = Some "Reviews code for quality and security issues"
          DefaultFormat = Markdown
          Constraints = [
              "Always explain the reasoning behind each suggestion"
              "Prioritize security vulnerabilities over style issues"
              "Be constructive and educational in feedback"
          ]
          Temperature = Some 0.3
          Examples = []
          Tags = ["development"; "quality"] }

    let documentationWriter =
        { Id = "documentation-writer"
          Name = "Documentation Writer"
          Role = "Act as a technical documentation specialist. You write clear, concise documentation that is accessible to both beginners and experts."
          Description = Some "Creates technical documentation"
          DefaultFormat = Markdown
          Constraints = [
              "Use simple language; avoid jargon unless necessary"
              "Include code examples where helpful"
              "Structure content with clear headings"
          ]
          Temperature = Some 0.5
          Examples = []
          Tags = ["documentation"; "writing"] }

    let testEngineer =
        { Id = "test-engineer"
          Name = "Test Engineer"
          Role = "Act as a QA specialist focused on test design and coverage. You create comprehensive test cases that catch edge cases and ensure reliability."
          Description = Some "Designs tests and identifies edge cases"
          DefaultFormat = BulletPoints
          Constraints = [
              "Consider boundary conditions and error states"
              "Include both positive and negative test cases"
              "Prioritize critical paths first"
          ]
          Temperature = Some 0.4
          Examples = []
          Tags = ["testing"; "quality"] }

    let promptEngineer =
        { Id = "prompt-engineer"
          Name = "Prompt Engineer"
          Role = "Act as an AI prompt engineering expert. You optimize prompts for clarity, specificity, and effectiveness while avoiding common pitfalls."
          Description = Some "Optimizes and refines prompts"
          DefaultFormat = Markdown
          Constraints = [
              "Explain changes and their expected impact"
              "Consider token efficiency"
              "Test suggestions against common failure modes"
          ]
          Temperature = Some 0.6
          Examples = []
          Tags = ["ai"; "optimization"] }

    let projectManager =
        { Id = "project-manager"
          Name = "Project Manager"
          Role = "Act as an experienced agile project manager. You break down complex goals into actionable tasks, identify risks, and create realistic timelines."
          Description = Some "Plans and organizes work"
          DefaultFormat = Table
          Constraints = [
              "Be realistic about estimates"
              "Identify dependencies explicitly"
              "Include risk mitigation strategies"
          ]
          Temperature = Some 0.4
          Examples = []
          Tags = ["planning"; "management"] }

    /// All built-in personas
    let all = [
        codeReviewer
        documentationWriter
        testEngineer
        promptEngineer
        projectManager
    ]
