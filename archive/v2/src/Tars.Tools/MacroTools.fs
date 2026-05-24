namespace Tars.Tools

open System
open Tars.Core
open Tars.Metascript

/// Macro management tools - creates Tool records for macro registry operations
module MacroTools =

    /// Creates tools for macro management given a registry
    let getTools (registry: IMacroRegistry) : Tool list =
        let registerTool: Tool = 
            Tool.InternalCreateMinimal(
                "register_macro",
                "Register a new workflow macro from JSON definition. Input must be a valid Metascript Workflow JSON.",
                fun (input: string) ->
                  async {
                      try
                          match Parser.parseJson input with
                          | Parser.Success (workflow: Domain.Workflow) ->
                              do! registry.Register(workflow) |> Async.AwaitTask
                              return Result.Ok $"Macro '{workflow.Name}' registered successfully (v{workflow.Version})"
                          | Parser.ParseError (line, col, msg) -> 
                              return Result.Error $"Parse error at line {line}, column {col}: {msg}"
                          | Parser.ValidationError errors ->
                              let errMsg = String.concat "; " errors
                              return Result.Error $"Validation errors: {errMsg}"
                      with ex ->
                          return Result.Error $"Error registering macro: {ex.Message}"
                  }
            )

        let listTool: Tool = 
            Tool.InternalCreateMinimal(
                "list_macros",
                "List all registered macros.",
                fun (_: string) ->
                  async {
                      try
                          let! macros = registry.List() |> Async.AwaitTask

                          let summary =
                              macros
                              |> List.map (fun (m: Domain.Workflow) -> $"- {m.Name} (v{m.Version}): {m.Description}")
                              |> String.concat "\n"

                          if String.IsNullOrWhiteSpace(summary) then
                              return Result.Ok "No macros registered."
                          else
                              return Result.Ok $"Registered Macros:\n{summary}"
                      with ex ->
                          return Result.Error $"Error listing macros: {ex.Message}"
                  }
            )

        let getTool: Tool = 
            Tool.InternalCreateMinimal(
                "get_macro",
                "Get the definition of a registered macro by name.",
                fun (name: string) ->
                  async {
                      try
                          let! macroOpt = registry.Get(name.Trim()) |> Async.AwaitTask

                          match macroOpt with
                          | Some (workflow: Domain.Workflow) ->
                              let json = Parser.toJson workflow
                              return Result.Ok json
                          | None -> 
                              return Result.Error $"Macro '{name}' not found."
                      with ex ->
                          return Result.Error $"Error retrieving macro: {ex.Message}"
                  }
            )

        [ registerTool; listTool; getTool ]
