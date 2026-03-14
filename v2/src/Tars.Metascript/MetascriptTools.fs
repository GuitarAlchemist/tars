namespace Tars.Metascript

open System
open System.Threading.Tasks
open Tars.Core

module MetascriptTools =

    /// Creates tools for macro management given a registry
    let getTools (registry: IMacroRegistry) : Tool list =
        [ { Name = "register_macro"
            Description =
              "Register a new workflow macro from JSON definition. Input must be a valid Metascript Workflow JSON."
            Version = "1.0.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute =
              fun (input: string) ->
                  async {
                      try
                          match Parser.parseJson input with
                          | Parser.Success workflow ->
                              do! registry.Register(workflow) |> Async.AwaitTask
                              return Ok $"Macro '{workflow.Name}' registered successfully (v{workflow.Version})"
                          | Parser.Failure error -> return Error $"Failed to parse workflow: {error}"
                      with ex ->
                          return Error $"Error registering macro: {ex.Message}"
                  } }

          { Name = "list_macros"
            Description = "List all registered macros."
            Version = "1.0.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute =
              fun (_: string) ->
                  async {
                      try
                          let! macros = registry.List() |> Async.AwaitTask

                          let summary =
                              macros
                              |> List.map (fun m -> $"- {m.Name} (v{m.Version}): {m.Description}")
                              |> String.concat "\n"

                          if String.IsNullOrWhiteSpace(summary) then
                              return Ok "No macros registered."
                          else
                              return Ok $"Registered Macros:\n{summary}"
                      with ex ->
                          return Error $"Error listing macros: {ex.Message}"
                  } }

          { Name = "get_macro"
            Description = "Get the definition of a registered macro by name."
            Version = "1.0.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute =
              fun (name: string) ->
                  async {
                      try
                          let! macroOpt = registry.Get(name.Trim()) |> Async.AwaitTask

                          match macroOpt with
                          | Some workflow ->
                              let json = Parser.toJson workflow
                              return Ok json
                          | None -> return Error $"Macro '{name}' not found."
                      with ex ->
                          return Error $"Error retrieving macro: {ex.Message}"
                  } } ]
