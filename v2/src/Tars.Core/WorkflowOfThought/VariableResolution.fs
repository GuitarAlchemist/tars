namespace Tars.Core.WorkflowOfThought

open System
open System.Text.RegularExpressions

module VariableResolution =

  let private varRx = Regex(@"\$\{([^}]+)\}", RegexOptions.Compiled)

  let resolveString (ctx: ExecContext) (text: string) : Result<string,string> =
    let mutable err : string option = None

    let replaced =
      varRx.Replace(text, fun (m: Match) ->
        if err.IsSome then m.Value else
        let key = m.Groups.[1].Value.Trim()
        match ctx.Vars.TryFind key with
        | Some (:? string as s) -> s
        | Some _ ->
            err <- Some $"Variable '{key}' exists in Vars but is not a string."
            m.Value
        | None ->
            match ctx.Inputs.TryFind key with
            | Some s -> s
            | None ->
                err <- Some $"Unknown variable '{key}'."
                m.Value)

    match err with
    | Some e -> Error e
    | None -> Ok replaced

  let resolveToolArgs (ctx: ExecContext) (args: Map<string,obj>) : Result<Map<string,string>,string> =
    args
    |> Map.toList
    |> List.fold (fun acc (k,v) ->
        match acc with
        | Error _ -> acc
        | Ok m ->
            match v with
            | :? string as s ->
                match resolveString ctx s with
                | Ok v2 -> Ok (m.Add(k, v2))
                | Error e -> Error $"Arg '{k}': {e}"
            | other -> 
                // For v0, we just toString other types, assuming they don't contain variables
                Ok (m.Add(k, other.ToString()))
      ) (Ok Map.empty)
