namespace Tars.Evolution

open System.IO
open System.Text.Json

/// The self-improvement backlog (ADR 0002 D5): a curated list of genuine
/// capability-gap entries, each a (failing test, source file) pair the hermetic
/// self-hosting gate can attempt. The loop is mechanically solid (best-of-N +
/// repair + multi-edit); its value is now gated by the *supply* of genuine red
/// tests — so that supply is declared as data here, growing by curation, not code.
module SelfImproveBacklog =

    /// One backlog item: drive `TargetTest` (currently failing) to green by editing
    /// `TargetFile`, verified by `TestProject`. Paths are relative to the repo root
    /// passed to the gate (`--repo`).
    type BacklogEntry =
        { Id: string
          TargetTest: string
          TargetFile: string
          TestProject: string
          Rationale: string }

    /// Parse the backlog JSON (an array of `{id, target_test, target_file,
    /// test_project, rationale?}`) into typed records. Entries missing a required
    /// field are skipped (a malformed line never silently mis-targets the gate).
    /// Pure; Error only on JSON that is not an array / not parseable.
    let parse (json: string) : Result<BacklogEntry list, string> =
        try
            use doc = JsonDocument.Parse(json)
            let root = doc.RootElement
            if root.ValueKind <> JsonValueKind.Array then
                Result.Error "backlog must be a JSON array"
            else
                let str (el: JsonElement) (name: string) =
                    match el.TryGetProperty name with
                    | true, v when v.ValueKind = JsonValueKind.String -> Some(v.GetString())
                    | _ -> None
                let entries =
                    root.EnumerateArray()
                    |> Seq.choose (fun el ->
                        match
                            str el "id", str el "target_test", str el "target_file", str el "test_project"
                        with
                        | Some id, Some test, Some file, Some proj ->
                            Some
                                { Id = id
                                  TargetTest = test
                                  TargetFile = file
                                  TestProject = proj
                                  Rationale = str el "rationale" |> Option.defaultValue "" }
                        | _ -> None)
                    |> Seq.toList
                Result.Ok entries
        with ex ->
            Result.Error(sprintf "backlog parse error: %s" ex.Message)

    /// Load and parse a backlog file. Error if the file is absent or malformed.
    let load (path: string) : Result<BacklogEntry list, string> =
        if File.Exists path then
            parse (File.ReadAllText path)
        else
            Result.Error(sprintf "backlog file not found: %s" path)
