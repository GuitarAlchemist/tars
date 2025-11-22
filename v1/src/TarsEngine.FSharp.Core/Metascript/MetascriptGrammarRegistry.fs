namespace TarsEngine.FSharp.Core.Metascript

open System
open System.IO

module MetascriptGrammarRegistry =

    let rec private findSpecsRoot (start: string) : string option =
        let candidate = Path.Combine(start, "specs", "metascripts")
        if Directory.Exists(candidate) then Some candidate
        else
            let parent = Directory.GetParent(start)
            if isNull parent then None else findSpecsRoot parent.FullName

    let private defaultRootLazy =
        lazy (
            match Environment.GetEnvironmentVariable("TARS_METASCRIPT_SPEC_ROOT") with
            | null | "" ->
                let cwd = Directory.GetCurrentDirectory()
                match findSpecsRoot cwd with
                | Some root -> root
                | None -> cwd
            | overrideRoot -> Path.GetFullPath(overrideRoot)
        )

    let private specsLazy =
        lazy (
            let root = defaultRootLazy.Value
            if Directory.Exists(root) then
                Directory.EnumerateFiles(root, "*.md", SearchOption.AllDirectories)
                |> Seq.map MetascriptSpecLoader.loadFromFile
                |> Seq.map (fun spec -> spec.Id.ToLowerInvariant(), spec)
                |> dict
            else
                dict []
        )

    let defaultRoot() = defaultRootLazy.Value

    let tryGetSpec (specId: string) =
        if String.IsNullOrWhiteSpace(specId) then None
        else
            let specs = specsLazy.Value
            match specs.TryGetValue(specId.ToLowerInvariant()) with
            | true, spec -> Some spec
            | _ -> None

    let listSpecs () = specsLazy.Value.Values |> Seq.toList
