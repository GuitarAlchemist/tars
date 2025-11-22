namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text.RegularExpressions
open ImprovementTypes

/// Deterministic source-to-source code transformations for known patterns.
module CodeTransforms =

    type TransformationOutcome =
        | NotSupported
        | NoChange
        | Applied of updatedContent: string * AppliedImprovement list

    let private createApplied filePath (improvement: ImprovementPattern) original improved lineNumber =
        {
            FilePath = filePath
            PatternId = improvement.Name // Using Name as ID for now
            PatternName = improvement.Name
            LineNumber = lineNumber
            OriginalCode = original
            ImprovedCode = improved
            AppliedAt = DateTime.UtcNow
        }

    let private normalizeTodoComments filePath improvement (content: string) =
        // Match comment lines starting with TODO/FIXME/HACK/XXX that have not been normalized yet.
        let regex = Regex(@"(^\s*//+\s*)(TODO|FIXME|HACK|XXX)(?!\[tracked-)(.*)$", RegexOptions.Multiline ||| RegexOptions.IgnoreCase)
        let applied = ResizeArray<AppliedImprovement>()

        let updated =
            regex.Replace(content, MatchEvaluator(fun m ->
                let prefix = m.Groups.[1].Value
                let suffix = m.Groups.[3].Value
                let identifier = Guid.NewGuid().ToString("N").Substring(0, 8)
                let improved = prefix + "TODO[tracked-" + identifier + "]" + suffix
                applied.Add(createApplied filePath improvement m.Value improved None)
                improved))

        if applied.Count = 0 then
            NoChange
        else
            Applied(updated, applied |> Seq.toList)

    let private fixEmptyCatchBlocks filePath improvement (content: string) =
        // Capture indentation and the catch head (with optional arguments), replace empty bodies with throw.
        let regex = Regex(@"(?<indent>^\s*)(?<head>catch(?:\s*\([^)]*\))?)\s*\{\s*\}", RegexOptions.Multiline)
        let applied = ResizeArray<AppliedImprovement>()

        let updated =
            regex.Replace(content, MatchEvaluator(fun m ->
                let indent = m.Groups.["indent"].Value
                let head = m.Groups.["head"].Value.TrimEnd()
                let improved = indent + head + " { throw; }"
                applied.Add(createApplied filePath improvement m.Value improved None)
                improved))

        if applied.Count = 0 then
            NoChange
        else
            Applied(updated, applied |> Seq.toList)

    /// Try to apply a deterministic transformation for the given improvement pattern.
    let apply (filePath: string) (improvement: ImprovementPattern) (content: string) =
        let name = improvement.Name.Trim().ToLowerInvariant()
        let extension = Path.GetExtension(filePath).ToLowerInvariant()

        match name, extension with
        | "todo comments", _ -> normalizeTodoComments filePath improvement content
        | "empty catch blocks", ".cs" -> fixEmptyCatchBlocks filePath improvement content
        | _ -> NotSupported
