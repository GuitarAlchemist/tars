namespace Tars.Core.HybridBrain

open System
open System.IO
open System.Threading.Tasks
open System.Text.RegularExpressions
open Tars.Core

/// Type alias to avoid collision with System.Action
type private HybridAction = Tars.Core.HybridBrain.Action

/// Real implementations of ActionExecutors for the HybridBrain
module ActionExecutor =

    /// Result of an action execution
    type ActionResult =
        { Success: bool
          Output: obj option
          Message: string
          Duration: TimeSpan }

    /// File-based refactoring operations
    module FileRefactoring =

        /// Extract lines from a file
        let readLines (filePath: string) =
            if File.Exists(filePath) then
                File.ReadAllLines(filePath) |> Array.toList |> Some
            else
                None

        /// Write lines back to a file
        let writeLines (filePath: string) (lines: string list) =
            File.WriteAllLines(filePath, lines |> List.toArray)

        /// Extract a function from specified lines
        let extractFunction (filePath: string) (functionName: string) (startLine: int) (endLine: int) =
            match readLines filePath with
            | None -> FSharp.Core.Error $"File not found: {filePath}"
            | Some lines ->
                if startLine < 1 || endLine > lines.Length || startLine > endLine then
                    FSharp.Core.Error $"Invalid line range: {startLine}-{endLine} (file has {lines.Length} lines)"
                else
                    // Get the lines to extract (1-indexed to 0-indexed)
                    let extractedLines = lines.[startLine - 1 .. endLine - 1]

                    // Find MINIMUM indentation across all non-empty lines in the block
                    let minIndent =
                        extractedLines
                        |> List.filter (not << String.IsNullOrWhiteSpace)
                        |> List.map (fun s -> s.Length - s.TrimStart().Length)
                        |> function
                            | [] -> 0
                            | indents -> List.min indents

                    // Indentation for the new function itself (match the start line)
                    let callIndentCount =
                        lines.[startLine - 1].Length - lines.[startLine - 1].TrimStart().Length

                    let callIndent = String.replicate callIndentCount " "

                    // Create the new function with slightly more indentation than the call
                    // Create the new function with 4 spaces relative to callIndent
                    let newFunction =
                        [ $"{callIndent}let {functionName} () ="
                          yield!
                              extractedLines
                              |> List.map (fun l ->
                                  let stripped =
                                      if l.Length >= minIndent then
                                          l.Substring(minIndent).TrimStart()
                                      else
                                          l.TrimStart()

                                  $"{callIndent}    {stripped}") ]

                    // Replace original lines with function call
                    let beforeLines = if startLine > 1 then lines.[0 .. startLine - 2] else []
                    let afterLines = if endLine < lines.Length then lines.[endLine..] else []
                    let callLine = $"{callIndent}{functionName} ()"

                    // Combine: before + new function + call + after
                    let newLines = beforeLines @ newFunction @ [ callLine ] @ afterLines

                    writeLines filePath newLines
                    FSharp.Core.Ok $"Extracted lines {startLine}-{endLine} into function '{functionName}'"

        /// Remove lines from a file (dead code removal)
        let removeLines (filePath: string) (startLine: int) (endLine: int) =
            match readLines filePath with
            | None -> FSharp.Core.Error $"File not found: {filePath}"
            | Some lines ->
                if startLine < 1 || endLine > lines.Length || startLine > endLine then
                    FSharp.Core.Error $"Invalid line range: {startLine}-{endLine}"
                else
                    let beforeLines = if startLine > 1 then lines.[0 .. startLine - 2] else []
                    let afterLines = if endLine < lines.Length then lines.[endLine..] else []
                    let newLines = beforeLines @ afterLines
                    writeLines filePath newLines
                    FSharp.Core.Ok $"Removed lines {startLine}-{endLine}"

        /// Add documentation comment above a line
        let addDocumentation (filePath: string) (lineNumber: int) (docText: string) =
            match readLines filePath with
            | None -> FSharp.Core.Error $"File not found: {filePath}"
            | Some lines ->
                if lineNumber < 1 || lineNumber > lines.Length then
                    FSharp.Core.Error $"Invalid line number: {lineNumber}"
                else
                    // Check if already documented
                    let alreadyDocumented =
                        if lineNumber > 1 then
                            lines.[lineNumber - 2].Trim().StartsWith("///")
                        else
                            false

                    if alreadyDocumented then
                        FSharp.Core.Ok $"Symbol at line {lineNumber} is already documented"
                    else
                        let indent = lines.[lineNumber - 1] |> fun s -> s.Length - s.TrimStart().Length
                        let docLine = String.replicate indent " " + $"/// <summary>{docText}</summary>"

                        let beforeLines = if lineNumber > 1 then lines.[0 .. lineNumber - 2] else []
                        let afterLines = lines.[lineNumber - 1 ..]
                        let newLines = beforeLines @ [ docLine ] @ afterLines

                        writeLines filePath newLines
                        FSharp.Core.Ok $"Added documentation at line {lineNumber}"

        let private isSafeInlineExpression (expr: string) =
            let trimmed = expr.Trim()

            if String.IsNullOrWhiteSpace(trimmed) then
                false
            else
                let keywordPattern =
                    Regex(@"\b(match|if|then|else|for|while|fun|try|with|let|do|yield)\b", RegexOptions.IgnoreCase)

                if keywordPattern.IsMatch(trimmed) then
                    false
                elif trimmed.Contains("->") || trimmed.Contains("|") then
                    false
                elif trimmed.Contains(";") then
                    false
                else
                    true

        /// Inline a simple value usage on a specific line
        let inlineValue (filePath: string) (name: string) (lineNumber: int) =
            match readLines filePath with
            | None -> FSharp.Core.Error $"File not found: {filePath}"
            | Some lines ->
                if lineNumber < 1 || lineNumber > lines.Length then
                    FSharp.Core.Error $"Invalid line number: {lineNumber}"
                else
                    let bindingRegex =
                        Regex(@"^(\s*)let\s+(?:mutable\s+)?(?<name>\w+)\s*=\s*(?<expr>.+)$")

                    let binding =
                        lines
                        |> List.mapi (fun idx line -> idx, line)
                        |> List.tryPick (fun (idx, line) ->
                            let m = bindingRegex.Match(line)

                            if
                                m.Success
                                && String.Equals(m.Groups.["name"].Value, name, StringComparison.Ordinal)
                            then
                                Some(idx, m.Groups.["expr"].Value)
                            else
                                None)

                    match binding with
                    | None -> FSharp.Core.Error $"No binding found for '{name}'"
                    | Some(bindingIndex, expr) ->
                        if bindingIndex = lineNumber - 1 then
                            FSharp.Core.Error $"Inline target is the binding line for '{name}'"
                        elif not (isSafeInlineExpression expr) then
                            FSharp.Core.Error $"Inline expression for '{name}' is not safe to inline"
                        else
                            let escaped = Regex.Escape(name)
                            let wordPattern = Regex($@"\b{escaped}\b")
                            let targetLine = lines.[lineNumber - 1]
                            let targetMatches = wordPattern.Matches(targetLine).Count

                            if targetMatches <> 1 then
                                FSharp.Core.Error $"Expected exactly one '{name}' occurrence at line {lineNumber}"
                            else
                                let otherUses =
                                    lines
                                    |> List.mapi (fun idx line ->
                                        if idx = bindingIndex then
                                            0
                                        else
                                            wordPattern.Matches(line).Count)
                                    |> List.sum

                                if otherUses <> 1 then
                                    FSharp.Core.Error $"Inline aborted: '{name}' has {otherUses} references"
                                else
                                    let replaced = wordPattern.Replace(targetLine, expr)

                                    let newLines =
                                        lines
                                        |> List.mapi (fun idx line ->
                                            let updated = if idx = lineNumber - 1 then replaced else line
                                            idx, updated)
                                        |> List.choose (fun (idx, line) ->
                                            if idx = bindingIndex then None else Some line)

                                    writeLines filePath newLines
                                    FSharp.Core.Ok $"Inlined '{name}' at line {lineNumber}"

        /// Simplify a basic boolean match into an if/then/else expression
        let simplifyPattern (filePath: string) (lineNumber: int) =
            match readLines filePath with
            | None -> FSharp.Core.Error $"File not found: {filePath}"
            | Some lines ->
                if lineNumber < 1 || lineNumber > lines.Length then
                    FSharp.Core.Error $"Invalid line number: {lineNumber}"
                else
                    let matchRegex = Regex(@"^(\s*)match\s+(.+)\s+with\s*$")
                    let matchLine = lines.[lineNumber - 1]
                    let matchMatch = matchRegex.Match(matchLine)

                    if not matchMatch.Success then
                        FSharp.Core.Error $"Line {lineNumber} is not a simple match expression"
                    else
                        let indent = matchMatch.Groups.[1].Value
                        let expr = matchMatch.Groups.[2].Value.Trim()

                        let mutable trueCase = None
                        let mutable falseCase = None

                        let maxLookahead = Math.Min(lines.Length - 1, lineNumber + 5)

                        for idx in lineNumber..maxLookahead do
                            let trimmed = lines.[idx].Trim()

                            if trimmed.StartsWith("|") then
                                let trueMatch = Regex(@"^\|\s*true\s*->\s*(.+)$").Match(trimmed)
                                let falseMatch = Regex(@"^\|\s*false\s*->\s*(.+)$").Match(trimmed)

                                if trueMatch.Success then
                                    trueCase <- Some(idx, trueMatch.Groups.[1].Value.Trim())
                                elif falseMatch.Success then
                                    falseCase <- Some(idx, falseMatch.Groups.[1].Value.Trim())

                        match trueCase, falseCase with
                        | Some(trueIdx, thenExpr), Some(falseIdx, elseExpr) ->
                            let replacement = $"{indent}if {expr} then {thenExpr} else {elseExpr}"

                            let newLines =
                                lines
                                |> List.mapi (fun idx line ->
                                    let updated = if idx = lineNumber - 1 then replacement else line
                                    idx, updated)
                                |> List.choose (fun (idx, line) ->
                                    if idx = trueIdx || idx = falseIdx then None else Some line)

                            writeLines filePath newLines
                            FSharp.Core.Ok $"Simplified match at line {lineNumber}"
                        | _ -> FSharp.Core.Error $"No simple boolean match found near line {lineNumber}"

        /// Rename a symbol throughout the file
        let renameSymbol (filePath: string) (oldName: string) (newName: string) =
            match readLines filePath with
            | None -> FSharp.Core.Error $"File not found: {filePath}"
            | Some lines ->
                let pattern = $@"\b{Regex.Escape(oldName)}\b"
                let mutable count = 0

                let newLines =
                    lines
                    |> List.map (fun line ->
                        let matches = Regex.Matches(line, pattern)

                        if matches.Count > 0 then
                            count <- count + matches.Count
                            Regex.Replace(line, pattern, newName)
                        else
                            line)

                if count = 0 then
                    FSharp.Core.Error $"Symbol '{oldName}' not found in {filePath}"
                else
                    writeLines filePath newLines
                    FSharp.Core.Ok $"Renamed '{oldName}' to '{newName}' ({count} occurrences)"

    /// Create a real executor for file-based refactoring
    let createFileRefactoringExecutor
        (targetFile: string)
        (constitution: AgentConstitution option)
        : HybridAction -> Task<Result<obj option, string>> =
        fun action ->
            task {
                try
                    // 1. Enforce Constitution if present
                    let validationResult =
                        match constitution with
                        | None -> FSharp.Core.Ok()
                        | Some c ->
                            // Map HybridAction to AgentAction
                            let agentAction =
                                match action with
                                | HybridAction.ExtractFunction(name, _, _) ->
                                    AgentAction.GenericAction("ExtractFunction", name)
                                | HybridAction.RemoveDeadCode(_, _) ->
                                    AgentAction.GenericAction("RemoveDeadCode", targetFile)
                                | HybridAction.AddDocumentation(_, _) -> AgentAction.WriteFile(targetFile)
                                | HybridAction.InlineValue(_, _) -> AgentAction.WriteFile(targetFile)
                                | HybridAction.SimplifyPattern(_) -> AgentAction.WriteFile(targetFile)
                                | HybridAction.RenameSymbol(oldName, newName) ->
                                    AgentAction.GenericAction("RenameSymbol", $"{oldName}->{newName}")
                                | _ -> AgentAction.GenericAction(action.GetType().Name, "Unknown")

                            match ContractEnforcement.validateAction c agentAction with
                            | FSharp.Core.Ok() -> FSharp.Core.Ok()
                            | FSharp.Core.Error violation ->
                                FSharp.Core.Error(sprintf "Constitution violation: %A" violation)

                    match validationResult with
                    | FSharp.Core.Error msg -> return FSharp.Core.Error msg
                    | FSharp.Core.Ok() ->

                        let startTime = DateTime.UtcNow

                        let result =
                            match action with
                            | HybridAction.ExtractFunction(name, startLine, endLine) ->
                                FileRefactoring.extractFunction targetFile name startLine endLine
                                |> Result.map (fun msg -> Some(box msg))

                            | HybridAction.RemoveDeadCode(startLine, endLine) ->
                                FileRefactoring.removeLines targetFile startLine endLine
                                |> Result.map (fun msg -> Some(box msg))

                            | HybridAction.AddDocumentation(docText, lineNumber) ->
                                FileRefactoring.addDocumentation targetFile lineNumber docText
                                |> Result.map (fun msg -> Some(box msg))

                            | HybridAction.NoOp -> FSharp.Core.Ok None

                            | HybridAction.InlineValue(name, line) ->
                                FileRefactoring.inlineValue targetFile name line
                                |> Result.map (fun msg -> Some(box msg))

                            | HybridAction.SimplifyPattern(line) ->
                                FileRefactoring.simplifyPattern targetFile line
                                |> Result.map (fun msg -> Some(box msg))

                            | HybridAction.RenameSymbol(oldName, newName) ->
                                FileRefactoring.renameSymbol targetFile oldName newName
                                |> Result.map (fun msg -> Some(box msg))

                            | HybridAction.UseTool(Tool.WebSearch _) -> FSharp.Core.Ok None
                            | HybridAction.UseTool _ -> FSharp.Core.Ok None

                            | _ -> FSharp.Core.Error $"Action not supported for file refactoring: {action}"

                        return result

                with ex ->
                    return FSharp.Core.Error(sprintf "Execution failed: %s" ex.Message)
            }

    /// Create a dry-run executor that just logs what would happen
    let createDryRunExecutor () : HybridAction -> Task<Result<obj option, string>> =
        fun action ->
            task {
                let description =
                    match action with
                    | HybridAction.ExtractFunction(name, s, e) ->
                        $"[DRY RUN] Would extract function '{name}' from lines {s}-{e}"
                    | HybridAction.RemoveDeadCode(s, e) -> $"[DRY RUN] Would remove lines {s}-{e}"
                    | HybridAction.AddDocumentation(doc, line) -> $"[DRY RUN] Would add doc '{doc}' at line {line}"
                    | HybridAction.InlineValue(name, line) -> $"[DRY RUN] Would inline '{name}' at line {line}"
                    | HybridAction.SimplifyPattern(line) -> $"[DRY RUN] Would simplify pattern at line {line}"
                    | HybridAction.RenameSymbol(old, new') -> $"[DRY RUN] Would rename '{old}' to '{new'}'"
                    | HybridAction.NoOp -> "[DRY RUN] No operation"
                    | _ -> $"[DRY RUN] Would execute: {action}"

                return FSharp.Core.Ok(Some(box description))
            }

    /// Create a composite executor that can handle multiple action types
    let createCompositeExecutor
        (fileExecutor: string -> HybridAction -> Task<Result<obj option, string>>)
        (fallback: HybridAction -> Task<Result<obj option, string>>)
        (targetFile: string option)
        : HybridAction -> Task<Result<obj option, string>> =

        fun action ->
            task {
                match action, targetFile with
                | (HybridAction.ExtractFunction _ | HybridAction.RemoveDeadCode _ | HybridAction.AddDocumentation _),
                  Some file -> return! fileExecutor file action
                | _, _ -> return! fallback action
            }
