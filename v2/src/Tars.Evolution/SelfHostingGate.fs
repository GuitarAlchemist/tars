namespace Tars.Evolution

open System
open System.Diagnostics
open System.IO
open System.Text
open System.Text.Json
open System.Xml.Linq
open Tars.Llm

/// Self-hosting recursive improvement (ADR 0002): TARS improving its own F#
/// source, gated by its own test suite.
///
/// A `GateTask` proposes editing one non-test source file to make a target test
/// pass. The gate builds + tests the edit inside an isolated git worktree
/// (never the live tree), then accepts iff the hermetic invariants hold: the
/// target test flipped to passing, no other test regressed, and the test set is
/// unchanged. Accepted edits are committed to a fresh `self-improve/*` branch.
///
/// The decision logic (parseTrx / isTestFile / decide) is pure and unit-tested;
/// the worktree+dotnet orchestration is the mechanic spiked in ADR 0002.
module SelfHostingGate =

    // ── Domain ────────────────────────────────────────────────────────────────

    /// One self-improvement task: make `TargetTest` pass by replacing `OldText`
    /// with `NewText` in `TargetFile` (repo-relative, must be a non-test source).
    type GateTask =
        { TargetTest: string
          TargetFile: string
          OldText: string
          NewText: string
          Rationale: string }

    /// The hermetic gate's verdict on a variant (pure; branch assigned by the runner).
    type GateDecision =
        | Accept of rationale: string
        | Reject of reason: string

    /// Outcome of running the full gate, including the promotion branch on accept.
    type GateVerdict =
        | Promoted of branch: string * rationale: string
        | Rejected of reason: string

    // ── Pure logic (no git / no dotnet — unit-tested) ─────────────────────────

    /// Hermetic boundary: a variant may never edit a test file. True for paths
    /// under a `tests/` dir or whose filename ends in `Tests`.
    let isTestFile (path: string) : bool =
        let p = path.Replace('\\', '/').ToLowerInvariant()
        p.Contains("/tests/")
        || p.StartsWith("tests/")
        || Path.GetFileNameWithoutExtension(p).EndsWith("tests")

    /// Parse a VSTest TRX document into testName → outcome ("Passed"/"Failed"/…).
    let parseTrx (trxXml: string) : Map<string, string> =
        try
            let doc = XDocument.Parse(trxXml)
            let ns = doc.Root.Name.Namespace
            doc.Descendants(ns + "UnitTestResult")
            |> Seq.choose (fun e ->
                match e.Attribute(XName.Get "testName"), e.Attribute(XName.Get "outcome") with
                | null, _
                | _, null -> None
                | n, o -> Some(n.Value, o.Value))
            |> Map.ofSeq
        with _ ->
            Map.empty

    let private isPass (o: string) = o = "Passed"

    /// The hermetic gate decision over baseline vs variant test outcomes.
    /// Accept iff: the test set is unchanged (no add/drop/Skip-gaming), no test
    /// regressed (Passed → non-Passed), and the target test was failing at
    /// baseline and now passes.
    let decide
        (targetTest: string)
        (baseline: Map<string, string>)
        (variant: Map<string, string>)
        : GateDecision =
        let baseNames = baseline |> Map.toSeq |> Seq.map fst |> Set.ofSeq
        let varNames = variant |> Map.toSeq |> Seq.map fst |> Set.ofSeq

        if baseNames.IsEmpty || varNames.IsEmpty then
            Reject "no test results parsed (build or run failure)"
        elif baseNames <> varNames then
            Reject(
                sprintf
                    "test set changed (%d → %d): hermetic violation"
                    baseNames.Count
                    varNames.Count)
        else
            let regressions =
                baseline
                |> Map.toSeq
                |> Seq.filter (fun (n, o) ->
                    isPass o
                    && (variant |> Map.tryFind n |> Option.map (isPass >> not) |> Option.defaultValue true))
                |> Seq.map fst
                |> Seq.toList

            if not (List.isEmpty regressions) then
                Reject(
                    sprintf
                        "%d regression(s): %s"
                        regressions.Length
                        (String.Join(", ", regressions |> List.truncate 3)))
            else
                let matches = varNames |> Set.filter (fun n -> n.Contains targetTest) |> Set.toList
                match matches with
                | [] -> Reject(sprintf "target test '%s' not found" targetTest)
                | _ ->
                    let allPass =
                        matches
                        |> List.forall (fun n -> variant |> Map.tryFind n |> Option.map isPass |> Option.defaultValue false)
                    let wasFailing =
                        matches
                        |> List.exists (fun n ->
                            baseline |> Map.tryFind n |> Option.map (isPass >> not) |> Option.defaultValue false)
                    if not allPass then
                        Reject(sprintf "target test '%s' does not pass in the variant" targetTest)
                    elif not wasFailing then
                        Reject(sprintf "target test '%s' already passed at baseline (no improvement)" targetTest)
                    else
                        Accept(
                            sprintf
                                "target '%s' now passes; 0 regressions; %d tests unchanged"
                                targetTest
                                varNames.Count)

    /// Reconcile an edit's line endings to the target content's EOL style, then
    /// apply it iff `oldText` occurs exactly once. Pure (no IO).
    ///
    /// Models routinely normalize multi-line `old_text` to LF, but Windows source
    /// files are CRLF, so an exact match would always miss. We coerce the needle
    /// (and replacement) to the file's EOL style before matching. Returns the new
    /// content on a clean single-occurrence replacement, else None.
    let applyEditPure (content: string) (oldText: string) (newText: string) : string option =
        let toContentEol (s: string) =
            let lf = s.Replace("\r\n", "\n")
            if content.Contains "\r\n" then lf.Replace("\n", "\r\n") else lf
        let o = toContentEol oldText
        let n = toContentEol newText
        if String.IsNullOrEmpty o || not (content.Contains o) then
            None
        else
            let occurrences = (content.Length - content.Replace(o, "").Length) / max 1 o.Length
            if occurrences <> 1 then None else Some(content.Replace(o, n))

    /// Winnow best-of-N proposals before spending any gate cycles: keep only those
    /// whose edit actually applies to `content` (the most common failure is an
    /// inapplicable/hallucinated old_text — caught here for free), de-duplicated by
    /// (oldText, newText) with order preserved. Pure. ADR 0002 D5.
    let viableProposals (content: string) (proposals: GateTask list) : GateTask list =
        proposals
        |> List.filter (fun t -> (applyEditPure content t.OldText t.NewText).IsSome)
        |> List.distinctBy (fun t -> t.OldText, t.NewText)

    // ── Generation (LLM proposes the edit — makes the loop self-driving) ──────

    /// Prompt the model to fix a failing test by editing one source file.
    let buildProposePrompt (targetTest: string) (targetFile: string) (fileContent: string) : string =
        sprintf
            "A test is failing. Edit the source file to make it pass without breaking other tests.\n\n\
             FAILING TEST: %s\n\n\
             SOURCE FILE (%s):\n%s\n\n\
             Output ONLY a JSON object:\n\
             {\"rationale\": \"one sentence\", \"old_text\": \"exact text to replace\", \"new_text\": \"replacement\"}\n\
             The old_text must appear EXACTLY ONCE in the file."
            targetTest
            targetFile
            fileContent

    /// Prompt the model to *repair* a rejected edit: show it the edit it tried and
    /// the hermetic gate's rejection so it can correct course, not start over. The
    /// rejection reason is the signal that makes this a repair and not a re-roll.
    /// Pure. ADR 0002 D5 (repair tail).
    let buildRepairPrompt
        (targetTest: string)
        (targetFile: string)
        (fileContent: string)
        (failedEdit: GateTask)
        (error: string)
        : string =
        sprintf
            "A previous attempt to fix a failing test was REJECTED by the hermetic gate. \
             Study the rejection and propose a CORRECTED edit — do not repeat the rejected one.\n\n\
             FAILING TEST: %s\n\n\
             PREVIOUS old_text:\n%s\n\n\
             PREVIOUS new_text:\n%s\n\n\
             GATE REJECTION: %s\n\n\
             SOURCE FILE (%s):\n%s\n\n\
             Output ONLY a JSON object:\n\
             {\"rationale\": \"one sentence\", \"old_text\": \"exact text to replace\", \"new_text\": \"replacement\"}\n\
             The old_text must appear EXACTLY ONCE in the file."
            targetTest
            failedEdit.OldText
            failedEdit.NewText
            error
            targetFile
            fileContent

    /// Rank a reject reason by how close the variant got to passing — higher is
    /// closer, so the repair round seeds from the most informative failure. A
    /// variant that built and ran (regression / target-didn't-pass) carries more
    /// repair signal than one that failed to build, drop tests, or even apply. Pure.
    let repairRank (reason: string) : int =
        let r = reason.ToLowerInvariant()
        if r.Contains "regression" then 3
        elif r.Contains "does not pass" then 2
        elif r.Contains "test set changed" then 1
        else 0

    /// Parse the model's JSON mutation response into (rationale, oldText, newText).
    /// Robust to prose around the JSON (extracts the first {...} block). Pure.
    let parseProposal (response: string) : Result<string * string * string, string> =
        try
            let t = response.Trim()
            let i = t.IndexOf '{'
            let j = t.LastIndexOf '}'
            if i < 0 || j <= i then
                Result.Error "no JSON object in response"
            else
                use doc = JsonDocument.Parse(t.Substring(i, j - i + 1))
                let root = doc.RootElement
                let get (names: string list) =
                    names
                    |> List.tryPick (fun n ->
                        match root.TryGetProperty n with
                        | true, v when v.ValueKind = JsonValueKind.String -> Some(v.GetString())
                        | _ -> None)
                match get [ "old_text"; "old" ], get [ "new_text"; "new" ] with
                | Some o, Some n when o <> "" ->
                    Result.Ok(get [ "rationale" ] |> Option.defaultValue "self-improvement", o, n)
                | _ -> Result.Error "missing old_text/new_text"
        with ex ->
            Result.Error(sprintf "parse error: %s" ex.Message)

    // ── SFT coupling (ADR 0003): verified wins become training data ───────────

    /// System prompt the self-improvement generator is trained to answer.
    let selfHostSystemPrompt =
        "You are the TARS self-improvement engine. A test is failing. Propose a single "
        + "JSON mutation {rationale, old_text, new_text} that makes it pass without "
        + "breaking other tests."

    /// Distill a verified (Accept'd) gate run into one SFT JSONL line, in the same
    /// `{messages:[system,user,assistant]}` shape as SelfTrain's benchmark dataset
    /// so the two merge. The assistant target is the mutation JSON — the exact thing
    /// the generator must emit at run time (ADR 0003 D2). Pure.
    let buildSftExample (task: GateTask) : string =
        let userContent =
            sprintf
                "Make the failing test `%s` pass by editing %s. Output a single JSON mutation."
                task.TargetTest
                task.TargetFile
        let assistantContent =
            JsonSerializer.Serialize(
                {| rationale = task.Rationale
                   old_text = task.OldText
                   new_text = task.NewText |})
        let ex =
            {| messages =
                [ {| role = "system"; content = selfHostSystemPrompt |}
                  {| role = "user"; content = userContent |}
                  {| role = "assistant"; content = assistantContent |} ] |}
        JsonSerializer.Serialize ex

    let private winsPath () =
        let dir =
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        Path.Combine(dir, "self_host_wins.jsonl")

    /// Append a verified self-hosting win as an SFT example (best-effort; never
    /// blocks the gate). Only Accept'd diffs reach here, so the dataset stays
    /// verified-only — the anti-collapse invariant (ADR 0003 D1/D4).
    let recordWin (task: GateTask) : unit =
        try
            File.AppendAllText(winsPath (), buildSftExample task + "\n")
        with _ ->
            ()

    // ── IO orchestration (thin; mechanic spiked in ADR 0002) ──────────────────

    /// Run a process, draining both pipes, returning (exitCode, stdout, stderr).
    let private run (workdir: string) (fileName: string) (args: string) : int * string * string =
        let psi = ProcessStartInfo()
        psi.FileName <- fileName
        psi.Arguments <- args
        psi.WorkingDirectory <- workdir
        psi.UseShellExecute <- false
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.CreateNoWindow <- true
        use proc = new Process()
        proc.StartInfo <- psi
        proc.Start() |> ignore
        let outT = proc.StandardOutput.ReadToEndAsync()
        let errT = proc.StandardError.ReadToEndAsync()
        proc.WaitForExit()
        proc.ExitCode, outT.Result, errT.Result

    /// Run the test project in `workdir`, emitting a TRX, and return parsed
    /// outcomes. An empty map signals a build/run failure (a Reject signal).
    let private runTests (workdir: string) (testProject: string) (trxName: string) : Map<string, string> =
        let _, _, _ =
            run
                workdir
                "dotnet"
                (sprintf
                    "test \"%s\" -p:NuGetAudit=false --logger \"trx;LogFileName=%s\""
                    testProject
                    trxName)
        // TRX lands under <testProjectDir>/TestResults/<trxName>
        let trxPath =
            Path.Combine(workdir, Path.GetDirectoryName(testProject), "TestResults", trxName)
        if File.Exists trxPath then parseTrx (File.ReadAllText trxPath) else Map.empty

    /// Apply the task's edit to a file under `root`. Returns false if `OldText`
    /// is absent or ambiguous (a precondition for a clean, reviewable mutation).
    let private applyEdit (root: string) (task: GateTask) : bool =
        let full = Path.Combine(root, task.TargetFile)
        if not (File.Exists full) then
            false
        else
            match applyEditPure (File.ReadAllText full) task.OldText task.NewText with
            | Some updated ->
                File.WriteAllText(full, updated)
                true
            | None -> false

    /// Serializes `git worktree add/remove` across threads. The metadata under
    /// `.git/worktrees` is race-prone, so only these ops are locked — the heavy
    /// `dotnet test` inside each worktree still runs in parallel.
    let private gitWorktreeLock = obj ()

    /// Create a detached worktree at HEAD, run `f` with its path, then always
    /// remove it (an accepted branch ref created inside `f` survives removal).
    let private withWorktree (repoRoot: string) (f: string -> 'a) : 'a =
        let id = Guid.NewGuid().ToString("n").Substring(0, 8)
        let wt = Path.Combine(Path.GetTempPath(), sprintf "tars_selfheal_%s" id)
        lock gitWorktreeLock (fun () ->
            let code, _, err = run repoRoot "git" (sprintf "worktree add --detach \"%s\" HEAD" wt)
            if code <> 0 then failwithf "git worktree add failed: %s" (err.Trim()))
        try
            f wt
        finally
            try
                lock gitWorktreeLock (fun () ->
                    run repoRoot "git" (sprintf "worktree remove --force \"%s\"" wt) |> ignore)
            with _ ->
                ()

    /// Baseline outcomes at HEAD — identical for every proposal, so compute once
    /// and share across the best-of-N evaluations (ADR 0002 D5).
    let private computeBaseline (repoRoot: string) (testProject: string) : Map<string, string> =
        withWorktree repoRoot (fun wt -> runTests wt testProject "base.trx")

    /// Evaluate one proposal against a shared baseline in its own worktree:
    /// apply → variant `dotnet test` → `decide`. No promotion (the caller promotes
    /// the chosen winner). Pure-by-result; the worktree is always cleaned up.
    let private evaluateVariant
        (repoRoot: string)
        (testProject: string)
        (baseline: Map<string, string>)
        (task: GateTask)
        : GateDecision =
        if isTestFile task.TargetFile then
            Reject "target is a test file (hermetic boundary)"
        else
            withWorktree repoRoot (fun wt ->
                if not (applyEdit wt task) then
                    Reject "edit precondition failed (OldText missing or not unique)"
                else
                    let trx = sprintf "var_%s.trx" (Path.GetFileName wt)
                    let variant = runTests wt testProject trx
                    decide task.TargetTest baseline variant)

    /// Promote a verified task: apply it in a fresh worktree, commit to a new
    /// `self-improve/<id>` branch, and record the SFT win (ADR 0003). The branch
    /// ref survives worktree removal.
    let private promoteTask (repoRoot: string) (task: GateTask) (rationale: string) : GateVerdict =
        let branch = sprintf "self-improve/%s" (Guid.NewGuid().ToString("n").Substring(0, 8))
        withWorktree repoRoot (fun wt ->
            if not (applyEdit wt task) then
                Rejected "edit precondition failed at promote (concurrent change?)"
            else
                let gitWt a =
                    let code, _, err = run wt "git" a
                    if code <> 0 then failwithf "git %s failed: %s" a (err.Trim())
                gitWt (sprintf "checkout -b %s" branch)
                gitWt (sprintf "add \"%s\"" task.TargetFile)
                gitWt (sprintf "commit -m \"self-improve: %s\"" (task.Rationale.Replace("\"", "'")))
                recordWin task
                Promoted(branch, rationale))

    /// Run the full hermetic gate for `task` against `repoRoot`. Creates a
    /// detached worktree at HEAD, captures baseline outcomes, applies the edit,
    /// captures variant outcomes, decides, and on Accept commits the edit to a
    /// fresh `self-improve/<id>` branch (the worktree's own branch). The worktree
    /// directory is always removed; an accepted branch ref survives.
    let runGate (repoRoot: string) (testProject: string) (task: GateTask) : GateVerdict =
        if isTestFile task.TargetFile then
            Rejected "target is a test file (hermetic boundary)"
        else
            let id = Guid.NewGuid().ToString("n").Substring(0, 8)
            let branch = sprintf "self-improve/%s" id
            let wt = Path.Combine(Path.GetTempPath(), sprintf "tars_selfheal_%s" id)
            let git args =
                let code, _, err = run repoRoot "git" args
                if code <> 0 then failwithf "git %s failed: %s" args (err.Trim())
            try
                try
                    git (sprintf "worktree add --detach \"%s\" HEAD" wt)
                    let baseline = runTests wt testProject (sprintf "base_%s.trx" id)
                    if not (applyEdit wt task) then
                        Rejected "edit precondition failed (OldText missing or not unique)"
                    else
                        let variant = runTests wt testProject (sprintf "var_%s.trx" id)
                        match decide task.TargetTest baseline variant with
                        | Reject reason -> Rejected reason
                        | Accept rationale ->
                            // Promote: turn the verified worktree into a branch commit.
                            let gitWt a =
                                let code, _, err = run wt "git" a
                                if code <> 0 then failwithf "git %s failed: %s" a (err.Trim())
                            gitWt (sprintf "checkout -b %s" branch)
                            gitWt (sprintf "add \"%s\"" task.TargetFile)
                            gitWt (sprintf "commit -m \"self-improve: %s\"" (task.Rationale.Replace("\"", "'")))
                            // Verified win → SFT training data (ADR 0003).
                            recordWin task
                            Promoted(branch, rationale)
                with ex ->
                    Rejected(sprintf "gate error: %s" ex.Message)
            finally
                try run repoRoot "git" (sprintf "worktree remove --force \"%s\"" wt) |> ignore with _ -> ()

    /// Generate up to `n` candidate edits for the failing test. Proposal 0 is
    /// greedy (temperature 0 — the model's best single guess); the rest are sampled
    /// (temperature 0.6, distinct seeds) for diversity (ADR 0002 D5). Parse failures
    /// are dropped. Sequential — the local model serves one request at a time.
    let private generateProposals
        (llm: ILlmService)
        (targetTest: string)
        (targetFile: string)
        (content: string)
        (n: int)
        : Async<GateTask list> =
        async {
            let prompt = buildProposePrompt targetTest targetFile content
            let mkReq i =
                { ModelHint = None
                  Model = None
                  SystemPrompt = Some selfHostSystemPrompt
                  MaxTokens = Some 2000
                  Temperature = Some(if i = 0 then 0.0 else 0.6)
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = Some i
                  ContextWindow = None }
            let tasks = ResizeArray<GateTask>()
            for i in 0 .. (max 1 n) - 1 do
                let! resp = llm.CompleteAsync(mkReq i) |> Async.AwaitTask
                match parseProposal resp.Text with
                | Result.Ok(rationale, oldText, newText) ->
                    tasks.Add
                        { TargetTest = targetTest
                          TargetFile = targetFile
                          OldText = oldText
                          NewText = newText
                          Rationale = rationale }
                | Result.Error _ -> ()
            return List.ofSeq tasks
        }

    /// Generate one greedy (temperature 0) repair edit from a repair prompt that
    /// already carries the prior failed edit and the gate's rejection. ADR 0002 D5.
    /// Returns None on generation/parse failure.
    let private generateRepair
        (llm: ILlmService)
        (targetTest: string)
        (targetFile: string)
        (prompt: string)
        : Async<GateTask option> =
        async {
            let req =
                { ModelHint = None
                  Model = None
                  SystemPrompt = Some selfHostSystemPrompt
                  MaxTokens = Some 2000
                  Temperature = Some 0.0
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = Some 0
                  ContextWindow = None }
            let! resp = llm.CompleteAsync(req) |> Async.AwaitTask
            match parseProposal resp.Text with
            | Result.Ok(rationale, oldText, newText) ->
                return
                    Some
                        { TargetTest = targetTest
                          TargetFile = targetFile
                          OldText = oldText
                          NewText = newText
                          Rationale = rationale }
            | Result.Error _ -> return None
        }

    /// Best-of-N self-driving gate (ADR 0002 D5): generate N diverse proposals,
    /// keep the ones whose edit applies (cheap pure pre-filter), evaluate them
    /// against a shared HEAD baseline with up to `maxConcurrency` parallel
    /// `dotnet test` runs, and promote the first that passes the hermetic gate.
    /// If none pass, a single error-fed *repair* round (D5 tail) feeds the most
    /// informative rejection back to the model for one corrected attempt before
    /// returning Rejected.
    let runGateBestOfN
        (llm: ILlmService)
        (repoRoot: string)
        (testProject: string)
        (targetTest: string)
        (targetFile: string)
        (n: int)
        (maxConcurrency: int)
        : Async<GateVerdict> =
        async {
            let full = Path.Combine(repoRoot, targetFile)
            if isTestFile targetFile then
                return Rejected "target is a test file (hermetic boundary)"
            elif not (File.Exists full) then
                return Rejected(sprintf "target file not found: %s" targetFile)
            else
                let content = File.ReadAllText full
                let! proposals = generateProposals llm targetTest targetFile content n
                let viable = viableProposals content proposals
                if List.isEmpty viable then
                    return Rejected(sprintf "no applicable proposal from %d generation(s)" (List.length proposals))
                else
                    let baseline = computeBaseline repoRoot testProject
                    if Map.isEmpty baseline then
                        return Rejected "baseline build/run produced no tests (build failure)"
                    else
                        let arr = List.toArray viable
                        let conc = max 1 maxConcurrency
                        let mutable winner = None
                        let mutable rejects = []
                        let mutable i = 0
                        // Evaluate in chunks; accept the first green and stop.
                        while Option.isNone winner && i < arr.Length do
                            let chunk = arr.[i .. min (i + conc - 1) (arr.Length - 1)]
                            let! decisions =
                                chunk
                                |> Array.map (fun t ->
                                    async {
                                        do! Async.SwitchToThreadPool()
                                        return (t, evaluateVariant repoRoot testProject baseline t)
                                    })
                                |> Async.Parallel
                            match decisions |> Array.tryPick (fun (t, d) -> match d with | Accept r -> Some(t, r) | _ -> None) with
                            | Some hit -> winner <- Some hit
                            | None ->
                                rejects <-
                                    rejects
                                    @ (decisions
                                       |> Array.choose (fun (t, d) -> match d with | Reject r -> Some(t, r) | _ -> None)
                                       |> Array.toList)
                            i <- i + conc
                        match winner with
                        | Some(task, rationale) -> return promoteTask repoRoot task rationale
                        | None ->
                            // D5 repair tail: seed one corrected attempt from the most
                            // informative rejection (the variant that got closest to green).
                            match rejects |> List.sortByDescending (snd >> repairRank) with
                            | (failedTask, error) :: _ ->
                                let prompt = buildRepairPrompt targetTest targetFile content failedTask error
                                let! repaired = generateRepair llm targetTest targetFile prompt
                                match
                                    repaired
                                    |> Option.filter (fun t -> (applyEditPure content t.OldText t.NewText).IsSome)
                                with
                                | Some t ->
                                    match evaluateVariant repoRoot testProject baseline t with
                                    | Accept r -> return promoteTask repoRoot t r
                                    | Reject rr ->
                                        return
                                            Rejected(
                                                sprintf
                                                    "best-of-%d + repair: none passed; repair: %s; first: %s"
                                                    arr.Length
                                                    rr
                                                    error)
                                | None ->
                                    return
                                        Rejected(
                                            sprintf
                                                "best-of-%d: none passed; repair did not apply; first: %s"
                                                arr.Length
                                                error)
                            | [] ->
                                return
                                    Rejected(
                                        sprintf "best-of-%d: none passed (no reject reasons captured)" arr.Length)
        }

    /// Self-driving gate (single-shot): the LLM proposes one edit for a failing
    /// test, verified by the hermetic gate. Thin wrapper over best-of-N with N=1.
    let runGateGenerated
        (llm: ILlmService)
        (repoRoot: string)
        (testProject: string)
        (targetTest: string)
        (targetFile: string)
        : Async<GateVerdict> =
        runGateBestOfN llm repoRoot testProject targetTest targetFile 1 1
