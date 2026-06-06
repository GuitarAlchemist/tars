namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Globalization
open System.Text.Json
open System.Text.Json.Nodes
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.SpecKitWorkspace
open TarsEngine.FSharp.Core.Services.CrossAgentValidation

type Tier2PolicyState =
    { RequireConsensus: bool
      RequireCritic: bool }

type Tier2Action =
    | PolicyTightened
    | PolicyRelaxed
    | RemediationEnqueued
    | NoChange

type Tier2IterationOutcome =
    | NoPendingWork
    | Success of SpecDrivenIterationResult * Tier2Action list
    | Failure of SpecDrivenIterationResult * Tier2Action list

module Tier2Runner =

    let private policyPath () =
        Path.Combine(Environment.CurrentDirectory, ".specify", "tier2_policy.json")

    let private ledgerDirectory =
        Path.Combine(Environment.CurrentDirectory, ".specify", "ledger", "iterations")

    let private ensureDirectory (path: string) =
        let directory = Path.GetDirectoryName(path)
        if not (String.IsNullOrWhiteSpace(directory)) && not (Directory.Exists(directory)) then
            Directory.CreateDirectory(directory) |> ignore

    let private defaultPolicy =
        { RequireConsensus = false
          RequireCritic = false }

    let private loadPolicy () =
        let path = policyPath ()
        if File.Exists(path) then
            try
                let json = File.ReadAllText(path)
                JsonSerializer.Deserialize<Tier2PolicyState>(json)
                |> Option.ofObj
                |> Option.defaultValue defaultPolicy
            with _ ->
                defaultPolicy
        else
            defaultPolicy

    let private savePolicy state =
        let path = policyPath ()
        ensureDirectory path
        let json = JsonSerializer.Serialize(state, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(path, json)

    type private GovernanceSnapshot =
        { CapabilityPassRatio: float option
          CapabilityTrend: float option
          SafetyConfidence: float option
          SafetyTrend: float option
          CriticStatus: string option
          CriticRejectRate: float option
          ValidatorDisagreementRatio: float option
          ValidatorFindings: int option
          ValidatorComments: int option
          ValidatorDisagreementsCount: int option }

    let private tryAsValue (node: JsonNode) =
        if isNull node then None else
        let value = node.AsValue()
        if isNull value then None else Some value

    let private parseFloat (node: JsonNode) =
        match tryAsValue node with
        | None -> None
        | Some value ->
            match value.GetValueKind() with
            | JsonValueKind.Number ->
                try Some (value.GetValue<double>())
                with _ -> None
            | JsonValueKind.String ->
                match Double.TryParse(value.GetValue<string>()) with
                | true, parsed -> Some parsed
                | _ -> None
            | _ -> None

    let private parseString (node: JsonNode) =
        match tryAsValue node with
        | Some value when value.GetValueKind() = JsonValueKind.String -> Some(value.GetValue<string>())
        | _ -> None

    let private parseInt (node: JsonNode) =
        match tryAsValue node with
        | None -> None
        | Some value ->
            match value.GetValueKind() with
            | JsonValueKind.Number ->
                let mutable parsedInt = 0
                if value.TryGetValue<int>(&parsedInt) then Some parsedInt
                else
                    let mutable parsedDouble = 0.0
                    if value.TryGetValue<double>(&parsedDouble) then
                        Some (Convert.ToInt32(Math.Round(parsedDouble, MidpointRounding.AwayFromZero)))
                    else
                        None
            | JsonValueKind.String ->
                match Int32.TryParse(value.GetValue<string>()) with
                | true, parsed -> Some parsed
                | _ -> None
            | _ -> None

    let private loadLedgerFiles limit =
        if not (Directory.Exists(ledgerDirectory)) then
            []
        else
            Directory.GetFiles(ledgerDirectory, "*.json", SearchOption.TopDirectoryOnly)
            |> Array.filter (fun path -> not (path.EndsWith("latest.json", StringComparison.OrdinalIgnoreCase)))
            |> Array.sortDescending
            |> Array.truncate limit
            |> Array.toList

    let private parseMetrics (path: string) =
        try
            let node = JsonNode.Parse(File.ReadAllText(path))
            let metrics = if isNull node then null else node.["metrics"]
            if isNull metrics then None
            else
                let capability = parseFloat metrics.["capability.pass_ratio"]
                let safety = parseFloat metrics.["safety.consensus_avg_confidence"]
                let critic = parseString metrics.["safety.critic_status"]
                let disagreements = parseFloat metrics.["validators.disagreement_ratio"]
                let findings = parseInt metrics.["validators.findings_total"]
                let comments = parseInt metrics.["validators.comments"]
                let disagreementCount = parseInt metrics.["validators.disagreements"]
                Some(capability, safety, critic, disagreements, findings, comments, disagreementCount)
        with _ ->
            None

    let private loadGovernanceSnapshot () =
        let latestPath = Path.Combine(ledgerDirectory, "latest.json")
        let latestMetrics =
            if File.Exists(latestPath) then parseMetrics latestPath else None

        let history =
            loadLedgerFiles 10
            |> List.choose parseMetrics

        let averageOptions values =
            values
            |> List.choose id
            |> fun lst -> if List.isEmpty lst then None else Some(List.average lst)

        let historyCaps = history |> List.map (fun (cap, _, _, _, _, _, _) -> cap)
        let historySafety = history |> List.map (fun (_, safety, _, _, _, _, _) -> safety)
        let historyCriticStatuses = history |> List.map (fun (_, _, status, _, _, _, _) -> status)
        let historyDisagreements = history |> List.map (fun (_, _, _, disagreement, _, _, _) -> disagreement)
        let historyFindings = history |> List.map (fun (_, _, _, _, findings, _, _) -> findings |> Option.map float)
        let historyComments = history |> List.map (fun (_, _, _, _, _, comments, _) -> comments |> Option.map float)
        let historyDisagreementCounts = history |> List.map (fun (_, _, _, _, _, _, disagreementCount) -> disagreementCount |> Option.map float)

        let criticRejectRate latest =
            let statuses =
                (latest :: historyCriticStatuses)
                |> List.choose id
            if List.isEmpty statuses then None
            else
                let rejectCount =
                    statuses
                    |> List.filter (fun status -> String.Equals(status, "reject", StringComparison.OrdinalIgnoreCase))
                    |> List.length
                Some (float rejectCount / float statuses.Length)

        match latestMetrics with
        | None ->
            { CapabilityPassRatio = None
              CapabilityTrend = averageOptions historyCaps
              SafetyConfidence = None
              SafetyTrend = averageOptions historySafety
              CriticStatus = None
              CriticRejectRate = criticRejectRate None
              ValidatorDisagreementRatio = averageOptions historyDisagreements
              ValidatorFindings =
                  averageOptions historyFindings
                  |> Option.map (fun value -> Convert.ToInt32(Math.Round(value)))
              ValidatorComments =
                  averageOptions historyComments
                  |> Option.map (fun value -> Convert.ToInt32(Math.Round(value)))
              ValidatorDisagreementsCount =
                  averageOptions historyDisagreementCounts
                  |> Option.map (fun value -> Convert.ToInt32(Math.Round(value))) }
        | Some (capability, safety, critic, disagreements, findings, comments, disagreementCount) ->
            { CapabilityPassRatio = capability
              CapabilityTrend = averageOptions (capability :: historyCaps)
              SafetyConfidence = safety
              SafetyTrend = averageOptions (safety :: historySafety)
              CriticStatus = critic
              CriticRejectRate = criticRejectRate critic
              ValidatorDisagreementRatio = averageOptions (disagreements :: historyDisagreements)
              ValidatorFindings = findings
              ValidatorComments = comments
              ValidatorDisagreementsCount = disagreementCount }

    let private buildConsensusRule () =
        { MinimumPassCount = 3
          RequiredRoles =
            [ AgentRole.SafetyGovernor
              AgentRole.PerformanceBenchmarker
              AgentRole.SpecGuardian ]
          AllowNeedsReview = false
          MinimumConfidence = Some 0.7
          MaxFailureCount = Some 0 }

    let private buildHarnessOptions (policy: Tier2PolicyState) =
        let validationCommand =
            { Name = "dotnet-test"
              Executable = "dotnet"
              Arguments = "test Tars.sln -c Release --no-build"
              WorkingDirectory = Some Environment.CurrentDirectory
              Timeout = None
              Environment = Map.empty }

        { defaultHarnessOptions with
            Commands =
                { defaultHarnessOptions.Commands with
                    Validation = [ validationCommand ] }
            EnableAutoPatch = true
            PersistAdaptiveMemory = true
            RequireCriticApproval = policy.RequireCritic
            ReasoningCritic = None
            ConsensusRule = if policy.RequireConsensus then Some(buildConsensusRule ()) else None }

    let private tightenPolicy (policy: Tier2PolicyState) =
        let tightened =
            { RequireConsensus = true
              RequireCritic = true }
        if tightened = policy then
            policy, []
        else
            tightened, [Tier2Action.PolicyTightened]

    let private relaxPolicy (policy: Tier2PolicyState) =
        let relaxed =
            { RequireConsensus = false
              RequireCritic = false }
        if relaxed = policy then
            policy, []
        else
            relaxed, [Tier2Action.PolicyRelaxed]

    let private evaluatePolicyAdjustments (snapshot: GovernanceSnapshot) (policy: Tier2PolicyState) (success: bool) =
        let criticReject = snapshot.CriticRejectRate |> Option.defaultValue 0.0
        let validatorDisagreements = snapshot.ValidatorDisagreementRatio |> Option.defaultValue 0.0
        let unsafeSafety =
            snapshot.SafetyTrend
            |> Option.exists (fun value -> value < 0.8)
            || criticReject > 0.25

        let validatorPressure = validatorDisagreements > 0.2

        let shouldRelax =
            success
            && (policy.RequireConsensus || policy.RequireCritic)
            && criticReject <= 0.1
            && validatorDisagreements < 0.05
            && (snapshot.SafetyConfidence
                |> Option.orElse snapshot.SafetyTrend
                |> Option.defaultValue 1.0) >= 0.85

        if not success || unsafeSafety || validatorPressure then
            tightenPolicy policy
        elif shouldRelax then
            relaxPolicy policy
        else
            policy, [Tier2Action.NoChange]

    let private formatProgress snapshot =
        let formatPercent (valueOpt: float option) =
            valueOpt
            |> Option.map (fun (v: float) -> v.ToString("P1", CultureInfo.InvariantCulture))
            |> Option.defaultValue "n/a"

        let formatScore (valueOpt: float option) =
            valueOpt
            |> Option.map (fun (v: float) -> v.ToString("F2", CultureInfo.InvariantCulture))
            |> Option.defaultValue "n/a"

        let capability = formatPercent snapshot.CapabilityPassRatio
        let safety = formatPercent snapshot.SafetyConfidence
        let critic = snapshot.CriticStatus |> Option.defaultValue "n/a"
        let disagreements = formatPercent snapshot.ValidatorDisagreementRatio
        let findings = snapshot.ValidatorFindings |> Option.map string |> Option.defaultValue "n/a"
        let comments = snapshot.ValidatorComments |> Option.map string |> Option.defaultValue "n/a"
        let criticReject = formatScore snapshot.CriticRejectRate

        $"capability={capability} safety={safety} critic={critic} criticReject={criticReject} validatorDisagreements={disagreements} findings={findings} comments={comments}"

    let private toOutcome (result: SpecDrivenIterationResult) (actions: Tier2Action list) =
        let harnessPassed =
            result.HarnessReport
            |> Option.exists (fun report ->
                match report.Outcome with
                | HarnessOutcome.AllPassed _ -> true
                | _ -> false)

        if harnessPassed then
            Success(result, actions |> List.filter ((<>) Tier2Action.NoChange))
        else
            Failure(result, (Tier2Action.RemediationEnqueued :: actions) |> List.filter ((<>) Tier2Action.NoChange))

    let runIterationAsync
        (service: ISelfImprovementService)
        (loggerFactory: ILoggerFactory)
        (baseDirectory: string option)
        (executor: ICommandExecutor option) =
        async {
            let logger = loggerFactory.CreateLogger("Tier2Runner")
            let policy = loadPolicy ()
            let snapshot = loadGovernanceSnapshot ()
            logger.LogInformation("Governance snapshot before iteration: {Snapshot}", formatProgress snapshot)

            let harnessOptions = buildHarnessOptions policy

            logger.LogInformation("Starting Tier2 iteration with policy RequireConsensus={RequireConsensus} RequireCritic={RequireCritic}",
                                  policy.RequireConsensus,
                                  policy.RequireCritic)

            let iterationAsync =
                match baseDirectory, executor with
                | None, None ->
                    service.RunNextSpecKitIterationAsync(loggerFactory, options = harnessOptions)
                | Some dir, None ->
                    service.RunNextSpecKitIterationAsync(loggerFactory, baseDirectory = dir, options = harnessOptions)
                | None, Some exec ->
                    service.RunNextSpecKitIterationAsync(loggerFactory, options = harnessOptions, executor = exec)
                | Some dir, Some exec ->
                    service.RunNextSpecKitIterationAsync(loggerFactory, baseDirectory = dir, options = harnessOptions, executor = exec)

            let! iteration = iterationAsync

            match iteration with
            | None ->
                logger.LogInformation("Tier2 runner found no pending Spec Kit tasks.")
                return NoPendingWork
            | Some result ->
                let harnessPassed =
                    result.HarnessReport
                    |> Option.exists (fun report ->
                        match report.Outcome with
                        | HarnessOutcome.AllPassed _ -> true
                        | _ -> false)

                let updatedPolicy, policyActions =
                    evaluatePolicyAdjustments snapshot policy harnessPassed

                if updatedPolicy <> policy then
                    savePolicy updatedPolicy
                    logger.LogInformation("Tier2 policy updated. RequireConsensus={RequireConsensus} RequireCritic={RequireCritic}",
                                          updatedPolicy.RequireConsensus,
                                          updatedPolicy.RequireCritic)

                let outcome = toOutcome result policyActions
                logger.LogInformation(
                    "Tier2 iteration result: {Outcome} actions={Actions} snapshot={Snapshot}",
                    (match outcome with
                     | Success _ -> "success"
                     | Failure _ -> "failure"
                     | NoPendingWork -> "no-work"),
                    policyActions,
                    formatProgress snapshot)
                return outcome
        }








