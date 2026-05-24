namespace EvolutionaryClosure.Tests

open System
open System.Globalization
open System.IO
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.Core.Closures
open TarsEngine.FSharp.Core.Closures.UnifiedEvolutionaryClosureFactory
open TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Core.Metascript.FractalGrammarMetascripts
open TarsEngine.FSharp.Core.Specs

module EvolutionaryClosureTests =

    let factory =
        UnifiedEvolutionaryClosureFactory(NullLogger<UnifiedEvolutionaryClosureFactory>.Instance)

    let service =
        MetascriptClosureIntegrationService(
            NullLogger<MetascriptClosureIntegrationService>.Instance,
            factory
        )

    let private specPath (fileName: string) =
        Path.Combine(__SOURCE_DIRECTORY__, "..", "specs", "metascripts", fileName)
        |> Path.GetFullPath

    let private blockToDynamicSpec name sourcePath expectations (block: FractalMetascriptBlock) =
        let rules =
            block.Rules
            |> List.map (fun rule ->
                match rule with
                | FractalRule.SpawnAgent(agentType, count, strategy) -> UnifiedEvolutionaryClosureFactory.DynamicMetascriptRule.SpawnAgentRule(agentType, count, strategy)
                | FractalRule.ConnectAgents(source, target, label) -> UnifiedEvolutionaryClosureFactory.DynamicMetascriptRule.ConnectAgentsRule(source, target, label)
                | FractalRule.EmitMetric(metric, value) -> UnifiedEvolutionaryClosureFactory.DynamicMetascriptRule.EmitMetricRule(metric, value)
                | FractalRule.RepeatPattern(name, depth) -> UnifiedEvolutionaryClosureFactory.DynamicMetascriptRule.RepeatPatternRule(name, depth))

        { Name = name
          Rules = rules
          MaxDepth = block.MaxDepth
          SourcePath = sourcePath
          Expectations = expectations }

    [<Fact>]
    let ``Grammar evolution closure ranks grammars deterministically`` () =
        let grammars =
            """
            [
              { "name": "navigation", "rules": ["start -> move", "move -> stop"], "usage": 5 },
              { "name": "diagnostics", "rules": ["check -> analyse", "analyse -> report", "report -> close"], "usage": 2 }
            ]
            """

        let parameters =
            Map.ofList [ "grammars", box grammars ]

        let closure =
            factory.CreateClosure(
                GrammarEvolutionClosure("Systems Team", "Reduce ambiguity"),
                "navigation-evolution",
                Map.empty
            )

        let result = closure parameters |> Async.RunSynchronously

        Assert.True(result.Success, result.Error |> Option.defaultValue "")
        let summary = result.Output |> Option.get :?> GrammarEvolutionSummary
        Assert.Equal("Systems Team", summary.TeamName)
        Assert.Equal(2, summary.RankedGrammars.Length)
        Assert.Equal("diagnostics", fst summary.RankedGrammars.Head)
        Assert.True(result.PerformanceImpact.ContainsKey("diagnostics"))

    [<Fact>]
    let ``Grammar evolution closure surfaces missing grammar error`` () =
        let closure =
            factory.CreateClosure(
                GrammarEvolutionClosure("Systems Team", "Reduce ambiguity"),
                "navigation-evolution",
                Map.empty
            )

        let result = closure Map.empty |> Async.RunSynchronously

        Assert.False(result.Success)
        Assert.Equal(
            Some "Grammar definitions are required for evolution analysis.",
            result.Error
        )

    [<Fact>]
    let ``Grammar evolution closure rejects empty grammar payload`` () =
        let parameters = Map.ofList [ "grammars", box "[]" ]

        let closure =
            factory.CreateClosure(
                GrammarEvolutionClosure("Systems Team", "Reduce ambiguity"),
                "navigation-evolution",
                Map.empty
            )

        let result = closure parameters |> Async.RunSynchronously

        Assert.False(result.Success)
        Assert.Equal(
            Some "At least one grammar definition must be provided.",
            result.Error
        )

    [<Fact>]
    let ``Multi agent coordination closure aggregates team metrics`` () =
        let agents =
            """
            [
              { "name": "alpha", "role": "lead", "skill": 0.9, "availability": 0.8 },
              { "name": "beta", "role": "analysis", "skill": 0.85, "availability": 0.9 },
              { "name": "gamma", "role": "analysis", "skill": 0.7, "availability": 0.75 }
            ]
            """

        let parameters =
            Map.ofList [ "agents", box agents ]

        let closure =
            factory.CreateClosure(
                MultiAgentCoordinationClosure(CoordinationStrategy.Hierarchical "lead"),
                "team-coordination",
                Map.empty
            )

        let result = closure parameters |> Async.RunSynchronously

        Assert.True(result.Success, result.Error |> Option.defaultValue "")
        let summary = result.Output |> Option.get :?> CoordinationSummary
        Assert.Equal(2, summary.RoleCoverage.Count)
        Assert.True(summary.RoleCoverage |> Map.containsKey "analysis")
        Assert.NotEmpty(summary.SuggestedPairings)
        Assert.True(result.PerformanceImpact.ContainsKey("team_score"))

    [<Fact>]
    let ``Multi agent coordination closure surfaces missing agent payload`` () =
        let closure =
            factory.CreateClosure(
                MultiAgentCoordinationClosure(CoordinationStrategy.Hierarchical "lead"),
                "team-coordination",
                Map.empty
            )

        let result = closure Map.empty |> Async.RunSynchronously

        Assert.False(result.Success)
        Assert.Equal(
            Some "Agent profiles are required for coordination analysis.",
            result.Error
        )

    [<Fact>]
    let ``Fractal metascript closure generates deterministic script`` () =
        let closure =
            factory.CreateClosure(
                FractalMetascriptGeneratorClosure(3, "coordination"),
                "fractal-demo",
                Map.empty
            )

        let result = closure Map.empty |> Async.RunSynchronously

        Assert.True(result.Success, result.Error |> Option.defaultValue "")
        let artifact = result.Output |> Option.get :?> FractalMetascriptArtifact
        Assert.Contains("NODE coordination-1", artifact.Script)
        Assert.True(artifact.NodeCount > 0)

    [<Fact>]
    let ``Fractal metascript closure rejects non positive depth`` () =
        let closure =
            factory.CreateClosure(
                FractalMetascriptGeneratorClosure(0, "coordination"),
                "fractal-demo",
                Map.empty
            )

        let result = closure Map.empty |> Async.RunSynchronously

        Assert.False(result.Success)
        Assert.Equal(
            Some "Depth must be greater than zero for fractal metascript generation.",
            result.Error
        )

    [<Fact>]
    let ``Metascript integration executes grammar evolution command`` () =
        let commandLine =
            "CLOSURE_CREATE GRAMMAR_EVOLUTION \"GrammarReview\" team_name=\"Navigation\" evolution_goal=\"Improve coverage\" grammars='[{\"name\":\"nav\",\"rules\":[\"start\"]}]'"

        let command =
            match service.ParseClosureCommand(commandLine) with
            | Some cmd -> cmd
            | None -> failwith "Failed to parse command."

        let result =
            service.ExecuteClosureCommand(command)
            |> Async.RunSynchronously

        Assert.True(result.Success, result.Error |> Option.defaultValue "")
        Assert.Contains("Grammar evolution", result.OutputSummary)

    [<Fact>]
    let ``Asynchronous metascript command completes and logs results`` () =
        let commandLine =
            "CLOSURE_CREATE GRAMMAR_EVOLUTION \"AsyncGrammar\" team_name=\"AsyncNav\" evolution_goal=\"Improve coverage\" grammars='[{\"name\":\"nav\",\"rules\":[\"start\"],\"usage\":1}]' mode=async context.note=\"test-run\""

        let command =
            match service.ParseClosureCommand(commandLine) with
            | Some cmd -> cmd
            | None -> failwith "Failed to parse asynchronous command."

        Assert.Equal(ExecutionMode.Asynchronous, command.Mode)
        Assert.Equal(Some "test-run", command.Context |> Map.tryFind "note" |> Option.map string)
        Assert.True(command.Parameters |> Map.containsKey "grammars")

        let placeholderResult =
            service.ExecuteClosureCommand(command)
            |> Async.RunSynchronously

        Assert.True(placeholderResult.Success)
        Assert.Equal("Command scheduled for asynchronous execution.", placeholderResult.OutputSummary)
        Assert.Equal("pending", placeholderResult.EvolutionData |> Map.tryFind "status" |> Option.map string |> Option.defaultValue "")

        let rec awaitCompletion attempts =
            if attempts = 0 then false
            else
                match service.TryGetActiveCommand(command.CommandId) with
                | None -> true
                | Some _ ->
                    Async.Sleep 20 |> Async.RunSynchronously
                    awaitCompletion (attempts - 1)

        Assert.True(awaitCompletion 100, "Asynchronous closure did not complete in time.")
        Assert.True(service.TryGetActiveCommand(command.CommandId).IsNone)

        let historyEntries =
            service.GetExecutionHistory()
            |> Array.filter (fun entry -> entry.CommandId = command.CommandId)

        Assert.True(historyEntries.Length >= 2)
        let finalResult =
            historyEntries
            |> Array.tryFind (fun entry -> entry.OutputSummary.Contains("Grammar evolution"))
            |> Option.defaultWith (fun () -> failwith "Asynchronous command did not record a final result.")
        Assert.True(finalResult.Success, finalResult.Error |> Option.defaultValue "")

    [<Fact>]
    let ``Spec contract metadata attaches to metascript command`` () =
        let specPath =
            Path.Combine(
                Path.GetTempPath(),
                "speckit_" + Guid.NewGuid().ToString("N") + ".md"
            )

        let specContent =
            """
# Feature Specification: Spec Grammar Quality

**Feature Branch**: `123-spec-grammar`
**Created**: 2025-03-01
**Status**: Review

### User Story 1 - Deterministic ranking (Priority: P1)

**Acceptance Scenarios**:
1. Given grammars with usage counts, When evaluating, Then the deterministic ranking favours balanced usage.

### Edge Cases
- What happens when no grammars are provided?
"""

        File.WriteAllText(specPath, specContent.Trim())

        try
            let commandLine =
                sprintf
                    "CLOSURE_CREATE GRAMMAR_EVOLUTION \"SpecGrammar\" team_name=\"Beta\" evolution_goal=\"Quality\" grammars='[{\"name\":\"alpha\",\"rules\":[\"start\"],\"usage\":1}]' spec_contract=\"@%s\""
                    (specPath.Replace("\\", "/"))

            let command =
                match service.ParseClosureCommand(commandLine) with
                | Some cmd -> cmd
                | None -> failwith "Failed to parse spec-driven grammar command."

            let specSummary =
                match command.Context |> Map.tryFind "spec.contract.summary" with
                | Some (:? SpecKitSummary as summary) -> summary
                | _ -> failwith "Spec contract summary not attached to command context."

            Assert.Equal("Spec Grammar Quality", specSummary.Title)
            Assert.Equal(1, specSummary.UserStories.Length)
            Assert.Equal(Some "Review", specSummary.Status)

            let result =
                service.ExecuteClosureCommand(command)
                |> Async.RunSynchronously

            Assert.True(result.Success, result.Error |> Option.defaultValue "")

            let storyCount =
                result.EvolutionData
                |> Map.tryFind "spec.contract.user_story_count"
                |> Option.map unbox<int>
                |> Option.defaultValue -1
            Assert.Equal(1, storyCount)

            let acceptanceCount =
                result.EvolutionData
                |> Map.tryFind "spec.contract.acceptance_count"
                |> Option.map unbox<int>
                |> Option.defaultValue -1
            Assert.Equal(1, acceptanceCount)

            Assert.True(
                result.Artifacts
                |> List.exists (fun artifact -> artifact.Contains("Spec contract \"Spec Grammar Quality\" captured")),
                "Spec contract artifact summary should be present.")

            Assert.Contains(
                "Validate acceptance scenarios defined in spec contract.",
                result.NextSteps)
        finally
            if File.Exists(specPath) then
                File.Delete(specPath)

    [<Fact>]
    let ``Metascript integration executes spec-driven dynamic command`` () =
        let doc = MetascriptSpecLoader.loadFromFile (specPath "swarm_plan.md")
        let commandLine =
            sprintf "CLOSURE_CREATE DYNAMIC_METASCRIPT \"SpecPlan\" spec=\"%s\"" doc.SourcePath

        let command =
            match service.ParseClosureCommand(commandLine) with
            | Some cmd -> cmd
            | None -> failwith "Failed to parse spec-driven command."

        let result =
            service.ExecuteClosureCommand(command)
            |> Async.RunSynchronously

        Assert.True(result.Success, result.Error |> Option.defaultValue "")

        let summary = result.Output |> Option.get :?> DynamicMetascriptSummary
        Assert.Equal(doc.Id, summary.Name)
        Assert.Equal(Some doc.SourcePath, summary.SourcePath)

        let expectations = doc.Expectations

        let expectInt key =
            expectations
            |> Map.tryFind key
            |> Option.map (fun value -> Int32.Parse(value, CultureInfo.InvariantCulture))

        expectInt "spawn_count" |> Option.iter (fun expected -> Assert.Equal(expected, summary.SpawnedAgents.Length))
        expectInt "connection_count" |> Option.iter (fun expected -> Assert.Equal(expected, summary.Connections.Length))

        expectations
        |> Map.toList
        |> List.filter (fun (key, _) -> key.StartsWith("metric."))
        |> List.iter (fun (key, value) ->
            let metricName = key.Substring("metric.".Length)
            let expected = Double.Parse(value, CultureInfo.InvariantCulture)
            let actual = summary.Metrics |> Map.tryFind metricName |> Option.defaultValue -1.0
            Assert.InRange(actual, expected - 0.01, expected + 0.01))

        Assert.True(
            result.NextSteps
            |> List.exists (fun step -> step.Contains("Validate dynamic closure grammar")))

    [<Fact>]
    let ``DSL resources are ingested and parsed from external files`` () =
        let tempFile =
            Path.Combine(
                Path.GetTempPath(),
                "metascript_" + Guid.NewGuid().ToString("N") + ".trsx")

        let metascriptContent =
            ("""
            # Sample metascript
            SPAWN QRE 2 HIERARCHICAL
            CONNECT leader agent-1 command
            METRIC stability 0.9
            """).Trim()

        File.WriteAllText(tempFile, metascriptContent)

        try
            let commandLine =
                sprintf "CLOSURE_CREATE DYNAMIC_METASCRIPT \"Generated\" grammar=\"@%s\"" tempFile

            let command =
                match service.ParseClosureCommand(commandLine) with
                | Some cmd -> cmd
                | None -> failwith "Failed to parse DSL-augmented command."

            let grammarContent =
                command.Parameters
                |> Map.tryFind "grammar"
                |> Option.map (fun value -> value :?> string)
                |> Option.defaultValue ""

            Assert.Equal(metascriptContent, grammarContent)

            let resourceInfo =
                match command.Context |> Map.tryFind "dsl.source.grammar" with
                | Some value -> value :?> TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService.DslResourceInfo
                | None -> failwith "DSL source metadata not captured."

            Assert.Equal(Path.GetFullPath(tempFile), resourceInfo.Path)
            Assert.Equal("trsx", resourceInfo.Kind)

            let parsedBlock =
                match command.Context |> Map.tryFind "dsl.parsed.grammar" with
                | Some value -> Some(value :?> FractalMetascriptBlock)
                | None -> None

            Assert.True(Option.isSome parsedBlock, "Fractal DSL should be parsed into a metascript block.")

            let immediateResult =
                service.ExecuteClosureCommand(command)
                |> Async.RunSynchronously

            Assert.True(immediateResult.EvolutionData |> Map.containsKey "dsl.metascript.rule_count")
            Assert.True(immediateResult.EvolutionData |> Map.containsKey "dsl.metascript.source")

            Assert.Contains("Dynamic metascript", immediateResult.OutputSummary)
            let sourcePath =
                immediateResult.EvolutionData
                |> Map.tryFind "dsl.metascript.source"
                |> Option.map (fun value -> value :?> string)
                |> Option.defaultValue ""
            Assert.Equal(Path.GetFullPath(tempFile), sourcePath)
            Assert.True(
                immediateResult.Artifacts
                |> List.exists (fun artifact -> artifact.Contains("dynamic metascript")))
        finally
            if File.Exists(tempFile) then
                File.Delete(tempFile)

    [<Fact>]
    let ``Dynamic metascript closure summarizes DSL metrics and structure`` () =
        let doc = MetascriptSpecLoader.loadFromFile (specPath "dynamic_plan.md")
        let parser = FractalMetascriptParser()
        let block = parser.ParseFractalMetascript(doc.Grammar)
        let dynamicSpec = blockToDynamicSpec doc.Id (Some doc.SourcePath) (Some doc.Expectations) block
        let closure = factory.CreateClosure(DynamicMetascriptClosure dynamicSpec, doc.Id, Map.empty)
        let result = closure Map.empty |> Async.RunSynchronously

        Assert.True(result.Success, result.Error |> Option.defaultValue "")

        let summary = result.Output |> Option.get :?> DynamicMetascriptSummary
        Assert.Equal(doc.Id, summary.Name)
        Assert.Equal(Some doc.SourcePath, summary.SourcePath)
        Assert.Equal(2, summary.SpawnedAgents.Length)
        Assert.Equal(2, summary.Connections.Length)

        let expectations = doc.Expectations

        let tryInt key =
            expectations
            |> Map.tryFind key
            |> Option.map (fun value -> Int32.Parse(value, CultureInfo.InvariantCulture))

        let tryFloat key =
            expectations
            |> Map.tryFind key
            |> Option.map (fun value -> Double.Parse(value, CultureInfo.InvariantCulture))

        let evolutionData = result.EvolutionData

        let ruleCount =
            evolutionData
            |> Map.tryFind "dsl.metascript.rule_count"
            |> Option.map (fun value -> unbox<int> value)
            |> Option.defaultValue -1
        tryInt "rules" |> Option.iter (fun expected -> Assert.Equal(expected, ruleCount))

        let maxDepth =
            evolutionData
            |> Map.tryFind "dsl.metascript.max_depth"
            |> Option.map (fun value -> unbox<int> value)
            |> Option.defaultValue -1
        tryInt "max_depth" |> Option.iter (fun expected -> Assert.Equal(expected, maxDepth))

        let spawnCount =
            evolutionData
            |> Map.tryFind "dsl.metascript.spawn_count"
            |> Option.map (fun value -> unbox<int> value)
            |> Option.defaultValue -1
        tryInt "spawn_count" |> Option.iter (fun expected ->
            Assert.Equal(expected, spawnCount)
            Assert.Equal(expected, summary.SpawnedAgents.Length))

        let connectionCount =
            evolutionData
            |> Map.tryFind "dsl.metascript.connection_count"
            |> Option.map (fun value -> unbox<int> value)
            |> Option.defaultValue -1
        tryInt "connection_count" |> Option.iter (fun expected ->
            Assert.Equal(expected, connectionCount)
            Assert.Equal(expected, summary.Connections.Length))

        let repeatedPatterns = summary.RepeatedPatterns |> List.map fst
        expectations |> Map.tryFind "pattern" |> Option.iter (fun pattern -> Assert.Contains(pattern, repeatedPatterns))

        expectations
        |> Map.toList
        |> List.filter (fun (key, _) -> key.StartsWith "metric.")
        |> List.iter (fun (key, value) ->
            let metricName = key.Substring("metric.".Length)
            let expected = Double.Parse(value, CultureInfo.InvariantCulture)
            let actual = summary.Metrics |> Map.tryFind metricName |> Option.defaultValue -1.0
            Assert.InRange(actual, expected - 0.01, expected + 0.01))

        Assert.True(
            result.NextEvolutionSteps
            |> List.exists (fun step -> step.Contains("Validate dynamic closure grammar")))

    [<Fact>]
    let ``Dynamic metascript closure handles varied grammars`` () =
        let specFiles = [ "swarm_plan.md"; "specialized_plan.md" ]
        let parser = FractalMetascriptParser()

        for file in specFiles do
            let doc = MetascriptSpecLoader.loadFromFile (specPath file)
            let block = parser.ParseFractalMetascript(doc.Grammar)
            let dynamicSpec = blockToDynamicSpec doc.Id (Some doc.SourcePath) (Some doc.Expectations) block
            let closure = factory.CreateClosure(DynamicMetascriptClosure dynamicSpec, doc.Id, Map.empty)
            let result = closure Map.empty |> Async.RunSynchronously

            Assert.True(result.Success, result.Error |> Option.defaultValue "")

            let summary = result.Output |> Option.get :?> DynamicMetascriptSummary
            Assert.Equal(doc.Id, summary.Name)
            Assert.Equal(Some doc.SourcePath, summary.SourcePath)

            let expectations = doc.Expectations

            let expectInt key = expectations |> Map.tryFind key |> Option.map int

            let evolutionData = result.EvolutionData

            let ruleCount =
                evolutionData
                |> Map.tryFind "dsl.metascript.rule_count"
                |> Option.map (fun value -> unbox<int> value)
                |> Option.defaultValue -1
            expectInt "rules" |> Option.iter (fun expected -> Assert.Equal(expected, ruleCount))

            let maxDepth =
                evolutionData
                |> Map.tryFind "dsl.metascript.max_depth"
                |> Option.map (fun value -> unbox<int> value)
                |> Option.defaultValue -1
            expectInt "max_depth" |> Option.iter (fun expected -> Assert.Equal(expected, maxDepth))

            let spawnCount =
                evolutionData
                |> Map.tryFind "dsl.metascript.spawn_count"
                |> Option.map (fun value -> unbox<int> value)
                |> Option.defaultValue -1
            expectInt "spawn_count" |> Option.iter (fun expected ->
                Assert.Equal(expected, spawnCount)
                Assert.Equal(expected, summary.SpawnedAgents.Length))

            let connectionCount =
                evolutionData
                |> Map.tryFind "dsl.metascript.connection_count"
                |> Option.map (fun value -> unbox<int> value)
                |> Option.defaultValue -1
            expectInt "connection_count" |> Option.iter (fun expected ->
                Assert.Equal(expected, connectionCount)
                Assert.Equal(expected, summary.Connections.Length))

            match expectations |> Map.tryFind "pattern" with
            | Some pattern -> Assert.Contains(pattern, summary.RepeatedPatterns |> List.map fst)
            | None -> ()

            expectations
            |> Map.toList
            |> List.filter (fun (key, _) -> key.StartsWith("metric."))
            |> List.iter (fun (key, value) ->
                let metricName = key.Substring("metric.".Length)
                let expected = Double.Parse(value, CultureInfo.InvariantCulture)
                let actual = summary.Metrics |> Map.tryFind metricName |> Option.defaultValue -1.0
                Assert.InRange(actual, expected - 0.01, expected + 0.01))

            Assert.True(
                result.NextEvolutionSteps
                |> List.exists (fun step -> step.Contains("Validate dynamic closure grammar")))

    [<Fact>]
    let ``Fractal metascript parser reads generated scripts`` () =
        let generator = FractalMetascriptGenerator()
        let parser = FractalMetascriptParser()

        let script = generator.GenerateTeamCoordinationMetascript(4, CoordinationStrategy.Swarm)
        let block = parser.ParseFractalMetascript(script)

        Assert.Equal("Parsed Fractal Metascript", block.Name)
        Assert.True(block.Rules.Length >= 4)
        Assert.True(block.MaxDepth >= 1)





