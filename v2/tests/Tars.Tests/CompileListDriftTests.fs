namespace Tars.Tests

open System
open System.IO
open System.Xml.Linq
open Xunit

/// Issue #224 — F# projects do not glob `.fs` files, so a file on disk that is missing
/// from its project's `<Compile>` list is never type-checked, never built, never run,
/// and nothing says so. That silence hid 64 files (~9,800 lines): whole CLI commands,
/// 20 test files that looked like coverage, one provably broken module, and a file
/// PR #236 edited without the build ever checking the edit.
///
/// This guard makes exclusion *declared* instead of silent. `knownExclusions` is the
/// burn-down list: shrink it as files are restored or deleted. Never grow it to quiet a
/// new failure — a new entry means a file is being added to the tree dead on arrival.
module CompileListDriftTests =

    /// Paths relative to `v2/`. Sorted. Captured 2026-09-12 from
    /// `dotnet msbuild -getItem:Compile`, not from this file's own parser.
    let private knownExclusions: string list =
        [ "src/Tars.Core/Example.fs"
          "src/Tars.Cortex/AdvancedPrompting.fs"
          "src/Tars.Cortex/AgentLifecycleAgent.fs"
          "src/Tars.Cortex/ArchitecturalReflection.fs"
          "src/Tars.Cortex/CognitiveGrounding.fs"
          "src/Tars.Cortex/ContextManager.fs"
          "src/Tars.Cortex/DiagnosticsAgent.fs"
          "src/Tars.Cortex/LedgerAwarePrompting.fs"
          "src/Tars.Cortex/MemoryScoringAgent.fs"
          "src/Tars.Cortex/Retrieval.fs"
          "src/Tars.Cortex/TermFrequencyStore.fs"
          "src/Tars.Cortex/TokenCounting.fs"
          "src/Tars.Evolution/DemoVisualizationTests.fs"
          "src/Tars.Evolution/NeuroSymbolicIntegration.fs"
          "src/Tars.Interface.Cli/AgentLifecycleBackgroundService.fs"
          "src/Tars.Interface.Cli/Commands/CritiqueCmd.fs"
          "src/Tars.Interface.Cli/Commands/ExtendCommand.fs"
          "src/Tars.Interface.Cli/Commands/GroundingCommands.fs"
          "src/Tars.Interface.Cli/Commands/IngestCommand.fs"
          "src/Tars.Interface.Cli/Commands/IngestRdfCommand.fs"
          "src/Tars.Interface.Cli/Commands/LlamaServer.fs"
          "src/Tars.Interface.Cli/Commands/LlmTest.fs"
          "src/Tars.Interface.Cli/Commands/LodCommand.fs"
          "src/Tars.Interface.Cli/Commands/QuerySparqlCommand.fs"
          "src/Tars.Interface.Cli/Commands/RefactorCommand.fs"
          "src/Tars.Interface.Cli/Commands/ReflectCommand.fs"
          "src/Tars.Interface.Cli/Commands/ResearchSynthesis.fs"
          "src/Tars.Interface.Cli/Commands/Run.fs"
          "src/Tars.Interface.Cli/Commands/SearchCodeCommand.fs"
          "src/Tars.Interface.Cli/Commands/TestGrammar.fs"
          "src/Tars.Interface.Cli/InfrastructureServer.fs"
          "src/Tars.Interface.Cli/ReflectionBackgroundService.fs"
          "src/Tars.Interface.Cli/ReflectionService.fs"
          "src/Tars.Knowledge/HybridPlanStorage.fs"
          "src/Tars.Knowledge/IngestionPipeline.fs"
          "src/Tars.Knowledge/ReflectionAgent.fs"
          "src/Tars.Knowledge/WikipediaExtractor.fs"
          "src/Tars.LinkedData/Library.fs"
          "src/Tars.Metascript/IrCompiler.fs"
          "src/Tars.Metascript/MetascriptTools.fs"
          "src/Tars.Metascript/TrsxParser.fs"
          "src/Tars.Metascript/WotCompiler.fs"
          "src/Tars.Metascript/WotEngine.fs"
          "src/Tars.Symbolic/SymbolicTypes.fs"
          "tests/Tars.Tests/CognitionCompilerTests.fs"
          "tests/Tars.Tests/ConstitutionTests.fs"
          "tests/Tars.Tests/ContextCompressionTests.fs"
          "tests/Tars.Tests/ContextEngineeringTests.fs"
          "tests/Tars.Tests/EvolutionFixesTests.fs"
          "tests/Tars.Tests/FSharpToolsTests.fs"
          "tests/Tars.Tests/FunctionalPatternsTests.fs"
          "tests/Tars.Tests/GroundingTests.fs"
          "tests/Tars.Tests/IntegrationTests.fs"
          "tests/Tars.Tests/KnowledgeTests.fs"
          "tests/Tars.Tests/MetricsTests.fs"
          "tests/Tars.Tests/OfflineEval.fs"
          "tests/Tars.Tests/PreLlmPipelineTests.fs"
          "tests/Tars.Tests/RdfParserTests.fs"
          "tests/Tars.Tests/RefactoringTaskTests.fs"
          "tests/Tars.Tests/SelfExtensionTests.fs"
          "tests/Tars.Tests/ToleranceAndGuardTests.fs"
          "tests/Tars.Tests/ToleranceEngineeringTests.fs"
          "tests/Tars.Tests/ValidationTests.fs"
          "tests/Tars.Tests/WotIntegrationTests.fs" ]

    type private ProjectScan =
        { Project: string
          Excluded: string list
          Unmodellable: string list }

    let private v2Root =
        lazy
            (let rec find (dir: DirectoryInfo) =
                if isNull dir then
                    failwith "Could not locate v2/ (no Tars.sln above the test directory)"
                elif dir.GetFiles("Tars.sln").Length > 0 then
                    dir.FullName
                else
                    find dir.Parent

             find (DirectoryInfo(Directory.GetCurrentDirectory())))

    let private toV2Relative (fullPath: string) =
        Path.GetRelativePath(v2Root.Value, fullPath).Replace('\\', '/')

    let private isBuildOutput (projectDir: string) (file: string) =
        Path.GetRelativePath(projectDir, file).Replace('\\', '/').Split('/')
        |> Array.exists (fun seg -> seg = "obj" || seg = "bin")

    let private scan (fsproj: string) : ProjectScan =
        let projectDir = Path.GetDirectoryName fsproj
        // XDocument drops comments, so `<!-- <Compile Include="X.fs" /> -->` correctly
        // counts as excluded. A regex over the raw text does not: it matches inside
        // comments, which is how two files first went uncounted.
        let doc = XDocument.Load fsproj

        let compileItems =
            doc.Descendants()
            |> Seq.filter (fun e -> e.Name.LocalName = "Compile")
            |> List.ofSeq

        let attr (name: string) (e: XElement) =
            match e.Attribute(XName.Get name) with
            | null -> None
            | a -> Some a.Value

        // The static model below is exact only while nothing needs MSBuild to evaluate.
        let unmodellable =
            compileItems
            |> List.choose (fun e ->
                let conditioned =
                    (attr "Condition" e).IsSome
                    || (match e.Parent with
                        | null -> false
                        | p -> (attr "Condition" p).IsSome)

                let wildcard =
                    [ attr "Include" e; attr "Remove" e ]
                    |> List.exists (fun v -> v |> Option.exists (fun s -> s.Contains '*' || s.Contains '?'))

                if conditioned || wildcard then
                    Some(e.ToString(SaveOptions.DisableFormatting))
                else
                    None)

        let resolve (value: string) =
            Path.GetFullPath(Path.Combine(projectDir, value.Replace('\\', '/')))

        let paths name =
            compileItems
            |> List.choose (attr name)
            |> List.map resolve
            |> fun xs -> Collections.Generic.HashSet<string>(xs, StringComparer.OrdinalIgnoreCase)

        let included = paths "Include"
        let removed = paths "Remove"
        included.ExceptWith removed

        let excluded =
            Directory.EnumerateFiles(projectDir, "*.fs", SearchOption.AllDirectories)
            |> Seq.filter (fun f -> not (isBuildOutput projectDir f))
            |> Seq.filter (fun f -> not (included.Contains(Path.GetFullPath f)))
            |> Seq.map toV2Relative
            |> List.ofSeq

        { Project = toV2Relative fsproj
          Excluded = excluded
          Unmodellable = unmodellable }

    let private allScans =
        lazy
            ([ "src"; "tests" ]
             |> List.collect (fun sub ->
                 Directory.EnumerateFiles(Path.Combine(v2Root.Value, sub), "*.fsproj", SearchOption.AllDirectories)
                 |> List.ofSeq)
             |> List.map scan)

    let private actualExclusions () =
        allScans.Value |> List.collect (fun s -> s.Excluded) |> Set.ofList

    [<Fact>]
    let ``scan covers the real tree rather than passing on an empty one`` () =
        let projects = allScans.Value.Length

        let files =
            [ "src"; "tests" ]
            |> List.sumBy (fun sub ->
                Directory.EnumerateFiles(Path.Combine(v2Root.Value, sub), "*.fs", SearchOption.AllDirectories)
                |> Seq.length)

        Assert.True(projects >= 15, $"Expected at least 15 .fsproj files under v2/, found {projects}")
        Assert.True(files >= 400, $"Expected at least 400 .fs files under v2/, found {files}")

    [<Fact>]
    let ``compile items use only constructs the static parse can model`` () =
        let offenders =
            allScans.Value
            |> List.collect (fun s -> s.Unmodellable |> List.map (fun x -> $"{s.Project}: {x}"))

        Assert.True(
            List.isEmpty offenders,
            "A <Compile> item uses a Condition or wildcard, which this guard cannot evaluate "
            + "statically. Either avoid it or switch the guard to `dotnet msbuild -getItem:Compile`:\n  "
            + String.Join("\n  ", offenders)
        )

    [<Fact>]
    let ``every F# file on disk is compiled or explicitly declared excluded`` () =
        let undeclared =
            Set.difference (actualExclusions ()) (Set.ofList knownExclusions) |> Set.toList

        Assert.True(
            List.isEmpty undeclared,
            "These .fs files exist on disk but are not in their project's <Compile> list, so "
            + "they are never built. Add each to its .fsproj, or delete it:\n  "
            + String.Join("\n  ", undeclared)
        )

    [<Fact>]
    let ``the exclusion allowlist has no stale entries`` () =
        let stale =
            Set.difference (Set.ofList knownExclusions) (actualExclusions ()) |> Set.toList

        Assert.True(
            List.isEmpty stale,
            "These knownExclusions entries are no longer excluded (now compiled, or deleted). "
            + "Remove them so the burn-down list stays accurate:\n  "
            + String.Join("\n  ", stale)
        )
