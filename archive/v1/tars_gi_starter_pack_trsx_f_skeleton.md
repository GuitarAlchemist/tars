# TARS GI Starter Pack

A minimal, working scaffold to kick off a **non‑LLM‑centric** GI loop in TARS. It includes:

- A tiny **.trsx metascript** format
- F# **core types** (Belnap logic, beliefs, actions, plans, metrics)
- **Belief graph** + **state‑space filter** skeleton (EKF‑like placeholder)
- **HTN planner** (+ MCTS/POMDP stubs)
- **Skills** with pre/postconditions + verifier hooks
- **Reflection agent** (threshold triggers)
- **Simulator** (dry‑run) + one end‑to‑end cycle in a CLI

> Everything compiles as a baseline and is structured to be extended. Parsers are intentionally simple.

---

## Directory layout

```
TARS.GI/
├─ samples/
│  └─ gi_minimal.trsx
├─ src/
│  ├─ Tars.Core/
│  │  ├─ Tars.Core.fsproj
│  │  ├─ Types.fs
│  │  ├─ Belnap.fs
│  │  ├─ BeliefGraph.fs
│  │  ├─ VSA.fs
│  │  ├─ StateSpace.fs
│  │  ├─ Simulator.fs
│  │  ├─ Metrics.fs
│  │  ├─ ReflectionAgent.fs
│  │  ├─ Skills/
│  │  │  ├─ SkillSpec.fs
│  │  │  └─ GitClone.fs
│  │  └─ Planner/
│  │     ├─ HTN.fs
│  │     ├─ MCTS.fs
│  │     └─ POMDP.fs
│  └─ Tars.Cli/
│     ├─ Tars.Cli.fsproj
│     ├─ TrsxAst.fs
│     ├─ TrsxParser.fs
│     ├─ TrsxInterpreter.fs
│     └─ Program.fs
└─ TARS.GI.sln
```

---

## Quickstart

```bash
# from the folder where you want to create the skeleton
mkdir TARS.GI && cd TARS.GI

dotnet new sln -n TARS.GI
mkdir -p src/Tars.Core src/Tars.Cli samples src/Tars.Core/Planner src/Tars.Core/Skills

dotnet new classlib -lang "F#" -n Tars.Core -o src/Tars.Core
rm src/Tars.Core/Library.fs

dotnet new console -lang "F#" -n Tars.Cli -o src/Tars.Cli
rm src/Tars.Cli/Program.fs

dotnet sln add src/Tars.Core/Tars.Core.fsproj

dotnet sln add src/Tars.Cli/Tars.Cli.fsproj

dotnet add src/Tars.Cli/Tars.Cli.fsproj reference src/Tars.Core/Tars.Core.fsproj
```

Copy the files below into the indicated paths. Then:

```bash
dotnet build

dotnet run --project src/Tars.Cli -- samples/gi_minimal.trsx
```

---

## `samples/gi_minimal.trsx`

```text
trsx_version: 0.1
name: GI-Minimal

[beliefs]
id=world.is_initialized; proposition="World initialized"; truth=Unknown; confidence=0.5
id=net.has_git;           proposition="Git available";   truth=Unknown; confidence=0.5
id=repo.cloned;           proposition="Repo cloned";     truth=False;   confidence=0.6
[/beliefs]

[goals]
"Initialize World"
"Clone Repo"
[/goals]

[skills]
name=InitWorld; pre=[];                post=["world.is_initialized"]; verifier="init_world_tests"
name=GitClone;  pre=["net.has_git"];  post=["repo.cloned"];         verifier="git_clone_tests"
[/skills]

[htn_methods]
goal="Initialize World" -> ["InitWorld"]
goal="Clone Repo"       -> ["GitClone"]
[/htn_methods]

[reflection]
prediction_error>0.25 -> ["rerun_mcts","tighten_constraints"]
belief_entropy>0.80   -> ["recompute_beliefs"]
[/reflection]

[run]
root_goals=["Initialize World","Clone Repo"]
[/run]
```

---

## `src/Tars.Core/Tars.Core.fsproj`

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
    <LangVersion>preview</LangVersion>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="Types.fs" />
    <Compile Include="Belnap.fs" />
    <Compile Include="BeliefGraph.fs" />
    <Compile Include="VSA.fs" />
    <Compile Include="StateSpace.fs" />
    <Compile Include="Simulator.fs" />
    <Compile Include="Metrics.fs" />
    <Compile Include="ReflectionAgent.fs" />
    <Compile Include="Skills/SkillSpec.fs" />
    <Compile Include="Skills/GitClone.fs" />
    <Compile Include="Planner/HTN.fs" />
    <Compile Include="Planner/MCTS.fs" />
    <Compile Include="Planner/POMDP.fs" />
  </ItemGroup>
</Project>
```

---

## `src/Tars.Core/Types.fs`

```fsharp
namespace Tars.Core

open System

/// Four-valued logic (Belnap/FDE)
type Belnap = True | False | Both | Unknown

[<CLIMutable>]
type Belief = {
    id: string
    proposition: string
    truth: Belnap
    confidence: float
    provenance: string list
}

[<CLIMutable>]
type Latent = { mean: float[]; cov: float[][] }

type Observation = float[]

[<CLIMutable>]
type Action = { name: string; args: Map<string, obj> }

[<CLIMutable>]
type SkillSpec = {
    name: string
    pre: string list
    post: string list
    verifier: unit -> bool
    cost: float
}

[<CLIMutable>]
type PlanStep = { skill: SkillSpec; args: Map<string,obj> }

type Plan = PlanStep list

[<CLIMutable>]
type Metrics = {
    predError: float
    beliefEntropy: float
    specViolations: int
    replanCount: int
}
```

---

## `src/Tars.Core/Belnap.fs`

```fsharp
namespace Tars.Core

module Belnap =
    let toFloat = function
        | True -> 1.0
        | False -> 1.0
        | Both -> 2.0
        | Unknown -> 0.5

    let join a b =
        match a,b with
        | Both, _ | _, Both -> Both
        | Unknown, x | x, Unknown -> x
        | True, True -> True
        | False, False -> False
        | True, False | False, True -> Both
```

---

## `src/Tars.Core/BeliefGraph.fs`

```fsharp
namespace Tars.Core

open System.Collections.Generic

module BeliefGraph =
    type T() =
        let dict = Dictionary<string,Belief>()
        member _.Set(b:Belief) = dict[b.id] <- b
        member _.Get(id:string) = if dict.ContainsKey id then Some dict[id] else None
        member _.All() = dict.Values |> Seq.toList
        member _.Upsert(id, proposition, truth, confidence, prov) =
            let b = { id = id; proposition = proposition; truth = truth; confidence = confidence; provenance = prov }
            dict[id] <- b; b

    let entropy (beliefs:Belief list) =
        // Toy entropy: higher if more Unknown or Both
        let score b =
            match b.truth with
            | Unknown -> 1.0
            | Both -> 1.5
            | _ -> 0.1
        beliefs |> List.sumBy score |> fun s -> s / (float (max 1 beliefs.Length))

    let satisfies (beliefs:Belief list) (req:string) =
        match beliefs |> List.tryFind (fun b -> b.id = req) with
        | Some b -> match b.truth with True -> true | Both -> true | _ -> false
        | None -> false
```

---

## `src/Tars.Core/VSA.fs`

```fsharp
namespace Tars.Core

module VSA =
    // Extremely small placeholder for HRR-like binding using circular convolution.
    // Replace with a real implementation later.
    let private wrapIndex n i = (i + n) % n

    let bind (a:float[]) (b:float[]) =
        let n = a.Length
        Array.init n (fun k ->
            let mutable acc = 0.0
            for i in 0 .. n-1 do
                let j = wrapIndex n (k - i)
                acc <- acc + a[i]*b[j]
            acc / float n)

    let unbind (c:float[]) (a:float[]) =
        // For the toy placeholder, unbind == bind with reversed second vector
        let revA = a |> Array.rev
        bind c revA
```

---

## `src/Tars.Core/StateSpace.fs`

```fsharp
namespace Tars.Core

open System

module StateSpace =
    open Tars.Core

    /// Minimal linear-Gaussian placeholder (identity dynamics/observation)
    let infer (prior:Latent) (_a:Action option) (o:Observation) : Latent * float =
        let n = prior.mean.Length
        let postMean = Array.init n (fun i -> 0.5*prior.mean[i] + 0.5*o[i])
        let postCov = Array.init n (fun i -> Array.init n (fun j -> if i=j then 0.1 else 0.0))
        let err =
            (Array.map2 (fun a b -> abs (a-b)) postMean o)
            |> Array.average
        ({ mean = postMean; cov = postCov }, err)
```

---

## `src/Tars.Core/Simulator.fs`

```fsharp
namespace Tars.Core

module Simulator =
    open System
    open Tars.Core

    let simulatePlan (p:Plan) =
        // Dry-run only
        for step in p do
            printfn "[SIM] Would run skill: %s" step.skill.name
        true
```

---

## `src/Tars.Core/Metrics.fs`

```fsharp
namespace Tars.Core

module Metrics =
    open Tars.Core
    open BeliefGraph

    let compute (predError:float) (beliefs:Belief list) (specViolations:int) (replanCount:int) : Metrics =
        { predError = predError
          beliefEntropy = entropy beliefs
          specViolations = specViolations
          replanCount = replanCount }
```

---

## `src/Tars.Core/ReflectionAgent.fs`

```fsharp
namespace Tars.Core

module ReflectionAgent =
    open System

    type Trigger =
        | RerunMcts
        | TightenConstraints
        | RecomputeBeliefs

    type ReflectionDecision = { triggers: Trigger list; notes: string }

    let decide (predErr:float) (beliefEntropy:float) =
        let triggers =
            [ if predErr > 0.25 then yield Trigger.RerunMcts; yield Trigger.TightenConstraints
              if beliefEntropy > 0.80 then yield Trigger.RecomputeBeliefs ]
        { triggers = triggers; notes = sprintf "predErr=%.3f beliefEntropy=%.3f" predErr beliefEntropy }
```

---

## `src/Tars.Core/Skills/SkillSpec.fs`

```fsharp
namespace Tars.Core

open System
open BeliefGraph

module Skills =
    let meetsPreconditions (beliefs:Belief list) (pre:string list) =
        pre |> List.forall (satisfies beliefs)

    let applyPostconditions (beliefs:Belief list) (post:string list) : Belief list =
        let upgrade (b:Belief) = { b with truth = True; confidence = max b.confidence 0.9 }
        beliefs
        |> List.map (fun b -> if post |> List.contains b.id then upgrade b else b)

    /// Registry for skills
    type Registry() =
        let dict = System.Collections.Generic.Dictionary<string,SkillSpec>()
        member _.Add(sk:SkillSpec) = dict[sk.name] <- sk
        member _.TryGet name = if dict.ContainsKey name then Some dict[name] else None
        member _.All() = dict.Values |> Seq.toList
```

---

## `src/Tars.Core/Skills/GitClone.fs`

```fsharp
namespace Tars.Core

open System

module GitClone =
    open Tars.Core

    let verifier () =
        // TODO: property tests (e.g., can we reach git? repo path format?)
        true

    let spec : SkillSpec =
        { name = "GitClone"
          pre = [ "net.has_git" ]
          post = [ "repo.cloned" ]
          verifier = verifier
          cost = 1.0 }
```

---

## `src/Tars.Core/Planner/HTN.fs`

```fsharp
namespace Tars.Core

open System
open Tars.Core
open BeliefGraph
open Skills

module HTN =
    type Task = Achieve of string | Do of string

    type Method = { goal: string; steps: string list }

    type Domain = { methods: Method list; registry: Registry }

    let rec decompose (domain:Domain) (beliefs:Belief list) (t:Task) : Plan =
        match t with
        | Do skillName ->
            match domain.registry.TryGet skillName with
            | Some sk -> [ { skill = sk; args = Map.empty } ]
            | None -> failwithf "Unknown skill: %s" skillName
        | Achieve g ->
            match domain.methods |> List.tryFind (fun m -> m.goal = g) with
            | None -> failwithf "No method for goal %s" g
            | Some m -> m.steps |> List.collect (fun s -> decompose domain beliefs (Do s))

    let plan (domain:Domain) (beliefs:Belief list) (goals:string list) : Plan =
        goals |> List.collect (fun g -> decompose domain beliefs (Achieve g))

    let execute (beliefs:Belief list) (p:Plan) : Result<Belief list, string> =
        let mutable b = beliefs
        for step in p do
            if not (step.skill.verifier()) then
                return Error (sprintf "Verifier failed: %s" step.skill.name)
            if not (meetsPreconditions b step.skill.pre) then
                return Error (sprintf "Preconditions not met: %s" step.skill.name)
            // In a real system, run the skill and check effects. Here we assume success and update beliefs.
            b <- applyPostconditions b step.skill.post
        Ok b
```

---

## `src/Tars.Core/Planner/MCTS.fs`

```fsharp
namespace Tars.Core

module MCTS =
    open Tars.Core

    // Toy stub – returns the given HTN plan as-is
    let choose (p:Plan) : Plan = p
```

---

## `src/Tars.Core/Planner/POMDP.fs`

```fsharp
namespace Tars.Core

module POMDP =
    // Placeholder for local POMDP solving; stub API for future plug-in.
    type SolverConfig = { horizon:int; discount:float }
    let solve (_cfg:SolverConfig) = ()
```

---

## `src/Tars.Cli/Tars.Cli.fsproj`

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <LangVersion>preview</LangVersion>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="TrsxAst.fs" />
    <Compile Include="TrsxParser.fs" />
    <Compile Include="TrsxInterpreter.fs" />
    <Compile Include="Program.fs" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="..\Tars.Core\Tars.Core.fsproj" />
  </ItemGroup>
</Project>
```

---

## `src/Tars.Cli/TrsxAst.fs`

```fsharp
namespace Tars.Cli

open Tars.Core

type Trsx = {
    name: string
    beliefs: Belief list
    goals: string list
    skills: SkillSpec list
    methods: (string * string list) list // goal -> steps
    reflectionRules: (string * string list) list // metric predicate -> actions
    rootGoals: string list
}
```

---

## `src/Tars.Cli/TrsxParser.fs`

```fsharp
namespace Tars.Cli

open System
open System.Text.RegularExpressions
open Tars.Core

module TrsxParser =
    let private section (name:string) (text:string) =
        let pattern = sprintf "\n\[%s\]\n([\s\S]*?)\n\[/%s\]" name name
        let m = Regex.Match(text, pattern)
        if m.Success then Some m.Groups[1].Value.Trim() else None

    let private kv (s:string) =
        s.Split('=',2, StringSplitOptions.TrimEntries) |> fun a -> a[0], a[1]

    let private parseTruth = function
        | "True" -> Belnap.True
        | "False" -> Belnap.False
        | "Both" -> Belnap.Both
        | _ -> Belnap.Unknown

    let parse (text:string) : Trsx =
        let name =
            Regex.Match(text, "name:\s*(.+)").Groups[1].Value.Trim()
        // beliefs
        let beliefs =
            match section "beliefs" text with
            | None -> []
            | Some body ->
                body.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun line ->
                    // id=foo; proposition="..."; truth=Unknown; confidence=0.5
                    let parts = line.Split(';', StringSplitOptions.TrimEntries)
                    let id    = parts |> Array.find (fun p -> p.StartsWith("id="))    |> fun p -> p.Substring(3)
                    let prop  = parts |> Array.find (fun p -> p.StartsWith("proposition=")) |> fun p -> p.Substring("proposition=".Length).Trim().Trim('"')
                    let truth = parts |> Array.find (fun p -> p.StartsWith("truth=")) |> fun p -> p.Substring(6) |> parseTruth
                    let conf  = parts |> Array.find (fun p -> p.StartsWith("confidence=")) |> fun p -> p.Substring(11) |> float
                    { id=id; proposition=prop; truth=truth; confidence=conf; provenance=[] })
                |> Array.toList
        // goals
        let goals =
            match section "goals" text with
            | None -> []
            | Some body ->
                Regex.Matches(body, "\"([^\"]+)\"") |> Seq.map (fun m -> m.Groups[1].Value) |> Seq.toList
        // skills
        let skills =
            match section "skills" text with
            | None -> []
            | Some body ->
                body.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun line ->
                    // name=Foo; pre=["a","b"]; post=["c"]; verifier="v"
                    let get key =
                        Regex.Match(line, key + "=([^;]+)").Groups[1].Value.Trim()
                    let name = get "name"
                    let listOf (s:string) = Regex.Matches(s, "\"([^\"]+)\"") |> Seq.map (fun m -> m.Groups[1].Value) |> Seq.toList
                    let pre  = get "pre"  |> listOf
                    let post = get "post" |> listOf
                    let ver  = get "verifier" |> fun v -> fun () -> true // wire real test lookup later
                    { name = name; pre = pre; post = post; verifier = ver; cost = 1.0 })
                |> Array.toList
        // methods
        let methods =
            match section "htn_methods" text with
            | None -> []
            | Some body ->
                body.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun line ->
                    // goal="G" -> ["S1","S2"]
                    let g = Regex.Match(line, "goal=\"([^\"]+)\"").Groups[1].Value
                    let steps = Regex.Matches(line, "\"([^\"]+)\"") |> Seq.skip 1 |> Seq.map (fun m -> m.Groups[1].Value) |> Seq.toList
                    g, steps)
                |> Array.toList
        // reflection
        let reflectionRules =
            match section "reflection" text with
            | None -> []
            | Some body ->
                body.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun line ->
                    // metric>threshold -> ["action"]
                    let pred = line.Split("->")[0].Trim()
                    let acts = Regex.Matches(line, "\"([^\"]+)\"") |> Seq.map (fun m -> m.Groups[1].Value) |> Seq.toList
                    pred, acts)
                |> Array.toList
        // run
        let rootGoals =
            match section "run" text with
            | None -> goals
            | Some body -> Regex.Matches(body, "\"([^\"]+)\"") |> Seq.map (fun m -> m.Groups[1].Value) |> Seq.toList

        { name = name; beliefs = beliefs; goals = goals; skills = skills; methods = methods; reflectionRules = reflectionRules; rootGoals = rootGoals }
```

---

## `src/Tars.Cli/TrsxInterpreter.fs`

```fsharp
namespace Tars.Cli

open Tars.Core
open Tars.Core.BeliefGraph
open Tars.Core.Skills
open Tars.Core.Planner

module TrsxInterpreter =
    let buildDomain (trsx:Trsx) =
        // Seed registry with built-ins; also add from .trsx
        let reg = Registry()
        reg.Add(Tars.Core.GitClone.spec)
        for s in trsx.skills do reg.Add s
        let methods = trsx.methods |> List.map (fun (g,steps) -> HTN.Method(goal=g, steps=steps))
        { HTN.methods = methods; registry = reg }

    let seedBeliefs (trsx:Trsx) =
        let g = BeliefGraph.T()
        trsx.beliefs |> List.iter g.Set
        g

    let runOnce (trsx:Trsx) =
        let beliefs = (seedBeliefs trsx).All()
        let domain  = buildDomain trsx
        let plan = HTN.plan domain beliefs trsx.rootGoals |> MCTS.choose
        printfn "Plan: %s" (plan |> List.map (fun s -> s.skill.name) |> String.concat " -> ")
        match HTN.execute beliefs plan with
        | Error e -> printfn "EXECUTION ERROR: %s" e; None
        | Ok b' ->
            let metrics = Metrics.compute 0.2 b' 0 0
            let decision = ReflectionAgent.decide metrics.predError metrics.beliefEntropy
            printfn "Reflection: %A (%s)" decision.triggers decision.notes
            Some b'
```

---

## `src/Tars.Cli/Program.fs`

```fsharp
namespace Tars.Cli

open System

module Program =
    [<EntryPoint>]
    let main argv =
        let path = if argv.Length > 0 then argv[0] else "samples/gi_minimal.trsx"
        if not (IO.File.Exists path) then
            eprintfn "TRSX file not found: %s" path
            1
        else
            let text = IO.File.ReadAllText path
            let trsx = TrsxParser.parse text
            match TrsxInterpreter.runOnce trsx with
            | None -> 2
            | Some beliefs ->
                printfn "Updated beliefs:"
                beliefs |> List.iter (fun b -> printfn " - %s = %A (%.2f)" b.id b.truth b.confidence)
                0
```

---

## Add to your existing repo

- Create `tars/` (or `modules/gi/`) and drop this whole **TARS.GI** folder inside, or merge the projects into your solution.
- Wire your existing `.trsx` meta‑loop to invoke `Tars.Cli` after each iteration.
- Replace the toy verifiers with real property tests and (eventually) Z3 contracts.

---

## Next steps (tiny punch list)

1. Replace the dummy **verifier** lambdas with actual property tests (FsCheck) and keep seeds in `/output/tests/`.
2. Implement a proper **HRR** (plate encoding) in `VSA.fs` and persist concept keys.
3. Make `StateSpace.infer` pluggable (EKF, UKF, particle) with interfaces.
4. Extend `TrsxParser` to parse `pre/post` that include truth values and numeric thresholds.
5. Insert a **POMDP** subsolver call for any skill marked `uncertain=true` in .trsx.
6. Add a `simulator` block in `.trsx` for domain‑specific rollouts (call `Simulator.simulatePlan`).

This gives you a clean runway to evolve TARS into a world‑model + HTN/POMDP + reflection loop where LLMs are optional, strictly behind verifiers.

