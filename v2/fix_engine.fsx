open System
open System.IO

let path = "src/Tars.Evolution/Engine.fs"
let content = File.ReadAllText(path)

// 1. Fix EvolutionContext
let oldCtx = "        { Registry: IAgentRegistry
          Llm: ILlmService
          VectorStore: IVectorStore
          SemanticMemory: ISemanticMemory option
          Epistemic: IEpistemicGovernor option
          PreLlm: PreLlmPipeline option
          Budget: BudgetGovernor option
          OutputGuard: IOutputGuard option
          KnowledgeBase: KnowledgeBase option
          KnowledgeGraph: TemporalKnowledgeGraph.TemporalGraph option
          MemoryBuffer: BufferAgent<MemoryItem> option // Added Capacitor
          EpisodeService: IEpisodeIngestionService option // Graphiti integration
          Ledger: KnowledgeLedger option
          Evaluator: IEvaluationStrategy option
          RunId: RunId option
          Logger: string -> unit
          Verbose: bool
          ShowSemanticMessage: Message -> bool -> unit }"

let newCtx = "        { Registry: IAgentRegistry
          Llm: ILlmService
          VectorStore: IVectorStore
          SemanticMemory: ISemanticMemory option
          Epistemic: IEpistemicGovernor option
          PreLlm: PreLlmPipeline option
          Budget: BudgetGovernor option
          OutputGuard: IOutputGuard option
          KnowledgeBase: KnowledgeBase option
          KnowledgeGraph: TemporalKnowledgeGraph.TemporalGraph option
          MemoryBuffer: BufferAgent<MemoryItem> option // Added Capacitor
          EpisodeService: IEpisodeIngestionService option // Graphiti integration
          Ledger: KnowledgeLedger option
          Evaluator: IEvaluationStrategy option
          RunId: Tars.Core.RunId option
          Logger: string -> unit
          Verbose: bool
          ShowSemanticMessage: Message -> bool -> unit
          Focus: string option
          ToolRegistry: Tars.Tools.IToolRegistry option // Added for hot reload of tools
          ResearchEnhanced: bool // Research-enhanced curriculum (Phase 18)
          CodebaseIndex: Tars.Cortex.CodebaseRAG.CodebaseIndex option
          ToleranceMetrics: Tars.Core.ToleranceEngineering.MetricsAggregator option }"

let content1 = content.Replace(oldCtx.Replace("\r\n", "\n"), newCtx.Replace("\r\n", "\n"))

// 2. Fix formatBelief
let oldFormat = "    let private formatBelief (belief: Belief) =
        let predicate =
            match belief.Predicate with
            | RelationType.Custom p -> p
            | _ -> belief.Predicate.ToString()

        $ירת"- [{belief.Confidence:F2}] {belief.Subject.Value} {predicate} {belief.Object.Value}""

let newFormat = "    let private formatBelief (belief: Tars.Core.Belief) =
        let predicate =
            match belief.Predicate with
            | Tars.Core.RelationType.Custom p -> p
            | _ -> belief.Predicate.ToString()

        sprintf \"- [%.2f] %s %s %s\" belief.Confidence belief.Subject.Value predicate belief.Object.Value"

let content2 = content1.Replace(oldFormat.Replace("\r\n", "\n"), newFormat.Replace("\r\n", "\n"))

File.WriteAllText(path, content2)
printfn "Done"
