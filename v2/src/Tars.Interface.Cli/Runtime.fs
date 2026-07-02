namespace Tars.Interface.Cli

open Serilog
open Tars.Core
open Tars.Llm

/// Which LLM backend/model variant a command wants from the runtime.
/// The common path is `Default`; the rarer variants become data, not separate
/// factory methods.
type ModelChoice =
    | Default
    | Model of string
    | ClaudeCode of string option

/// The CLI composition root: every dependency a command needs, built once and
/// injected. Commands accept an `ITarsRuntime` instead of constructing the LLM
/// service, the tool registry, or the configuration themselves. The production
/// adapter wires the real services; tests supply a stub adapter — so the seam is
/// the place command logic is tested.
type ITarsRuntime =
    /// Loaded TARS configuration.
    abstract member Config: TarsConfig
    /// Tool registry with all TARS tools registered (assembly scanned once).
    abstract member Tools: IToolRegistry
    /// Skill registry with all TARS skills registered.
    abstract member Skills: ISkillRegistry
    /// Resolve an LLM service for the requested model choice.
    abstract member Llm: ModelChoice -> ILlmService

/// Builds runtime adapters.
module TarsRuntime =

    /// Production runtime: real LLM via `LlmFactory`, a tool registry scanned
    /// once from the Tars.Tools assembly, and configuration loaded from disk.
    let production (logger: ILogger) : ITarsRuntime =
        let config = ConfigurationLoader.load ()

        // Scan the tool assembly once, lazily, on first access — commands that
        // never touch tools (e.g. `ask`) pay nothing.
        let tools =
            lazy
                (let registry = Tars.Tools.ToolRegistry()
                 registry.RegisterAssembly(typeof<Tars.Tools.TarsToolAttribute>.Assembly)
                 registry :> IToolRegistry)

        let skills = lazy (Tars.Tools.SkillRegistry.GlobalSkillRegistry() :> ISkillRegistry)

        { new ITarsRuntime with
            member _.Config = config
            member _.Tools = tools.Value
            member _.Skills = skills.Value

            member _.Llm choice =
                match choice with
                | Default -> LlmFactory.create logger
                | Model model -> LlmFactory.createWithModel logger model
                | ClaudeCode model -> LlmFactory.createClaudeCode model }
