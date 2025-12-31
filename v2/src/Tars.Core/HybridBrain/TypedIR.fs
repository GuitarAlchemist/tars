namespace Tars.Core.HybridBrain

// =============================================================================
// PHASE 17.1: TYPED INTERMEDIATE REPRESENTATION WITH PHANTOM STATES
// =============================================================================
//
// This module implements the core IR for the "Cognition Compiler" architecture.
// Key insight: LLM outputs become compilation artifacts that must pass through
// parsing → typing → validation before execution is allowed.
//
// Reference: docs/1_Vision/hybrid_brain_architecture.md

open System

// =============================================================================
// PHANTOM STATE TYPES
// =============================================================================
// These empty interfaces serve as phantom types to enforce state transitions
// at compile time. A Plan<Draft> cannot be executed - only Plan<Executable> can.

/// Represents a plan in draft state (just created from LLM output)
type Draft = interface end

/// Represents a plan that has been parsed and has valid syntax
type Parsed = interface end

/// Represents a plan that has passed all validation checks
type Validated = interface end

/// Represents a plan ready for execution
type Executable = interface end

// =============================================================================
// CORE IR TYPES
// =============================================================================

/// Confidence level for sources and beliefs
type ConfidenceLevel =
    | High // 0.8-1.0 - Verified from authoritative source
    | Medium // 0.5-0.8 - From reliable but unverified source
    | Low // 0.2-0.5 - LLM generated, needs verification
    | Unknown // 0.0-0.2 - No confidence information

/// Source of a piece of information
type Source =
    | KnowledgeGraph of id: string
    | WebSource of url: string
    | LlmGenerated of model: string * confidence: float
    | UserProvided
    | DatabaseQuery of query: string
    | RdfTriple of subject: string * predicate: string * obj: string

/// Evidence for a claim
type Evidence =
    { Id: Guid
      Source: Source
      Claim: string
      Confidence: float
      RetrievedAt: DateTimeOffset
      ExpiresAt: DateTimeOffset option }

/// A structured belief in the system
type Belief =
    { Id: Guid
      Subject: string
      Predicate: string
      Object: string
      Confidence: float
      Source: Source
      Timestamp: DateTimeOffset }

/// A structured goal definition
type Goal =
    { Id: Guid
      Description: string
      SuccessCriteria: string list
      Priority: int
      Deadline: DateTimeOffset option
      // Linked beliefs that motivate this goal?
      Motivations: Belief list }

/// Available tools for execution
type Tool =
    | WebSearch of query: string
    | FileRead of path: string
    | FileWrite of path: string * content: string
    | LlmCall of prompt: string * model: string option
    | DatabaseQuery of query: string
    | KnowledgeGraphQuery of subject: string option * predicate: string option * obj: string option
    | SandboxExec of command: string * timeout: TimeSpan
    | HttpRequest of method: string * url: string * body: string option
    | Custom of name: string * args: Map<string, obj>

/// Condition that must be true
type Condition =
    { Description: string
      Predicate: unit -> bool }

/// A primitive action in the plan
type Action =
    | UseTool of Tool
    | AssertBelief of subject: string * predicate: string * obj: string * confidence: float
    | RetractBelief of subject: string * predicate: string * obj: string
    | QueryKnowledge of query: string
    | RequestEvidence of claim: string
    | Summarize of content: string * maxLength: int
    | Branch of condition: string * thenSteps: int list * elseSteps: int list
    | Loop of condition: string * bodySteps: int list * maxIterations: int
    | Parallel of stepGroups: int list list
    | Sequence of steps: int list
    // Refactoring Actions (Phase 17 Canonical Task)
    | ExtractFunction of name: string * lines: int * int
    | InlineValue of name: string * line: int
    | SimplifyPattern of line: int
    | RemoveDeadCode of lineStart: int * lineEnd: int
    | AddDocumentation of target: string * line: int
    | RenameSymbol of oldName: string * newName: string
    | NoOp

/// A step in the plan
type Step =
    { Id: int
      Name: string
      Description: string
      Action: Action
      Preconditions: Condition list
      Postconditions: Condition list
      EvidenceRequired: bool
      Timeout: TimeSpan option
      RetryCount: int }

/// Budget constraints for execution
type Budget =
    { MaxTokens: int
      MaxTime: TimeSpan
      MaxCost: decimal
      MaxMemory: int64
      MaxApiCalls: int }

    static member default' =
        { MaxTokens = 10000
          MaxTime = TimeSpan.FromMinutes(5.0)
          MaxCost = 1.0m
          MaxMemory = 1024L * 1024L * 1024L
          MaxApiCalls = 100 }

/// Policy defining what is allowed
type Policy =
    { AllowedTools: Tool list option // None = all allowed
      ForbiddenTools: Tool list
      AllowedDomains: string list option // For web access
      ForbiddenDomains: string list
      RequireEvidence: bool
      MaxConfidenceWithoutEvidence: float
      SandboxOnly: bool }

/// The Plan type with phantom type parameter for state
type Plan<'State> =
    { Id: Guid
      Goal: Goal
      Description: string
      Steps: Step list
      Assumptions: string list
      Unknowns: string list
      RequiresSources: string list
      Budget: Budget
      Policy: Policy
      CreatedAt: DateTimeOffset
      CreatedBy: string
      Version: int
      ParentVersion: Guid option }

// =============================================================================
// VALIDATION ERRORS
// =============================================================================

/// Types of parse errors
type ParseError =
    | SyntaxError of line: int * column: int * message: string
    | MissingField of fieldName: string
    | InvalidValue of fieldName: string * value: string * expectedType: string
    | UnknownAction of actionName: string
    | UnknownTool of toolName: string

/// Types of validation errors
type ValidationError =
    | BudgetExceeded of resource: string * limit: float * actual: float
    | ForbiddenTool of toolName: string
    | ForbiddenDomain of domain: string
    | MissingEvidence of claim: string
    | PreconditionViolated of stepId: int * condition: string
    | PostconditionViolated of stepId: int * condition: string
    | CircularDependency of stepIds: int list
    | InvariantViolated of invariantId: string * message: string
    | PolicyViolation of policyRule: string * violation: string
    | TimeoutExceeded of stepId: int * timeout: TimeSpan
    | UnreachableStep of stepId: int
    // Refactoring Validation Errors
    | OverlappingRefactor of step1: int * step2: int
    | InvalidLineRange of stepId: int * start: int * stop: int
    | SymbolNotFound of symbol: string * stepId: int

/// Context for validation (environment state)
type ValidationContext =
    { KnownSymbols: Set<string>
      KnownFiles: Set<string> }

    static member Empty =
        { KnownSymbols = Set.empty
          KnownFiles = Set.empty }

    static member Create(symbols: seq<string>, files: seq<string>) =
        { KnownSymbols = Set.ofSeq symbols
          KnownFiles = Set.ofSeq files }

/// Parse result
type ParseResult<'T> = Result<'T, ParseError list>

/// Validation result
type ValidationResult<'T> = Result<'T, ValidationError list>

// =============================================================================
// FORMAL CRITIQUE (Structured Feedback for LLM)
// =============================================================================

/// Suggested fix for an error
type SuggestedFix =
    { OriginalText: string
      ReplacementText: string
      Rationale: string }

/// Formal critique returned to LLM on failure
type FormalCritique =
    { ParseErrors: ParseError list
      ValidationErrors: ValidationError list
      Suggestions: SuggestedFix list
      MinimalCounterExample: string option }

// =============================================================================
// STATE TRANSITIONS
// =============================================================================

module StateTransitions =

    /// Create a draft plan from raw input
    let createDraft (goal: string) (description: string) : Plan<Draft> =
        { Id = Guid.NewGuid()
          Goal =
            { Id = Guid.NewGuid()
              Description = goal
              SuccessCriteria = []
              Priority = 1
              Deadline = None
              Motivations = [] }
          Description = description
          Steps = []
          Assumptions = []
          Unknowns = []
          RequiresSources = []
          Budget =
            { MaxTokens = 10000
              MaxTime = TimeSpan.FromMinutes(5.0)
              MaxCost = 0.10m
              MaxMemory = 100L * 1024L * 1024L // 100MB
              MaxApiCalls = 10 }
          Policy =
            { AllowedTools = None
              ForbiddenTools = []
              AllowedDomains = None
              ForbiddenDomains = []
              RequireEvidence = false
              MaxConfidenceWithoutEvidence = 0.5
              SandboxOnly = false }
          CreatedAt = DateTimeOffset.UtcNow
          CreatedBy = "LLM"
          Version = 1
          ParentVersion = None }

    /// Parse a draft plan (validates syntax)
    let parse (plan: Plan<Draft>) : ParseResult<Plan<Parsed>> =
        let errors = ResizeArray<ParseError>()

        // Check required fields
        if String.IsNullOrWhiteSpace(plan.Goal.Description) then
            errors.Add(MissingField "Goal")

        // Validate steps
        for step in plan.Steps do
            if String.IsNullOrWhiteSpace(step.Name) then
                errors.Add(MissingField $"Step[{step.Id}].Name")

        if errors.Count > 0 then
            Error(errors |> Seq.toList)
        else
            // Cast to Parsed state (safe because we validated)
            Ok
                { Id = plan.Id
                  Goal = plan.Goal
                  Description = plan.Description
                  Steps = plan.Steps
                  Assumptions = plan.Assumptions
                  Unknowns = plan.Unknowns
                  RequiresSources = plan.RequiresSources
                  Budget = plan.Budget
                  Policy = plan.Policy
                  CreatedAt = plan.CreatedAt
                  CreatedBy = plan.CreatedBy
                  Version = plan.Version
                  ParentVersion = plan.ParentVersion }

    /// Validate a parsed plan (checks invariants)
    let validate (ctx: ValidationContext) (plan: Plan<Parsed>) : ValidationResult<Plan<Validated>> =
        let errors = ResizeArray<ValidationError>()

        // Check budget
        if plan.Steps.Length > plan.Budget.MaxApiCalls then
            errors.Add(BudgetExceeded("ApiCalls", float plan.Budget.MaxApiCalls, float plan.Steps.Length))

        // Check for forbidden tools
        for step in plan.Steps do
            match step.Action with
            | UseTool tool ->
                let isForbidden =
                    plan.Policy.ForbiddenTools
                    |> List.exists (fun ft ->
                        match ft, tool with
                        | SandboxExec _, SandboxExec _ when plan.Policy.SandboxOnly -> false
                        | SandboxExec _, SandboxExec _ -> false
                        | Custom(n1, _), Custom(n2, _) -> n1 = n2
                        | _ -> false)

                if isForbidden then
                    errors.Add(ForbiddenTool(sprintf "%A" tool))
            | _ -> ()

        // Check for circular dependencies (simplified - just check for duplicate IDs)
        let ids = plan.Steps |> List.map (_.Id)

        let duplicates =
            ids
            |> List.groupBy id
            |> List.filter (fun (_, g) -> g.Length > 1)
            |> List.map fst

        if not duplicates.IsEmpty then
            errors.Add(CircularDependency duplicates)

        // Check refactoring line ranges
        for step in plan.Steps do
            match step.Action with
            | ExtractFunction(_, s, e) ->
                if s <= 0 || e <= s then
                    errors.Add(InvalidLineRange(step.Id, s, e))
            | RemoveDeadCode(s, e) ->
                if s <= 0 || e <= s then
                    errors.Add(InvalidLineRange(step.Id, s, e))
            | _ -> ()

        // Check for overlapping refactors (simplified)
        let ranges =
            plan.Steps
            |> List.choose (fun s ->
                match s.Action with
                | ExtractFunction(_, start, stop) -> Some(s.Id, start, stop)
                | RemoveDeadCode(start, stop) -> Some(s.Id, start, stop)
                | _ -> None)

        for i in 0 .. ranges.Length - 1 do
            for j in i + 1 .. ranges.Length - 1 do
                let (id1, s1, e1) = ranges.[i]
                let (id2, s2, e2) = ranges.[j]

                if (s1 <= e2 && s2 <= e1) then
                    errors.Add(OverlappingRefactor(id1, id2))

        // Check for symbol existence if context is provided
        if not ctx.KnownSymbols.IsEmpty then
            for step in plan.Steps do
                match step.Action with
                | ExtractFunction(name, _, _) ->
                    // For extraction, we check if the NEW symbol already exists (shadowing check)
                    // In F#, shadowing is allowed, so this might just be a warning or allowed.
                    // For now, let's not block it unless it's a conflict.
                    ()
                | AddDocumentation(_, line) ->
                    if line <= 0 then
                        errors.Add(InvalidLineRange(step.Id, line, line))
                | _ -> ()

        if errors.Count > 0 then
            Error(errors |> Seq.toList)
        else
            Ok
                { Id = plan.Id
                  Goal = plan.Goal
                  Description = plan.Description
                  Steps = plan.Steps
                  Assumptions = plan.Assumptions
                  Unknowns = plan.Unknowns
                  RequiresSources = plan.RequiresSources
                  Budget = plan.Budget
                  Policy = plan.Policy
                  CreatedAt = plan.CreatedAt
                  CreatedBy = plan.CreatedBy
                  Version = plan.Version
                  ParentVersion = plan.ParentVersion }

    /// Prepare validated plan for execution
    let prepareExecution (plan: Plan<Validated>) : Plan<Executable> =
        // This transition is guaranteed safe because we validated
        { Id = plan.Id
          Goal = plan.Goal
          Description = plan.Description
          Steps = plan.Steps
          Assumptions = plan.Assumptions
          Unknowns = plan.Unknowns
          RequiresSources = plan.RequiresSources
          Budget = plan.Budget
          Policy = plan.Policy
          CreatedAt = plan.CreatedAt
          CreatedBy = plan.CreatedBy
          Version = plan.Version
          ParentVersion = plan.ParentVersion }

    /// Full pipeline: Draft -> Executable (or Critique)
    let fullPipeline (ctx: ValidationContext) (plan: Plan<Draft>) : Result<Plan<Executable>, FormalCritique> =
        match parse plan with
        | Error parseErrors ->
            Error
                { ParseErrors = parseErrors
                  ValidationErrors = []
                  Suggestions = []
                  MinimalCounterExample = None }
        | Ok parsed ->
            match validate ctx parsed with
            | Error validationErrors ->
                Error
                    { ParseErrors = []
                      ValidationErrors = validationErrors
                      Suggestions = []
                      MinimalCounterExample = None }
            | Ok validated -> Ok(prepareExecution validated)

// =============================================================================
// CRITIQUE FORMATTING
// =============================================================================

module CritiqueFormatter =

    /// Format a parse error for LLM feedback
    let formatParseError (error: ParseError) : string =
        match error with
        | SyntaxError(line, col, msg) -> $"Syntax Error at line {line}, column {col}: {msg}"
        | MissingField field -> $"Missing required field: '{field}'"
        | InvalidValue(field, value, expected) -> $"Invalid value for '{field}': got '{value}', expected {expected}"
        | UnknownAction action -> $"Unknown action type: '{action}'"
        | UnknownTool tool -> $"Unknown tool: '{tool}'"

    /// Format a validation error for LLM feedback
    let formatValidationError (error: ValidationError) : string =
        match error with
        | BudgetExceeded(resource, limit, actual) -> $"Budget exceeded for {resource}: limit={limit}, actual={actual}"
        | ForbiddenTool tool -> $"Tool '{tool}' is not allowed by policy"
        | ForbiddenDomain domain -> $"Domain '{domain}' is not allowed by policy"
        | MissingEvidence claim -> $"Evidence required for claim: '{claim}'"
        | PreconditionViolated(stepId, condition) -> $"Step {stepId} precondition violated: {condition}"
        | PostconditionViolated(stepId, condition) -> $"Step {stepId} postcondition violated: {condition}"
        | CircularDependency steps ->
            let stepList = String.Join(", ", steps |> List.map string)
            $"Circular dependency detected involving steps: {stepList}"
        | InvariantViolated(id, msg) -> $"Invariant {id} violated: {msg}"
        | PolicyViolation(rule, violation) -> $"Policy violation - Rule: {rule}, Violation: {violation}"
        | TimeoutExceeded(stepId, timeout) -> $"Step {stepId} timeout exceeded: {timeout}"
        | UnreachableStep stepId -> $"Step {stepId} is unreachable"
        | OverlappingRefactor(s1, s2) ->
            $"Refactoring steps {s1} and {s2} overlap in line ranges. This would cause safe execution failure."
        | InvalidLineRange(stepId, s, e) ->
            $"Step {stepId} has invalid line range: {s}-{e}. Range must be positive and start < end."
        | SymbolNotFound(sym, stepId) ->
            $"Step {stepId} refers to symbol '{sym}' which was not found in the analysis context."

    /// Format full critique for LLM
    let formatForLlm (critique: FormalCritique) : string =
        let sb = System.Text.StringBuilder()

        sb.AppendLine("═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.AppendLine("                     VALIDATION FAILURE REPORT                  ")
        |> ignore

        sb.AppendLine("═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.AppendLine() |> ignore

        if not critique.ParseErrors.IsEmpty then
            sb.AppendLine("PARSE ERRORS:") |> ignore

            for err in critique.ParseErrors do
                sb.AppendLine($"  • {formatParseError err}") |> ignore

            sb.AppendLine() |> ignore

        if not critique.ValidationErrors.IsEmpty then
            sb.AppendLine("VALIDATION ERRORS:") |> ignore

            for err in critique.ValidationErrors do
                sb.AppendLine($"  • {formatValidationError err}") |> ignore

            sb.AppendLine() |> ignore

        if not critique.Suggestions.IsEmpty then
            sb.AppendLine("SUGGESTED FIXES:") |> ignore

            for fix in critique.Suggestions do
                sb.AppendLine($"  • Replace: {fix.OriginalText}") |> ignore
                sb.AppendLine($"    With:    {fix.ReplacementText}") |> ignore
                sb.AppendLine($"    Reason:  {fix.Rationale}") |> ignore

            sb.AppendLine() |> ignore

        match critique.MinimalCounterExample with
        | Some example ->
            sb.AppendLine("MINIMAL COUNTER-EXAMPLE:") |> ignore
            sb.AppendLine($"  {example}") |> ignore
            sb.AppendLine() |> ignore
        | None -> ()

        sb.AppendLine("═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.ToString()
