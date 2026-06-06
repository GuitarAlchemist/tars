namespace Tars.Core

open System

// ===================================
// START OF CONSTITUTION TYPES
// ===================================

/// What specific role/function the agent's neural net performs
type NeuralRole =
    | Generate of domain: AgentDomain
    | Explore of searchSpace: string
    | Summarize of contentType: string
    | Mutate of target: string
    | Review of aspect: string
    | Coordinate of agents: AgentId list
    | GeneralReasoning

/// A symbolic invariant that must hold true in a constitution
type ConstitutionInvariant =
    | ParseCompleteness
    | BackwardCompatibility
    | GrammarValidity
    | TestPassing
    | CoverageMaintained
    | CustomInvariant of name: string * predicate: string

/// A hard limit on resources
type ResourceLimit =
    | MaxIterations of int
    | MaxTokens of int
    | MaxTimeMinutes of int
    | MaxMemoryMB of int64
    | MaxCpuPercent of int
    | MaxDiskWritesMB of int
    | MaxCost of decimal

/// A permission granted to the agent
type Permission =
    | ReadKnowledgeGraph
    | ModifyKnowledgeGraph
    | ReadCode of pattern: string
    | ModifyCode of pattern: string
    | SpawnAgent of agentType: string
    | CallTool of toolName: string
    | AccessSecret of secretName: string
    | ExecuteShellCommand of pattern: string
    | All // Administrator/God mode

/// A prohibition forcing the agent to avoid certain actions
type Prohibition =
    | CannotModifyCore
    | CannotDeleteData
    | CannotAccessNetwork
    | CannotSpawnUnlimited
    | CannotExceedBudget
    | CannotViolateInvariant of ConstitutionInvariant
    | CannotUseTool of toolName: string
    | CannotAccessPath of path: string

/// A goal the agent must achieve
type AchievementGoal =
    | ReduceComplexity of percent: int
    | MaintainCoverage
    | CompleteWithin of time: TimeSpan
    | CustomGoal of description: string

/// Time constraint for execution
type TimeConstraint =
    | MustCompleteWithin of TimeSpan
    | MustStartAfter of DateTimeOffset

/// The symbolic contract defining the agent's constraints and obligations
type SymbolicContract =
    { MustPreserve: ConstitutionInvariant list
      MustAchieve: AchievementGoal list
      ResourceBounds: ResourceLimit list
      Dependencies: AgentId list
      ConflictsWith: AgentId list
      TimeConstraints: TimeConstraint list }

    static member Empty =
        { MustPreserve = []
          MustAchieve = []
          ResourceBounds = []
          Dependencies = []
          ConflictsWith = []
          TimeConstraints = [] }

/// The full constitution combining role and contract
type AgentConstitution =
    { AgentId: AgentId
      NeuralRole: NeuralRole
      SymbolicContract: SymbolicContract
      // Explicit lists for easier access/overrides
      Invariants: ConstitutionInvariant list
      Permissions: Permission list
      Prohibitions: Prohibition list
      // Additional direct resource bounds if needed outside contract
      HardResourceBounds: ResourceLimit list }

    /// Create a new constitution with default empty values
    static member Create(id: AgentId, role: NeuralRole) =
        { AgentId = id
          NeuralRole = role
          SymbolicContract = SymbolicContract.Empty
          Invariants = []
          Permissions = []
          Prohibitions = []
          HardResourceBounds = [] }

// ===================================
// RUNTIME TYPES FOR ENFORCEMENT
// ===================================

/// An action the agent attempts to perform that requires checking
type AgentAction =
    | ExecuteTool of toolName: string * args: string
    | ReadFile of path: string
    | WriteFile of path: string
    | SpawnChild of role: NeuralRole
    | NetworkRequest of url: string
    | ModifyLedger of operation: string
    | GenericAction of name: string * details: string

/// Represents a violation of the agent's constitution
type Violation =
    | ProhibitionViolated of rule: Prohibition * details: string
    | InvariantBroken of invariant: ConstitutionInvariant * details: string
    | ResourceQuotaExceeded of limit: ResourceLimit * current: obj * max: obj
    | PermissionDenied of action: AgentAction * reason: string
    | DependencyMissing of missingAgent: AgentId
    | TimeConstraintViolated of timeLimit: TimeConstraint * elapsed: TimeSpan
