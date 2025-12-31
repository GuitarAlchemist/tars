namespace Tars.Core

open System

/// <summary>
/// Represents the explicit execution context for a reasoning cycle.
/// This captures the intent, constraints, and audit trail for a unit of agency.
/// </summary>
[<NoComparison; NoEquality>]
type RunCycleContext =
    {
        /// <summary>Unique identifier for this execution cycle (correlation ID).</summary>
        TraceId: string
        
        /// <summary>When the cycle started.</summary>
        StartTime: DateTimeOffset
        
        /// <summary>The budget governor for this cycle.</summary>
        Budget: BudgetGovernor
        
        /// <summary>The high-level goal or intent driving this cycle.</summary>
        Goal: string
        
        /// <summary>Maximum number of agents that can be spawned in this cycle.</summary>
        SpawningBudget: int
    }
