namespace Tars.Cortex

open System
open Tars.Core

/// <summary>
/// Defines the high-level strategy for reasoning.
/// This is the "Substrate" for Neuro-Symbolic Self-Improvement.
/// Patterns are blueprints that are compiled into executable WoT (Workflows of Thought).
/// </summary>
module ReasoningPattern =

    /// <summary>
    /// The structural topology of the reasoning pattern.
    /// </summary>
    type PatternKind =
        /// <summary>
        /// A linear sequence of steps (e.g. Chain of Thought).
        /// Simple, fast, good for routine tasks.
        /// </summary>
        | Linear

        /// <summary>
        /// A branching tree structure (e.g. Tree of Thoughts).
        /// Good for exploration and backtracking.
        /// </summary>
        | TreeSearch

        /// <summary>
        /// A circular feedback loop (e.g. ReAct, Reflexion).
        /// Good for interactive tasks and self-correction.
        /// </summary>
        | Loop

        /// <summary>
        /// A directed acyclic graph (e.g. Graph of Thoughts).
        /// Good for aggregation and complex dependencies.
        /// </summary>
        | Graph

        /// <summary>
        /// Parallel execution with consensus or voting.
        /// Good for robustness and validation.
        /// </summary>
        | Parallel

    /// <summary>
    /// Semantic constraints applied to the pattern execution.
    /// </summary>
    type PatternConstraint =
        /// <summary>
        /// Maximum depth or number of steps allowed.
        /// </summary>
        | MaxSteps of int

        /// <summary>
        /// A logical invariant that must hold true at verification points.
        /// </summary>
        | Invariant of description: string

        /// <summary>
        /// Tools that are mandatory for this pattern.
        /// </summary>
        | RequiredTools of toolNames: string list

        /// <summary>
        /// A required format for the final output.
        /// </summary>
        | OutputFormat of format: string

    /// <summary>
    /// A single abstract step in the reasoning pattern blueprint.
    /// </summary>
    type PatternStep =
        {
            /// <summary>
            /// Unique identifier for this step within the pattern.
            /// </summary>
            Id: string

            /// <summary>
            /// The broad category of this step (Reason, Tool, Validate, Memory, Control).
            /// Maps to WoTNodeKind.
            /// </summary>
            Role: string

            /// <summary>
            /// The template for the prompt or instruction at this step.
            /// Mutations often target this field.
            /// </summary>
            InstructionTemplate: string option

            /// <summary>
            /// Step-specific constraints or configuration.
            /// </summary>
            Parameters: Map<string, string>

            /// <summary>
            /// IDs of steps that must complete before this one.
            /// </summary>
            Dependencies: string list
        }

    /// <summary>
    /// A declarative definition of a reasoning strategy.
    /// This is the top-level artifact that Tars evolves.
    /// </summary>
    type ReasoningPattern =
        {
            /// <summary>
            /// Unique identifier for this pattern version.
            /// </summary>
            Id: string

            /// <summary>
            /// Human-readable name (e.g. "Linear Chain of Thought v1").
            /// </summary>
            Name: string

            /// <summary>
            /// The structural topology.
            /// </summary>
            Kind: PatternKind

            /// <summary>
            /// The list of steps that define the workflow.
            /// </summary>
            Steps: PatternStep list

            /// <summary>
            /// Global constraints for this pattern.
            /// </summary>
            Constraints: PatternConstraint list

            /// <summary>
            /// Version number for evolution tracking.
            /// </summary>
            Version: int

            /// <summary>
            /// Description of what this pattern is good for (used for selection).
            /// </summary>
            Description: string

            /// <summary>
            /// Metadata for the mutation engine (e.g., parent pattern ID, generation).
            /// </summary>
            Metadata: Map<string, string>
        }

    /// <summary>
    /// Creates a default empty pattern.
    /// </summary>
    let empty =
        { Id = Guid.NewGuid().ToString()
          Name = "New Pattern"
          Kind = Linear
          Steps = []
          Constraints = []
          Version = 1
          Description = ""
          Metadata = Map.empty }

    /// <summary>
    /// Standard library of reasoning patterns.
    /// </summary>
    module Library =

        let private mkStep id role instr deps =
            { Id = id
              Role = role
              InstructionTemplate = Some instr
              Parameters = Map.empty
              Dependencies = deps }

        /// A standard Chain of Thought pattern
        let linearCoT =
            { empty with
                Name = "Linear Chain of Thought"
                Kind = Linear
                Description = "A linear sequence of analysis, decomposition, reasoning, and synthesis."
                Steps =
                    [ mkStep "decompose" "Reason" "Decompose the problem into key components: {goal}" []
                      mkStep "reason" "Reason" "Analyze each component step by step." [ "decompose" ]
                      mkStep "synthesize" "Reason" "Synthesize insights into a final solution." [ "reason" ]
                      mkStep "verify" "Validate" "Verify the solution meets the goal." [ "synthesize" ] ] }

        /// A Critic-Refinement pattern
        let criticRefinement =
            { empty with
                Name = "Critic Refinement"
                Kind = Loop
                Description = "Draft a solution, critique it, and refine."
                Steps =
                    [ mkStep "draft" "Reason" "Draft an initial solution for: {goal}" []
                      mkStep "critique" "Critique" "Critique the draft. Identify flaws and gaps." [ "draft" ]
                      mkStep "refine" "Reason" "Refine the solution based on the critique." [ "critique" ]
                      mkStep "final_check" "Validate" "Ensure the refined solution is robust." [ "refine" ] ] }

        /// Parallel Brainstorming
        let parallelBrainstorming =
            { empty with
                Name = "Parallel Brainstorming"
                Kind = Graph
                Description = "Generate multiple independent ideas and synthesize."
                Steps =
                    [ mkStep "decompose" "Reason" "Decompose the problem: {goal}" []
                      mkStep "idea_1" "Reason" "Generate solution approach 1." [ "decompose" ]
                      mkStep "idea_2" "Reason" "Generate solution approach 2." [ "decompose" ]
                      mkStep "idea_3" "Reason" "Generate solution approach 3." [ "decompose" ]
                      mkStep
                          "synthesize"
                          "Reason"
                          "Synthesize the best aspects of all approaches."
                          [ "idea_1"; "idea_2"; "idea_3" ] ] }
