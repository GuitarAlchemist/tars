namespace Tars.Evolution

open System

/// Origin of the problem
type ProblemSource =
    | ProjectEuler
    | ARC
    | LogicGrid
    | Cryptarithmetic
    | Custom of string
    | ResearchGenerated

/// Difficulty level of the problem
type ProblemDifficulty =
    | Beginner // e.g. "Hello World", Basic Math
    | Intermediate // e.g. Sorting, Basic Algorithms
    | Advanced // e.g. Graph Algorithms, Optimization
    | Expert // e.g. Open Research, Complex System Design
    | Unascertained

/// Unique identifier for a problem
type ProblemId = ProblemId of string

/// A standardized problem for TARS to solve
type Problem =
    {
        Id: ProblemId
        Source: ProblemSource
        Title: string
        Description: string
        Difficulty: ProblemDifficulty
        Tags: string list
        /// Expected output or validation script/logic
        ValidationCriteria: string option
        /// Example Solution (if available, for training/fine-tuning)
        ReferenceSolution: string option
    }

/// Tracking the agent's progress through the curriculum
type CurriculumState =
    { CompletedProblems: Set<ProblemId>
      FailedProblems: Map<ProblemId, int> // count of failures
      CurrentDifficulty: ProblemDifficulty
      MasteryScore: float } // 0.0 to 1.0 representation of capability
