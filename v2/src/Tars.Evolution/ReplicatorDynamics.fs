namespace Tars.Evolution

/// Replicator dynamics for grammar rule ecosystems.
/// Models grammar rules as "species" competing for selection.
/// Rules that succeed in execution grow in population; failures shrink.
///
/// Implements the replicator equation: dx_i/dt = x_i * (f_i(x) - f_avg(x))
/// where x_i is the population proportion and f_i is the fitness of rule i.
///
/// This is Step 4 from the probabilistic grammar brainstorm:
/// "Each grammar rule is a species with a population proportion."
module ReplicatorDynamics =

    open System

    // =========================================================================
    // Types
    // =========================================================================

    /// A species in the grammar ecosystem
    type GrammarSpecies = {
        /// Unique identifier
        PatternId: string
        /// Human-readable name
        PatternName: string
        /// Promotion level
        Level: PromotionLevel
        /// Current population proportion [0.0, 1.0]
        Proportion: float
        /// Fitness: success rate * (1 / avg_duration_penalty)
        Fitness: float
        /// Whether this species is an ESS (evolutionarily stable strategy)
        IsStable: bool
    }

    /// Configuration for replicator dynamics
    type ReplicatorConfig = {
        /// Time step for discrete replicator equation
        TimeStep: float
        /// Number of simulation steps
        Steps: int
        /// Minimum proportion before a species is pruned
        PruneThreshold: float
        /// Smoothing factor: prevents instant extinction (0.01 = 1% floor)
        SmoothingFloor: float
    }

    let defaultConfig = {
        TimeStep = 0.1
        Steps = 50
        PruneThreshold = 0.001
        SmoothingFloor = 0.01
    }

    /// Result of running replicator dynamics
    type ReplicatorResult = {
        /// Final species proportions
        Species: GrammarSpecies list
        /// Species that were pruned (proportion below threshold)
        Pruned: GrammarSpecies list
        /// Species identified as ESS
        Stable: GrammarSpecies list
        /// Total simulation steps run
        StepsRun: int
        /// Trajectory: proportion history for each species
        Trajectory: (string * float list) list
    }

    // =========================================================================
    // Fitness computation
    // =========================================================================

    /// Compute fitness for a grammar rule from execution outcomes.
    /// Fitness = successRate - durationPenalty
    let computeSpeciesFitness
        (outcomes: (bool * int64) list) // (success, durationMs)
        : float =
        if outcomes.IsEmpty then 0.5 // neutral fitness for unseen rules
        else
            let successRate =
                float (outcomes |> List.filter fst |> List.length) / float outcomes.Length
            let avgDuration =
                outcomes |> List.map (fun (_, d) -> float d) |> List.average
            // Penalize slow rules: 1s=0, 5s=0.1, 10s=0.2
            let durationPenalty = min 0.3 (avgDuration / 50000.0)
            max 0.0 (successRate - durationPenalty)

    /// Build species from weighted rules with outcome data
    let buildSpecies
        (rules: WeightedGrammar.WeightedRule list)
        (outcomesById: Map<string, (bool * int64) list>)
        : GrammarSpecies list =
        if rules.IsEmpty then []
        else
            let total = rules |> List.sumBy (fun r -> r.Weight)
            rules |> List.map (fun r ->
                let outcomes = outcomesById |> Map.tryFind r.PatternId |> Option.defaultValue []
                let fitness = computeSpeciesFitness outcomes
                { PatternId = r.PatternId
                  PatternName = r.PatternName
                  Level = r.Level
                  Proportion = if total > 0.0 then r.Weight / total else 1.0 / float rules.Length
                  Fitness = fitness
                  IsStable = false })

    // =========================================================================
    // Replicator equation
    // =========================================================================

    /// One step of discrete replicator dynamics:
    /// x_i(t+1) = x_i(t) + dt * x_i(t) * (f_i - f_avg)
    let step (dt: float) (floor: float) (species: GrammarSpecies list) : GrammarSpecies list =
        if species.IsEmpty then []
        else
            let avgFitness =
                species |> List.sumBy (fun s -> s.Proportion * s.Fitness)

            let updated =
                species |> List.map (fun s ->
                    let delta = dt * s.Proportion * (s.Fitness - avgFitness)
                    let newProp = s.Proportion + delta
                    { s with Proportion = newProp })

            // Renormalize to sum to 1.0
            let total = updated |> List.sumBy (fun s -> s.Proportion)
            if total < 1e-15 then updated
            else
                let renormalized = updated |> List.map (fun s -> { s with Proportion = s.Proportion / total })
                // The smoothing floor is a post-normalization invariant, but clamping up
                // to it adds mass. Reclaim that mass from the species with headroom above
                // the floor, proportionally to that headroom — clamping alone pushes the
                // sum past 1.0, and the next step would then compute average fitness
                // against something that is no longer a probability distribution.
                let n = float renormalized.Length

                if floor <= 0.0 then
                    renormalized
                elif floor * n >= 1.0 then
                    // Floor is infeasible for this many species; the uniform distribution
                    // is the closest point that still satisfies sum-to-one.
                    renormalized |> List.map (fun s -> { s with Proportion = 1.0 / n })
                else
                    let floored =
                        renormalized |> List.map (fun s -> { s with Proportion = max floor s.Proportion })

                    let excess = (floored |> List.sumBy (fun s -> s.Proportion)) - 1.0
                    let headroom = floored |> List.sumBy (fun s -> s.Proportion - floor)

                    if excess <= 1e-15 || headroom <= 1e-15 then
                        floored
                    else
                        // excess < headroom whenever floor * n < 1.0, so no species is
                        // pushed back below the floor by this subtraction.
                        floored
                        |> List.map (fun s ->
                            let share = (s.Proportion - floor) / headroom
                            { s with Proportion = s.Proportion - excess * share })

    // =========================================================================
    // ESS detection
    // =========================================================================

    /// Check if a species is an Evolutionarily Stable Strategy.
    /// ESS condition: f_i > f_j for all j != i when population is mostly i,
    /// or (f_i = f_j and stability condition holds).
    /// Simplified: species with highest fitness AND proportion > 1/n is ESS.
    let detectESS (species: GrammarSpecies list) : GrammarSpecies list =
        if species.IsEmpty then []
        else
            let maxFitness = species |> List.map (fun s -> s.Fitness) |> List.max
            let threshold = 1.0 / float species.Length
            species |> List.map (fun s ->
                { s with
                    IsStable = s.Fitness >= maxFitness - 0.01
                              && s.Proportion >= threshold * 0.5 })

    // =========================================================================
    // Full simulation
    // =========================================================================

    /// Run replicator dynamics simulation
    let simulate (config: ReplicatorConfig) (initial: GrammarSpecies list) : ReplicatorResult =
        if initial.IsEmpty then
            { Species = []; Pruned = []; Stable = []; StepsRun = 0; Trajectory = [] }
        else
            // Initialize trajectory tracking
            let trajectories =
                initial |> List.map (fun s -> s.PatternId, ResizeArray<float>([s.Proportion]))

            let mutable current = initial
            let mutable stepsRun = 0

            for _ in 1 .. config.Steps do
                current <- step config.TimeStep config.SmoothingFloor current
                stepsRun <- stepsRun + 1
                // Record trajectory
                for (id, history) in trajectories do
                    let prop =
                        current |> List.tryFind (fun s -> s.PatternId = id)
                        |> Option.map (fun s -> s.Proportion) |> Option.defaultValue 0.0
                    history.Add(prop)

            // Detect ESS
            current <- detectESS current

            // Prune near-extinct species
            let surviving, pruned =
                current |> List.partition (fun s -> s.Proportion >= config.PruneThreshold)

            { Species = surviving
              Pruned = pruned
              Stable = surviving |> List.filter (fun s -> s.IsStable)
              StepsRun = stepsRun
              Trajectory = trajectories |> List.map (fun (id, h) -> (id, h |> Seq.toList)) }

    /// Quick run with default config
    let evolveEcosystem
        (rules: WeightedGrammar.WeightedRule list)
        (outcomesById: Map<string, (bool * int64) list>)
        : ReplicatorResult =
        let species = buildSpecies rules outcomesById
        simulate defaultConfig species
