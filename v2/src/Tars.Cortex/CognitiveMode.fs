namespace Tars.Cortex

/// <summary>
/// Represents the cognitive mode of the system.
/// </summary>
type CognitiveMode =
    /// <summary>High entropy, seeking new information.</summary>
    | Exploratory
    /// <summary>Low entropy, optimizing and executing.</summary>
    | Convergent
    /// <summary>System under stress or error state.</summary>
    | Critical

/// The one decision rule for the cognitive mode (#243). CognitiveAnalyzer (agent-registry
/// snapshot) and CognitiveStateManager (WoT state machine) feed it their own signals.
module CognitiveMode =

    /// Normalized entropy (0.0 - 1.0) above which the system is exploring.
    let exploratoryEntropy = 0.7

    /// Critical overrides everything, high entropy means exploring, and otherwise the
    /// caller's `settled` mode applies (a snapshot says Convergent; a state machine may
    /// keep its current mode).
    let classify (critical: bool) (entropy: float) (settled: CognitiveMode) : CognitiveMode =
        if critical then Critical
        elif entropy > exploratoryEntropy then Exploratory
        else settled
