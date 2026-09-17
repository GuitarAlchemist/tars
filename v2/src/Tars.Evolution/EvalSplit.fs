namespace Tars.Evolution

open System.Security.Cryptography
open System.Text

/// Train/eval split of the benchmark problems (#294).
///
/// Held-out problems are still benchmarked, but `SelfTrain` never exports their
/// solutions as SFT data. A pass-rate gain on them after fine-tuning is therefore
/// not memorisation of the training set, which is what the self-train A/B reports.
module EvalSplit =

    /// Salt chosen so the current banks hold out exactly one problem per difficulty
    /// tier plus one GA problem, keeping the only timed problem in training.
    /// Changing it reshuffles the split and invalidates earlier A/B comparisons.
    [<Literal>]
    let private Salt = "holdout-28:"

    /// True for roughly one problem in four, decided by a SHA-256 of the id alone,
    /// so adding or editing another problem never moves this one between splits.
    let isHeldOut (problem: BenchmarkProblem) : bool =
        use sha = SHA256.Create()
        let digest = sha.ComputeHash(Encoding.UTF8.GetBytes(Salt + problem.Id))
        digest.[0] % 4uy = 0uy

    /// Problems whose solutions may become training data.
    let training (problems: BenchmarkProblem list) : BenchmarkProblem list =
        problems |> List.filter (isHeldOut >> not)

    /// Problems reserved for evaluation.
    let heldOut (problems: BenchmarkProblem list) : BenchmarkProblem list = problems |> List.filter isHeldOut
