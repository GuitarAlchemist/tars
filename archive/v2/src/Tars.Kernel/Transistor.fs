namespace Tars.Kernel

open System
open System.Threading.Tasks

/// <summary>
/// A gate that blocks execution until a condition is met.
/// </summary>
type Gate(condition: unit -> Task<bool>, checkInterval: TimeSpan) =

    new(condition: unit -> Task<bool>) = Gate(condition, TimeSpan.FromMilliseconds(100.0))

    /// <summary>
    /// Waits until the condition becomes true.
    /// </summary>
    member this.WaitForOpen() =
        task {
            let mutable isOpen = false

            while not isOpen do
                let! result = condition ()

                if result then
                    isOpen <- true
                else
                    do! Task.Delay(checkInterval)
        }

/// <summary>
/// A gate that opens when a set of dependencies are met.
/// </summary>
type DependencyGate(dependencies: (unit -> Task<bool>) list) =
    inherit
        Gate(fun () ->
            task {
                let! results = Task.WhenAll(dependencies |> List.map (fun f -> f ()))
                return results |> Array.forall id
            })
