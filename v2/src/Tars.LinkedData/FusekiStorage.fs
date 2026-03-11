namespace Tars.LinkedData

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Knowledge
open Tars.LinkedData

/// Ledger storage that replicates assertions to a Fuseki/SPARQL endpoint
type FusekiStorage(endpointUri: Uri, auth: string option) =

    // Helper to create URIs from entity names
    let makeUri (name: string) =
        let clean =
            name.Replace(" ", "_").Replace("\"", "").Replace("<", "").Replace(">", "")

        $"<http://tars.ai/resource/{clean}>"

    let relationToUri (rel: RelationType) =
        let ns = "http://tars.ai/ns#"

        match rel with
        | IsA -> ns + "isA"
        | PartOf -> ns + "partOf"
        | HasProperty -> ns + "hasProperty"
        | Supports -> ns + "supports"
        | Contradicts -> ns + "contradicts"
        | DerivedFrom -> ns + "derivedFrom"
        | Causes -> ns + "causes"
        | Prevents -> ns + "prevents"
        | Enables -> ns + "enables"
        | Precedes -> ns + "precedes"
        | Supersedes -> ns + "supersedes"
        | Mentions -> ns + "mentions"
        | Cites -> ns + "cites"
        | Implements -> ns + "implements"
        | Custom s -> ns + s.Replace(" ", "_")

    interface ILedgerStorage with
        member _.Append(entry: BeliefEventEntry) =
            task {
                try
                    match entry.Event with
                    | BeliefEvent.Assert belief ->
                        // 1. Direct Triple
                        let s = makeUri belief.Subject.Value
                        let o = makeUri belief.Object.Value
                        let p = $"<{relationToUri belief.Predicate}>"
                        let directTriple = $"{s} {p} {o} ."

                        // 2. Reified Belief Entity (Metadata)
                        let beliefUri = $"<http://tars.ai/resource/{belief.Id}>"

                        let meta =
                            [ $"{beliefUri} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://tars.ai/ns#Belief> ."
                              $"{beliefUri} <http://tars.ai/ns#subject> {s} ."
                              $"{beliefUri} <http://tars.ai/ns#predicate> {p} ."
                              $"{beliefUri} <http://tars.ai/ns#object> {o} ."
                              $"{beliefUri} <http://tars.ai/ns#confidence> \"{belief.Confidence}\"^^<http://www.w3.org/2001/XMLSchema#double> ." ]
                            |> String.concat " "

                        // Combine
                        let allTriples = directTriple + " " + meta


                        let! (updateResult: Result<unit, string>) =
                            SparqlUpdateClient.insertData endpointUri auth allTriples |> Async.StartAsTask

                        match updateResult with
                        | Result.Ok _ -> return Result.Ok(())
                        | Result.Error e ->
                            // Log but valid as "Ok" because this is secondary storage
                            Console.WriteLine($"[Fuseki] Update failed: {e}")
                            return Result.Ok(())

                    | BeliefEvent.Retract(beliefId, _, _) ->
                        // Delete logic is harder without knowing the S-P-O.
                        // But we know the Belief URI: <http://tars.ai/resource/beliefId>
                        // We can delete the reified object.
                        // We CANNOT delete the direct triple safely unless we query it first or store it.
                        // For now, just delete the reified object.

                        let beliefUri = $"<http://tars.ai/resource/{beliefId}>"
                        let deleteQuery = $"DELETE WHERE {{ {beliefUri} ?p ?o }}"

                        let! res = SparqlUpdateClient.update endpointUri auth deleteQuery |> Async.StartAsTask
                        return Result.Ok(())

                    | _ -> return Result.Ok(())
                with ex ->
                    Console.WriteLine($"[Fuseki] Exception: {ex.Message}")
                    return Result.Ok(()) // Swallowing error to safe-guard primary ledger
            }

        member _.GetEvents(since) = Task.FromResult([])
        member _.GetEventsByBelief(beliefId) = Task.FromResult([])
        member _.GetSnapshot() = Task.FromResult([])
