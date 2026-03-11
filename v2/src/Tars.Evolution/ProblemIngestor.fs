namespace Tars.Evolution

open System
open System.IO
open System.Text.Json

module ProblemIngestor =

    let private eulerProblems =
        [ { Id = ProblemId "PE-001"
            Source = ProjectEuler
            Title = "Multiples of 3 or 5"
            Description = "Find the sum of all the multiples of 3 or 5 below 1000."
            Difficulty = Beginner
            Tags = [ "math"; "arithmetic" ]
            ValidationCriteria = Some "233168"
            ReferenceSolution = None }
          { Id = ProblemId "PE-002"
            Source = ProjectEuler
            Title = "Even Fibonacci Numbers"
            Description =
              "By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms."
            Difficulty = Beginner
            Tags = [ "math"; "fibonacci"; "sequence" ]
            ValidationCriteria = Some "4613732"
            ReferenceSolution = None }
          { Id = ProblemId "PE-003"
            Source = ProjectEuler
            Title = "Largest Prime Factor"
            Description = "What is the largest prime factor of the number 600851475143?"
            Difficulty = Intermediate
            Tags = [ "math"; "primes" ]
            ValidationCriteria = Some "6857"
            ReferenceSolution = None } ]

    let private logicResultProblems =
        [ { Id = ProblemId "LOG-001"
            Source = LogicGrid
            Title = "River Crossing"
            Description =
              "A farmer wants to cross a river and take with him a wolf, a goat, and a cabbage. There is a boat that can fit himself plus either the wolf, the goat, or the cabbage. If the wolf and the goat are alone on one shore, the wolf will eat the goat. If the goat and the cabbage are alone on the shore, the goat will eat the cabbage. How can the farmer bring the wolf, the goat, and the cabbage across the river?"
            Difficulty = Beginner
            Tags = [ "logic"; "constraint-satisfaction" ]
            ValidationCriteria = None // Requires semantic validation
            ReferenceSolution = None } ]

    /// Get the hardcoded starter curriculum
    let getStarterPack () = eulerProblems @ logicResultProblems

    type private ProblemDto =
        { Id: string
          Source: string
          Title: string
          Description: string
          Difficulty: string
          Tags: string list
          ValidationCriteria: string option
          ReferenceSolution: string option }

    /// Load problems from a local directory of NDJSON files
    let loadFromDirectory (path: string) : Problem list =
        if Directory.Exists(path) then
            let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
            options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())

            Directory.GetFiles(path, "*.ndjson", SearchOption.AllDirectories)
            |> Array.collect (fun file ->
                try
                    let lines = File.ReadAllLines(file)
                    printfn $"[diag] Reading {file} ({lines.Length} lines)"

                    lines
                    |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
                    |> Array.choose (fun line ->
                        try
                            let dto = JsonSerializer.Deserialize<ProblemDto>(line, options)
                            let id = ProblemId dto.Id

                            let diff =
                                match dto.Difficulty.ToLower() with
                                | "beginner" -> Beginner
                                | "intermediate" -> Intermediate
                                | "advanced" -> Advanced
                                | "expert" -> Expert
                                | _ -> Unascertained

                            let source =
                                match dto.Source.ToLower() with
                                | "projecteuler" -> ProjectEuler
                                | "arc" -> ARC
                                | "logicgrid" -> LogicGrid
                                | s -> Custom s

                            let p: Problem =
                                { Id = id
                                  Source = source
                                  Title = dto.Title
                                  Description = dto.Description
                                  Difficulty = diff
                                  Tags = dto.Tags
                                  ValidationCriteria = dto.ValidationCriteria
                                  ReferenceSolution = dto.ReferenceSolution }

                            Some p
                        with ex ->
                            None)
                with ex ->
                    Array.empty)
            |> Array.toList
        else
            []

    /// Get all available problems (Starter + Local)
    let getAllProblems (localPath: string) =
        getStarterPack () @ loadFromDirectory localPath
