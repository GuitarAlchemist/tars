module TARS.Programming.Validation.MetascriptEvolution

open System
open System.IO
open System.Text.RegularExpressions

type MetascriptGeneration =
    { Id: string
      Generation: int
      Fitness: float
      Content: string }

type MetascriptEvolutionValidator() =

    member _.LoadMetascript(path: string) =
        if File.Exists path then File.ReadAllText path else String.Empty

    member _.ScoreContent(content: string) =
        let metrics = Regex.Matches(content, "METRIC\\s+[a-z0-9_]+", RegexOptions.IgnoreCase)
        let agents = Regex.Matches(content, "SPAWN\\s+[A-Za-z0-9]+", RegexOptions.IgnoreCase)
        float metrics.Count * 0.3 + float agents.Count * 0.4

    member this.Evolve(path: string) =
        let content = this.LoadMetascript path
        if String.IsNullOrWhiteSpace content then
            { Id = Path.GetFileNameWithoutExtension path
              Generation = 0
              Fitness = 0.0
              Content = content }
        else
            let fitness = this.ScoreContent content
            { Id = Path.GetFileNameWithoutExtension path
              Generation = 1
              Fitness = fitness
              Content = content }

    member this.ValidateEvolution() =
        let recursive = this.Evolve ".specify/meta/tier4/recursive-loop.trsx"
        let release = this.Evolve ".specify/meta/tier4/release-train.trsx"

        printfn "?? METASCRIPT ANALYSIS"
        printfn "  Recursive-loop fitness: %.2f" recursive.Fitness
        printfn "  Release-train fitness: %.2f" release.Fitness

        recursive.Fitness > 0.0 && release.Fitness > 0.0
