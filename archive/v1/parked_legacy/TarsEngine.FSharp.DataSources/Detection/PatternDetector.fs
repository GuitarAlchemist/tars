namespace TarsEngine.FSharp.DataSources.Detection

open System
open System.Text.RegularExpressions
open System.Threading.Tasks
open TarsEngine.FSharp.DataSources.Core

/// Basic pattern detector implementation
type PatternDetector() =
    
    let patterns = [
        ("postgresql", @"^postgresql://.*", DatabaseType.PostgreSQL, 0.95)
        ("mysql", @"^mysql://.*", DatabaseType.MySQL, 0.95)
        ("mongodb", @"^mongodb://.*", DatabaseType.MongoDB, 0.90)
        ("http_api", @"^https?://.*/(api|v\d+)/", ApiType.REST, 0.85)
        ("json_api", @"^https?://.*\.json.*", ApiType.REST, 0.90)
        ("csv_file", @".*\.csv$", FileType.CSV, 0.90)
        ("json_file", @".*\.json$", FileType.JSON, 0.90)
        ("kafka", @"^kafka://.*", StreamType.Kafka, 0.90)
        ("redis", @"^redis://.*", CacheType.Redis, 0.90)
    ]
    
    interface IPatternDetector with
        member this.DetectAsync(source: string) =
            Task.Run(fun () ->
                let matchedPattern = 
                    patterns
                    |> List.tryFind (fun (_, pattern, _, _) -> 
                        Regex.IsMatch(source, pattern, RegexOptions.IgnoreCase))
                
                match matchedPattern with
                | Some (name, _, dbType, confidence) ->
                    {
                        SourceType = Database dbType
                        Confidence = confidence
                        Protocol = Some name
                        Schema = None
                        Metadata = Map.ofList [("pattern", name :> obj); ("source", source :> obj)]
                    }
                | None ->
                    {
                        SourceType = Unknown source
                        Confidence = 0.5
                        Protocol = None
                        Schema = None
                        Metadata = Map.ofList [("source", source :> obj)]
                    }
            )
        
        member this.GetSupportedPatterns() =
            patterns |> List.map (fun (name, pattern, _, _) -> $"{name}: {pattern}")
        
        member this.GetConfidenceThreshold() = 0.8
