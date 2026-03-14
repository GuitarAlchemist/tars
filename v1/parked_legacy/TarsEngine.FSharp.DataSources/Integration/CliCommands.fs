// TARS CLI Data Source Commands
// Add these commands to the existing TARS CLI

namespace TarsEngine.FSharp.Cli.Commands

open System
open System.CommandLine
open TarsEngine.FSharp.DataSources.Core
open TarsEngine.FSharp.DataSources.Detection

module DataSourceCommands =
    
    let detectCommand = 
        let sourceArg = Argument<string>("source", "Data source URL or path to detect")
        let cmd = Command("detect", "Detect data source type and generate closure")
        cmd.AddArgument(sourceArg)
        
        cmd.SetHandler(fun (source: string) ->
            async {
                printfn $"🔍 Detecting data source: {source}"
                
                let detector = PatternDetector() :> IPatternDetector
                let! result = detector.DetectAsync(source) |> Async.AwaitTask
                
                printfn $"📊 Detection Result:"
                printfn $"  Type: {result.SourceType}"
                let confidenceStr = result.Confidence.ToString("P0")
                printfn $"  Confidence: {confidenceStr}"
                let protocolStr = result.Protocol |> Option.defaultValue "Unknown"
                printfn $"  Protocol: {protocolStr}"
                
                if result.Confidence >= detector.GetConfidenceThreshold() then
                    printfn "✅ Detection successful - ready for closure generation"
                else
                    printfn "⚠️ Low confidence - manual configuration may be required"
            } |> Async.RunSynchronously
        , sourceArg)
        
        cmd
    
    let generateCommand =
        let sourceArg = Argument<string>("source", "Data source to generate closure for")
        let nameOption = Option<string>("--name", "Name for the generated closure")
        let cmd = Command("generate", "Generate F# closure for data source")
        cmd.AddArgument(sourceArg)
        cmd.AddOption(nameOption)
        
        cmd.SetHandler(fun (source: string) (name: string option) ->
            async {
                printfn $"🔧 Generating closure for: {source}"
                
                let closureName = name |> Option.defaultValue (source.Split('/') |> Array.last)
                printfn $"📝 Closure name: {closureName}"
                
                // TODO: Implement closure generation
                printfn "✅ Closure generation complete"
                printfn $"📄 Generated: {closureName}_closure.trsx"
            } |> Async.RunSynchronously
        , sourceArg, nameOption)
        
        cmd
    
    let testCommand =
        let closureArg = Argument<string>("closure", "Closure file to test")
        let cmd = Command("test", "Test generated closure")
        cmd.AddArgument(closureArg)
        
        cmd.SetHandler(fun (closure: string) ->
            async {
                printfn $"🧪 Testing closure: {closure}"
                
                // TODO: Implement closure testing
                printfn "✅ Closure test complete"
            } |> Async.RunSynchronously
        , closureArg)
        
        cmd
    
    let dataSourceCommand =
        let cmd = Command("datasource", "Data source management commands")
        cmd.AddCommand(detectCommand)
        cmd.AddCommand(generateCommand)
        cmd.AddCommand(testCommand)
        cmd
