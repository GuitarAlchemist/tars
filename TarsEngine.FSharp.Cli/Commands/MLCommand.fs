namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Core.Types

/// <summary>
/// Command for machine learning operations using consolidated services.
/// </summary>
type MLCommand(mlService: MLService) =
    interface ICommand with
        member _.Name = "ml"
        
        member _.Description = "Machine learning operations with real ML capabilities"
        
        member self.Usage = "tars ml [subcommand] [options]"
        
        member self.Examples = [
            "tars ml train --model classifier --data training.csv"
            "tars ml predict --model classifier --input test.csv"
            "tars ml list-models"
        ]
        
        member self.ValidateOptions(options) = true
        
        member self.ExecuteAsync(options) =
            Task.Run(fun () ->
                try
                    let subcommand = 
                        match options.Arguments with
                        | arg :: _ -> arg
                        | [] -> "help"
                    
                    match subcommand.ToLowerInvariant() with
                    | "train" ->
                        let model = options.Options.TryFind("model") |> Option.defaultValue "default"
                        let data = options.Options.TryFind("data") |> Option.defaultValue "training.csv"
                        Console.WriteLine(sprintf "Training ML model: %s with data: %s" model data)
                        
                        let modelType = Classification
                        let trainingResult = mlService.TrainModelAsync(model, modelType, data).Result
                        match trainingResult with
                        | Ok trainedModel ->
                            Console.WriteLine("Training completed successfully")
                            Console.WriteLine(sprintf "Model '%s' trained" trainedModel.Name)
                            CommandResult.success("ML model training completed")
                        | Error error ->
                            Console.WriteLine(sprintf "Training Error: %s" error.Message)
                            CommandResult.failure("ML model training failed")
                    
                    | "predict" ->
                        let model = options.Options.TryFind("model") |> Option.defaultValue "default"
                        let input = options.Options.TryFind("input") |> Option.defaultValue "input.csv"
                        Console.WriteLine(sprintf "Making predictions with model: %s" model)
                        
                        let predictionResult = mlService.PredictAsync(model, input).Result
                        match predictionResult with
                        | Ok (predictions, confidence) ->
                            Console.WriteLine("Predictions generated:")
                            predictions |> List.iteri (fun i pred ->
                                Console.WriteLine(sprintf "  Sample %d: %s (confidence: %.2f)" 
                                    (i + 1) pred confidence)
                            )
                            CommandResult.success("ML predictions completed")
                        | Error error ->
                            Console.WriteLine(sprintf "Prediction Error: %s" error.Message)
                            CommandResult.failure("ML predictions failed")
                    
                    | "list-models" | "list" ->
                        Console.WriteLine("Listing ML models...")
                        
                        let modelsResult = mlService.ListModelsAsync().Result
                        match modelsResult with
                        | Ok models ->
                            Console.WriteLine("Available ML Models:")
                            if models.IsEmpty then
                                Console.WriteLine("  No models found. Use 'tars ml train' to create models.")
                            else
                                models |> List.iter (fun model ->
                                    Console.WriteLine(sprintf "  %s - %A - %A" 
                                        model.Name model.ModelType model.Status)
                                )
                            CommandResult.success("ML models listed")
                        | Error error ->
                            Console.WriteLine(sprintf "Error listing models: %s" error.Message)
                            CommandResult.failure("Failed to list ML models")
                    
                    | "help" | _ ->
                        Console.WriteLine("ML Command Help")
                        Console.WriteLine("Available subcommands:")
                        Console.WriteLine("  train      - Train a machine learning model")
                        Console.WriteLine("  predict    - Make predictions with a trained model")
                        Console.WriteLine("  list       - List available models")
                        CommandResult.success("ML help displayed")
                        
                with
                | ex ->
                    CommandResult.failure(sprintf "ML command failed: %s" ex.Message)
            )
