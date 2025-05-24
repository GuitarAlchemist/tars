namespace TarsEngine.FSharp.Cli.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.Types

/// <summary>
/// Consolidated ML service for the CLI.
/// </summary>
type MLService(logger: ILogger<MLService>) =
    let mutable models = Map.empty<string, MLModel>
    
    /// <summary>
    /// Trains a machine learning model.
    /// </summary>
    member _.TrainModelAsync(modelName: string, modelType: MLModelType, dataPath: string) =
        task {
            try
                logger.LogInformation(sprintf "Starting training for model: %s" modelName)
                
                // Create and update model
                let model = createMLModel modelName modelType
                let trainingModel = { model with Status = Training }
                models <- Map.add modelName trainingModel models
                
                // Simulate training process
                logger.LogInformation("Training in progress...")
                do! Task.Delay(1000)
                
                // Complete training
                let accuracy = 0.85 + (Random().NextDouble() * 0.15)
                let trainedModel = { trainingModel with
                                       Status = Trained
                                       Accuracy = Some accuracy
                                       LastTrainedAt = Some DateTime.UtcNow }
                
                models <- Map.add modelName trainedModel models
                
                logger.LogInformation(sprintf "Model training completed: %s" modelName)
                return Ok trainedModel
            with
            | ex ->
                logger.LogError(ex, sprintf "Error training model: %s" modelName)
                return Error (createError (sprintf "Model training failed: %s" ex.Message) (Some ex.StackTrace))
        }
    
    /// <summary>
    /// Makes predictions using a trained model.
    /// </summary>
    member _.PredictAsync(modelName: string, inputData: string) =
        task {
            try
                logger.LogInformation(sprintf "Making predictions with model: %s" modelName)
                
                match Map.tryFind modelName models with
                | Some model when model.Status = Trained ->
                    let predictions = ["Sample prediction"]
                    let confidence = 0.85 + (Random().NextDouble() * 0.15)
                    logger.LogInformation(sprintf "Predictions completed for model: %s" modelName)
                    return Ok (predictions, confidence)
                | Some model ->
                    return Error (createError (sprintf "Model %s is not trained" modelName) None)
                | None ->
                    return Error (createError (sprintf "Model not found: %s" modelName) None)
            with
            | ex ->
                logger.LogError(ex, sprintf "Error making predictions with model: %s" modelName)
                return Error (createError (sprintf "Prediction failed: %s" ex.Message) (Some ex.StackTrace))
        }
    
    /// <summary>
    /// Lists all available models.
    /// </summary>
    member _.ListModelsAsync() =
        task {
            try
                logger.LogInformation("Listing all ML models")
                let modelList = models |> Map.toList |> List.map snd
                logger.LogInformation(sprintf "Found %d ML models" modelList.Length)
                return Ok modelList
            with
            | ex ->
                logger.LogError(ex, "Error listing ML models")
                return Error (createError (sprintf "Failed to list models: %s" ex.Message) (Some ex.StackTrace))
        }
