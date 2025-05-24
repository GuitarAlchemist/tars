namespace TarsEngine.FSharp.ML.Core

open System
open System.IO
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.ML

/// <summary>
/// Core machine learning framework for TARS intelligence.
/// </summary>
type MLFramework(logger: ILogger<MLFramework>, ?options: MLFrameworkOptions) =
    let options = defaultArg options MLFrameworkOptionsDefaults.defaultOptions
    let mlContext = new MLContext(seed = (defaultArg options.Seed 42))
    let modelBasePath = 
        match options.ModelBasePath with
        | Some path -> path
        | None -> Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models")
    
    let loadedModels = Dictionary<string, ITransformer>()
    let predictionEngines = Dictionary<string, PredictionEngine<obj, obj>>()
    let modelLastUpdated = Dictionary<string, DateTime>()
    let modelMetadata = Dictionary<string, MLModelMetadata>()
    
    do
        // Ensure model directory exists
        Directory.CreateDirectory(modelBasePath) |> ignore
    
    /// <summary>
    /// Loads a model from disk or creates a new one if it doesn't exist.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <param name="createModelPipeline">Function to create the model pipeline if needed.</param>
    /// <param name="trainingData">Optional training data to train/retrain the model.</param>
    /// <returns>True if the model was loaded or created successfully.</returns>
    member this.LoadOrCreateModelAsync<'TData, 'TPrediction>
        (
            modelName: string,
            createModelPipeline: MLContext -> IEstimator<ITransformer>,
            ?trainingData: seq<'TData>
        ) : Task<bool> =
        
        task {
            try
                let modelPath = Path.Combine(modelBasePath, $"{modelName}.zip")
                
                // Check if model exists and is up to date
                if File.Exists(modelPath) then
                    let fileInfo = FileInfo(modelPath)
                    
                    // If model is already loaded and hasn't changed, return
                    if loadedModels.ContainsKey(modelName) &&
                       modelLastUpdated.TryGetValue(modelName, &(let mutable lastUpdated = DateTime.MinValue; lastUpdated)) &&
                       lastUpdated >= fileInfo.LastWriteTimeUtc then
                        return true
                    
                    // Load model
                    logger.LogInformation("Loading model: {ModelName}", modelName)
                    let modelSchema = DataViewSchema()
                    let loadedModel = mlContext.Model.Load(modelPath, &modelSchema)
                    
                    // Store model
                    loadedModels.[modelName] <- loadedModel
                    modelLastUpdated.[modelName] <- fileInfo.LastWriteTimeUtc
                    
                    // Create prediction engine
                    let typedPredictionEngine = mlContext.Model.CreatePredictionEngine<'TData, 'TPrediction>(loadedModel)
                    predictionEngines.[modelName] <- typedPredictionEngine :?> PredictionEngine<obj, obj>
                    
                    // Load metadata if exists
                    do! this.LoadModelMetadataAsync(modelName)
                    
                    return true
                else
                    // If no training data provided, can't create model
                    match trainingData with
                    | None -> 
                        logger.LogWarning("No model exists and no training data provided for: {ModelName}", modelName)
                        return false
                    | Some data ->
                        if Seq.isEmpty data then
                            logger.LogWarning("No model exists and empty training data provided for: {ModelName}", modelName)
                            return false
                        else
                            // Create and train model
                            logger.LogInformation("Creating new model: {ModelName}", modelName)
                            
                            // Create data view
                            let dataView = mlContext.Data.LoadFromEnumerable(data)
                            
                            // Create pipeline
                            let pipeline = createModelPipeline mlContext
                            
                            // Train model
                            let model = pipeline.Fit(dataView)
                            
                            // Save model
                            mlContext.Model.Save(model, dataView.Schema, modelPath)
                            
                            // Store model
                            loadedModels.[modelName] <- model
                            modelLastUpdated.[modelName] <- DateTime.UtcNow
                            
                            // Create prediction engine
                            let predictionEngine = mlContext.Model.CreatePredictionEngine<'TData, 'TPrediction>(model)
                            predictionEngines.[modelName] <- predictionEngine :?> PredictionEngine<obj, obj>
                            
                            // Create and save metadata
                            let metadata = {
                                ModelName = modelName
                                CreatedAt = DateTime.UtcNow
                                LastUpdatedAt = DateTime.UtcNow
                                DataType = typeof<'TData>.Name
                                PredictionType = typeof<'TPrediction>.Name
                                TrainingExamples = Seq.length data
                                ModelPath = modelPath
                                Metrics = Dictionary<string, double>()
                                HyperParameters = Dictionary<string, string>()
                                Tags = List<string>()
                            }
                            
                            modelMetadata.[modelName] <- metadata
                            do! this.SaveModelMetadataAsync(modelName)
                            
                            return true
            with
            | ex ->
                logger.LogError(ex, "Error loading or creating model: {ModelName}", modelName)
                return false
        }
    
    /// <summary>
    /// Trains or retrains a model.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <param name="createModelPipeline">Function to create the model pipeline.</param>
    /// <param name="trainingData">The training data.</param>
    /// <param name="evaluateModel">Optional function to evaluate the model and return metrics.</param>
    /// <param name="hyperParameters">Optional hyperparameters for the model.</param>
    /// <returns>True if the model was trained successfully.</returns>
    member this.TrainModelAsync<'TData, 'TPrediction>
        (
            modelName: string,
            createModelPipeline: MLContext -> IEstimator<ITransformer>,
            trainingData: seq<'TData>,
            ?evaluateModel: ITransformer * IDataView -> Dictionary<string, double>,
            ?hyperParameters: Dictionary<string, string>
        ) : Task<bool> =
        
        task {
            try
                logger.LogInformation("Training model: {ModelName}", modelName)
                
                let modelPath = Path.Combine(modelBasePath, $"{modelName}.zip")
                
                // Create data view
                let dataView = mlContext.Data.LoadFromEnumerable(trainingData)
                
                // Split data for training and evaluation
                let dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction = 0.2)
                let trainData = dataSplit.TrainSet
                let testData = dataSplit.TestSet
                
                // Create pipeline
                let pipeline = createModelPipeline mlContext
                
                // Train model
                let model = pipeline.Fit(trainData)
                
                // Evaluate model if evaluator provided
                let metrics = Dictionary<string, double>()
                match evaluateModel with
                | Some evaluator ->
                    let evaluationMetrics = evaluator(model, testData)
                    for kvp in evaluationMetrics do
                        metrics.[kvp.Key] <- kvp.Value
                    
                    logger.LogInformation("Model evaluation metrics: {Metrics}",
                        String.Join(", ", metrics |> Seq.map (fun kvp -> $"{kvp.Key}={kvp.Value:F4}")))
                | None -> ()
                
                // Save model
                mlContext.Model.Save(model, dataView.Schema, modelPath)
                
                // Store model
                loadedModels.[modelName] <- model
                modelLastUpdated.[modelName] <- DateTime.UtcNow
                
                // Create prediction engine
                let predictionEngine = mlContext.Model.CreatePredictionEngine<'TData, 'TPrediction>(model)
                predictionEngines.[modelName] <- predictionEngine :?> PredictionEngine<obj, obj>
                
                // Update metadata
                let metadata =
                    if modelMetadata.ContainsKey(modelName) then
                        let existingMetadata = modelMetadata.[modelName]
                        { existingMetadata with
                            LastUpdatedAt = DateTime.UtcNow
                            TrainingExamples = Seq.length trainingData
                            Metrics = metrics
                            HyperParameters = defaultArg hyperParameters (Dictionary<string, string>())
                        }
                    else
                        {
                            ModelName = modelName
                            CreatedAt = DateTime.UtcNow
                            LastUpdatedAt = DateTime.UtcNow
                            DataType = typeof<'TData>.Name
                            PredictionType = typeof<'TPrediction>.Name
                            TrainingExamples = Seq.length trainingData
                            ModelPath = modelPath
                            Metrics = metrics
                            HyperParameters = defaultArg hyperParameters (Dictionary<string, string>())
                            Tags = List<string>()
                        }
                
                modelMetadata.[modelName] <- metadata
                do! this.SaveModelMetadataAsync(modelName)
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Error training model: {ModelName}", modelName)
                return false
        }
    
    /// <summary>
    /// Makes a prediction using a loaded model.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <param name="data">The input data.</param>
    /// <returns>The prediction.</returns>
    member this.Predict<'TData, 'TPrediction>
        (
            modelName: string,
            data: 'TData
        ) : 'TPrediction option =
        
        try
            let mutable engine = Unchecked.defaultof<PredictionEngine<obj, obj>>
            if not (predictionEngines.TryGetValue(modelName, &engine)) then
                logger.LogWarning("Prediction engine not found for model: {ModelName}", modelName)
                None
            else
                let typedEngine = engine :?> PredictionEngine<'TData, 'TPrediction>
                if isNull (box typedEngine) then
                    logger.LogWarning("Invalid prediction engine type for model: {ModelName}", modelName)
                    None
                else
                    Some (typedEngine.Predict(data))
        with
        | ex ->
            logger.LogError(ex, "Error making prediction with model: {ModelName}", modelName)
            None
    
    /// <summary>
    /// Gets all available models.
    /// </summary>
    /// <returns>List of model metadata.</returns>
    member this.GetAvailableModels() : List<MLModelMetadata> =
        List<MLModelMetadata>(modelMetadata.Values)
    
    /// <summary>
    /// Gets metadata for a specific model.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <returns>The model metadata, or None if not found.</returns>
    member this.GetModelMetadata(modelName: string) : MLModelMetadata option =
        let mutable metadata = Unchecked.defaultof<MLModelMetadata>
        if modelMetadata.TryGetValue(modelName, &metadata) then
            Some metadata
        else
            None
    
    /// <summary>
    /// Deletes a model.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <returns>True if the model was deleted successfully.</returns>
    member this.DeleteModelAsync(modelName: string) : Task<bool> =
        task {
            try
                let modelPath = Path.Combine(modelBasePath, $"{modelName}.zip")
                let metadataPath = Path.Combine(modelBasePath, $"{modelName}.metadata.json")
                
                // Remove from dictionaries
                loadedModels.Remove(modelName) |> ignore
                predictionEngines.Remove(modelName) |> ignore
                modelLastUpdated.Remove(modelName) |> ignore
                modelMetadata.Remove(modelName) |> ignore
                
                // Delete files
                if File.Exists(modelPath) then
                    File.Delete(modelPath)
                
                if File.Exists(metadataPath) then
                    File.Delete(metadataPath)
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Error deleting model: {ModelName}", modelName)
                return false
        }
    
    /// <summary>
    /// Loads model metadata from disk.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    member private this.LoadModelMetadataAsync(modelName: string) : Task =
        task {
            try
                let metadataPath = Path.Combine(modelBasePath, $"{modelName}.metadata.json")
                if not (File.Exists(metadataPath)) then
                    return ()
                
                let! json = File.ReadAllTextAsync(metadataPath)
                let metadata = System.Text.Json.JsonSerializer.Deserialize<MLModelMetadata>(json)
                
                if not (isNull metadata) then
                    modelMetadata.[modelName] <- metadata
            with
            | ex ->
                logger.LogError(ex, "Error loading model metadata: {ModelName}", modelName)
        }
    
    /// <summary>
    /// Saves model metadata to disk.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    member private this.SaveModelMetadataAsync(modelName: string) : Task =
        task {
            try
                let mutable metadata = Unchecked.defaultof<MLModelMetadata>
                if not (modelMetadata.TryGetValue(modelName, &metadata)) then
                    return ()
                
                let metadataPath = Path.Combine(modelBasePath, $"{modelName}.metadata.json")
                let options = System.Text.Json.JsonSerializerOptions()
                options.WriteIndented <- true
                let json = System.Text.Json.JsonSerializer.Serialize(metadata, options)
                
                do! File.WriteAllTextAsync(metadataPath, json)
            with
            | ex ->
                logger.LogError(ex, "Error saving model metadata: {ModelName}", modelName)
        }
    
    interface IDisposable with
        member this.Dispose() =
            // Dispose prediction engines
            for engine in predictionEngines.Values do
                if not (isNull engine) then
                    engine.Dispose()
            
            predictionEngines.Clear()
            loadedModels.Clear()
            modelLastUpdated.Clear()
            modelMetadata.Clear()
