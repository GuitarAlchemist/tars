namespace TarsEngine.FSharp.Core.ML.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.ML.Core

/// <summary>
/// Interface for machine learning services.
/// </summary>
type IMLService =
    /// <summary>
    /// Creates a new model.
    /// </summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="modelType">The type of the model.</param>
    /// <param name="features">The features of the model.</param>
    /// <param name="label">The label of the model, if any.</param>
    /// <param name="hyperparameters">The hyperparameters of the model.</param>
    /// <returns>The created model.</returns>
    abstract member CreateModel : name:string * modelType:ModelType * features:Feature list * ?label:Label * ?hyperparameters:Hyperparameters -> Task<Model>
    
    /// <summary>
    /// Gets a model by ID.
    /// </summary>
    /// <param name="modelId">The ID of the model.</param>
    /// <returns>The model, if found.</returns>
    abstract member GetModel : modelId:Guid -> Task<Model option>
    
    /// <summary>
    /// Gets all models.
    /// </summary>
    /// <returns>The list of all models.</returns>
    abstract member GetAllModels : unit -> Task<Model list>
    
    /// <summary>
    /// Trains a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to train.</param>
    /// <param name="config">The training configuration.</param>
    /// <returns>The training result.</returns>
    abstract member TrainModel : modelId:Guid * config:TrainingConfig -> Task<TrainingResult>
    
    /// <summary>
    /// Makes a prediction using a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to use.</param>
    /// <param name="request">The prediction request.</param>
    /// <returns>The prediction result.</returns>
    abstract member Predict : modelId:Guid * request:PredictionRequest -> Task<PredictionResult>
    
    /// <summary>
    /// Makes batch predictions using a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to use.</param>
    /// <param name="requests">The prediction requests.</param>
    /// <returns>The prediction results.</returns>
    abstract member PredictBatch : modelId:Guid * requests:PredictionRequest list -> Task<PredictionResult list>
    
    /// <summary>
    /// Evaluates a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to evaluate.</param>
    /// <param name="dataset">The dataset to use for evaluation.</param>
    /// <returns>The model metrics.</returns>
    abstract member EvaluateModel : modelId:Guid * dataset:Dataset -> Task<ModelMetrics>
    
    /// <summary>
    /// Gets the feature importances for a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to analyze.</param>
    /// <param name="method">The method to use for calculating feature importances.</param>
    /// <returns>The feature importance result.</returns>
    abstract member GetFeatureImportances : modelId:Guid * ?method:string -> Task<FeatureImportanceResult>
    
    /// <summary>
    /// Compares multiple models.
    /// </summary>
    /// <param name="modelIds">The IDs of the models to compare.</param>
    /// <param name="dataset">The dataset to use for comparison.</param>
    /// <param name="criteria">The criteria to use for determining the best model.</param>
    /// <returns>The model comparison result.</returns>
    abstract member CompareModels : modelIds:Guid list * dataset:Dataset * ?criteria:string -> Task<ModelComparisonResult>
    
    /// <summary>
    /// Exports a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to export.</param>
    /// <param name="format">The format to export the model in.</param>
    /// <param name="path">The path to export the model to.</param>
    /// <returns>The model export result.</returns>
    abstract member ExportModel : modelId:Guid * format:string * path:string -> Task<ModelExportResult>
    
    /// <summary>
    /// Imports a model.
    /// </summary>
    /// <param name="format">The format of the model to import.</param>
    /// <param name="path">The path to import the model from.</param>
    /// <param name="name">The name to give the imported model.</param>
    /// <returns>The model import result.</returns>
    abstract member ImportModel : format:string * path:string * ?name:string -> Task<ModelImportResult>
    
    /// <summary>
    /// Deletes a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to delete.</param>
    /// <returns>Whether the model was deleted.</returns>
    abstract member DeleteModel : modelId:Guid -> Task<bool>
    
    /// <summary>
    /// Creates a dataset.
    /// </summary>
    /// <param name="name">The name of the dataset.</param>
    /// <param name="features">The features of the dataset.</param>
    /// <param name="label">The label of the dataset, if any.</param>
    /// <param name="dataPoints">The data points of the dataset.</param>
    /// <returns>The created dataset.</returns>
    abstract member CreateDataset : name:string * features:Feature list * ?label:Label * ?dataPoints:DataPoint list -> Task<Dataset>
    
    /// <summary>
    /// Splits a dataset.
    /// </summary>
    /// <param name="dataset">The dataset to split.</param>
    /// <param name="trainRatio">The ratio of data to use for training.</param>
    /// <param name="testRatio">The ratio of data to use for testing.</param>
    /// <param name="validationRatio">The ratio of data to use for validation, if any.</param>
    /// <param name="stratified">Whether to use stratified sampling.</param>
    /// <returns>The dataset split.</returns>
    abstract member SplitDataset : dataset:Dataset * trainRatio:float * testRatio:float * ?validationRatio:float * ?stratified:bool -> Task<DatasetSplit>
    
    /// <summary>
    /// Loads a dataset from a file.
    /// </summary>
    /// <param name="path">The path to load the dataset from.</param>
    /// <param name="format">The format of the dataset.</param>
    /// <param name="name">The name to give the dataset.</param>
    /// <returns>The loaded dataset.</returns>
    abstract member LoadDataset : path:string * format:string * ?name:string -> Task<Dataset>
    
    /// <summary>
    /// Saves a dataset to a file.
    /// </summary>
    /// <param name="dataset">The dataset to save.</param>
    /// <param name="path">The path to save the dataset to.</param>
    /// <param name="format">The format to save the dataset in.</param>
    /// <returns>Whether the dataset was saved.</returns>
    abstract member SaveDataset : dataset:Dataset * path:string * format:string -> Task<bool>
