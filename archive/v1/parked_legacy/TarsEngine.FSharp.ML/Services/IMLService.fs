namespace TarsEngine.FSharp.ML.Services

open System
open System.Collections.Generic
open System.Threading.Tasks

/// <summary>
/// Type of a machine learning model.
/// </summary>
type ModelType =
    | Classification
    | Regression
    | Clustering
    | Recommendation
    | Ranking
    | AnomalyDetection
    | Custom of string

/// <summary>
/// Status of a machine learning model.
/// </summary>
type ModelStatus =
    | NotTrained
    | Training
    | Trained
    | Failed
    | Deprecated

/// <summary>
/// Feature of a machine learning model.
/// </summary>
type Feature = {
    /// <summary>
    /// The name of the feature.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The type of the feature.
    /// </summary>
    Type: Type
    
    /// <summary>
    /// The importance of the feature, if known.
    /// </summary>
    Importance: float option
    
    /// <summary>
    /// Whether the feature is categorical.
    /// </summary>
    IsCategorical: bool
    
    /// <summary>
    /// The possible values of the feature, if categorical.
    /// </summary>
    PossibleValues: string list option
    
    /// <summary>
    /// Additional metadata for the feature.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Label of a machine learning model.
/// </summary>
type Label = {
    /// <summary>
    /// The name of the label.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The type of the label.
    /// </summary>
    Type: Type
    
    /// <summary>
    /// Whether the label is categorical.
    /// </summary>
    IsCategorical: bool
    
    /// <summary>
    /// The possible values of the label, if categorical.
    /// </summary>
    PossibleValues: string list option
    
    /// <summary>
    /// Additional metadata for the label.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Hyperparameters for a machine learning model.
/// </summary>
type Hyperparameters = {
    /// <summary>
    /// The hyperparameters.
    /// </summary>
    Parameters: Map<string, obj>
    
    /// <summary>
    /// Additional metadata for the hyperparameters.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Metrics for a machine learning model.
/// </summary>
type ModelMetrics = {
    /// <summary>
    /// The accuracy of the model, if applicable.
    /// </summary>
    Accuracy: float option
    
    /// <summary>
    /// The precision of the model, if applicable.
    /// </summary>
    Precision: float option
    
    /// <summary>
    /// The recall of the model, if applicable.
    /// </summary>
    Recall: float option
    
    /// <summary>
    /// The F1 score of the model, if applicable.
    /// </summary>
    F1Score: float option
    
    /// <summary>
    /// The mean squared error of the model, if applicable.
    /// </summary>
    MeanSquaredError: float option
    
    /// <summary>
    /// The root mean squared error of the model, if applicable.
    /// </summary>
    RootMeanSquaredError: float option
    
    /// <summary>
    /// The mean absolute error of the model, if applicable.
    /// </summary>
    MeanAbsoluteError: float option
    
    /// <summary>
    /// The R-squared value of the model, if applicable.
    /// </summary>
    RSquared: float option
    
    /// <summary>
    /// The confusion matrix of the model, if applicable.
    /// </summary>
    ConfusionMatrix: float[,] option
    
    /// <summary>
    /// The area under the ROC curve of the model, if applicable.
    /// </summary>
    AUC: float option
    
    /// <summary>
    /// Custom metrics for the model.
    /// </summary>
    CustomMetrics: Map<string, float>
    
    /// <summary>
    /// Additional metadata for the metrics.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// A machine learning model.
/// </summary>
type Model = {
    /// <summary>
    /// The ID of the model.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the model.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The type of the model.
    /// </summary>
    Type: ModelType
    
    /// <summary>
    /// The status of the model.
    /// </summary>
    Status: ModelStatus
    
    /// <summary>
    /// The features of the model.
    /// </summary>
    Features: Feature list
    
    /// <summary>
    /// The label of the model, if any.
    /// </summary>
    Label: Label option
    
    /// <summary>
    /// The hyperparameters of the model.
    /// </summary>
    Hyperparameters: Hyperparameters
    
    /// <summary>
    /// The metrics of the model, if any.
    /// </summary>
    Metrics: ModelMetrics option
    
    /// <summary>
    /// The creation time of the model.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The last training time of the model, if any.
    /// </summary>
    LastTrainingTime: DateTime option
    
    /// <summary>
    /// The version of the model.
    /// </summary>
    Version: string
    
    /// <summary>
    /// Additional metadata for the model.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Configuration for training a machine learning model.
/// </summary>
type TrainingConfig = {
    /// <summary>
    /// The hyperparameters for the model.
    /// </summary>
    Hyperparameters: Hyperparameters
    
    /// <summary>
    /// The maximum number of iterations for training.
    /// </summary>
    MaxIterations: int option
    
    /// <summary>
    /// The maximum training time in seconds.
    /// </summary>
    MaxTrainingTimeSeconds: int option
    
    /// <summary>
    /// The early stopping criteria, if any.
    /// </summary>
    EarlyStopping: Map<string, obj> option
    
    /// <summary>
    /// Additional metadata for the training configuration.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Result of training a machine learning model.
/// </summary>
type TrainingResult = {
    /// <summary>
    /// The trained model.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The metrics of the trained model.
    /// </summary>
    Metrics: ModelMetrics
    
    /// <summary>
    /// The training time.
    /// </summary>
    TrainingTime: TimeSpan
    
    /// <summary>
    /// The number of iterations performed during training.
    /// </summary>
    Iterations: int
    
    /// <summary>
    /// Whether the training was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The error message if the training failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata for the training result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// A data point for a machine learning model.
/// </summary>
type DataPoint = {
    /// <summary>
    /// The features of the data point.
    /// </summary>
    Features: Map<string, obj>
    
    /// <summary>
    /// The label of the data point, if any.
    /// </summary>
    Label: obj option
    
    /// <summary>
    /// The weight of the data point, if any.
    /// </summary>
    Weight: float option
    
    /// <summary>
    /// Additional metadata for the data point.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// A dataset for a machine learning model.
/// </summary>
type Dataset = {
    /// <summary>
    /// The name of the dataset.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The features of the dataset.
    /// </summary>
    Features: Feature list
    
    /// <summary>
    /// The label of the dataset, if any.
    /// </summary>
    Label: Label option
    
    /// <summary>
    /// The data points of the dataset.
    /// </summary>
    DataPoints: DataPoint list
    
    /// <summary>
    /// Additional metadata for the dataset.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// A split of a dataset for training, testing, and validation.
/// </summary>
type DatasetSplit = {
    /// <summary>
    /// The training set.
    /// </summary>
    TrainingSet: Dataset
    
    /// <summary>
    /// The testing set.
    /// </summary>
    TestingSet: Dataset
    
    /// <summary>
    /// The validation set, if any.
    /// </summary>
    ValidationSet: Dataset option
    
    /// <summary>
    /// The split ratio (training, testing, validation).
    /// </summary>
    SplitRatio: float * float * float option
    
    /// <summary>
    /// Whether the split is stratified.
    /// </summary>
    IsStratified: bool
    
    /// <summary>
    /// Additional metadata for the dataset split.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// A request for a prediction from a machine learning model.
/// </summary>
type PredictionRequest = {
    /// <summary>
    /// The features for the prediction.
    /// </summary>
    Features: Map<string, obj>
    
    /// <summary>
    /// Additional metadata for the prediction request.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// A prediction from a machine learning model.
/// </summary>
type Prediction = {
    /// <summary>
    /// The predicted value.
    /// </summary>
    Value: obj
    
    /// <summary>
    /// The confidence of the prediction, if applicable.
    /// </summary>
    Confidence: float option
    
    /// <summary>
    /// The probability distribution of the prediction, if applicable.
    /// </summary>
    ProbabilityDistribution: Map<string, float> option
    
    /// <summary>
    /// Additional metadata for the prediction.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// The result of a prediction from a machine learning model.
/// </summary>
type PredictionResult = {
    /// <summary>
    /// The prediction.
    /// </summary>
    Prediction: Prediction
    
    /// <summary>
    /// The model that made the prediction.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The request that was used to make the prediction.
    /// </summary>
    Request: PredictionRequest
    
    /// <summary>
    /// The time it took to make the prediction.
    /// </summary>
    PredictionTime: TimeSpan
    
    /// <summary>
    /// Whether the prediction was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The error message if the prediction failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata for the prediction result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// The result of a feature importance analysis.
/// </summary>
type FeatureImportanceResult = {
    /// <summary>
    /// The model that was analyzed.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The feature importances.
    /// </summary>
    FeatureImportances: Map<string, float>
    
    /// <summary>
    /// The method used to calculate feature importances.
    /// </summary>
    Method: string
    
    /// <summary>
    /// Additional metadata for the feature importance result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// The result of a model comparison.
/// </summary>
type ModelComparisonResult = {
    /// <summary>
    /// The models that were compared.
    /// </summary>
    Models: Model list
    
    /// <summary>
    /// The metrics for each model.
    /// </summary>
    Metrics: Map<Guid, ModelMetrics>
    
    /// <summary>
    /// The best model, if any.
    /// </summary>
    BestModel: Model option
    
    /// <summary>
    /// The criteria used to determine the best model.
    /// </summary>
    BestModelCriteria: string option
    
    /// <summary>
    /// Additional metadata for the model comparison result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// The result of exporting a model.
/// </summary>
type ModelExportResult = {
    /// <summary>
    /// The model that was exported.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The format of the exported model.
    /// </summary>
    Format: string
    
    /// <summary>
    /// The path to the exported model.
    /// </summary>
    Path: string
    
    /// <summary>
    /// The size of the exported model in bytes.
    /// </summary>
    SizeInBytes: int64
    
    /// <summary>
    /// Whether the export was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The error message if the export failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata for the model export result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// The result of importing a model.
/// </summary>
type ModelImportResult = {
    /// <summary>
    /// The imported model.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The format of the imported model.
    /// </summary>
    Format: string
    
    /// <summary>
    /// The path to the imported model.
    /// </summary>
    Path: string
    
    /// <summary>
    /// Whether the import was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The error message if the import failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata for the model import result.
    /// </summary>
    Metadata: Map<string, obj>
}

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
    abstract member CreateModel : name: string * modelType: ModelType * features: Feature list * ?label: Label * ?hyperparameters: Hyperparameters -> Task<Model>
    
    /// <summary>
    /// Gets a model by ID.
    /// </summary>
    /// <param name="modelId">The ID of the model.</param>
    /// <returns>The model, if found.</returns>
    abstract member GetModel : modelId: Guid -> Task<Model option>
    
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
    abstract member TrainModel : modelId: Guid * config: TrainingConfig -> Task<TrainingResult>
    
    /// <summary>
    /// Makes a prediction using a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to use.</param>
    /// <param name="request">The prediction request.</param>
    /// <returns>The prediction result.</returns>
    abstract member Predict : modelId: Guid * request: PredictionRequest -> Task<PredictionResult>
    
    /// <summary>
    /// Makes batch predictions using a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to use.</param>
    /// <param name="requests">The prediction requests.</param>
    /// <returns>The prediction results.</returns>
    abstract member PredictBatch : modelId: Guid * requests: PredictionRequest list -> Task<PredictionResult list>
    
    /// <summary>
    /// Evaluates a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to evaluate.</param>
    /// <param name="dataset">The dataset to use for evaluation.</param>
    /// <returns>The model metrics.</returns>
    abstract member EvaluateModel : modelId: Guid * dataset: Dataset -> Task<ModelMetrics>
    
    /// <summary>
    /// Gets the feature importances for a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to analyze.</param>
    /// <param name="method">The method to use for calculating feature importances.</param>
    /// <returns>The feature importance result.</returns>
    abstract member GetFeatureImportances : modelId: Guid * ?method: string -> Task<FeatureImportanceResult>
    
    /// <summary>
    /// Compares multiple models.
    /// </summary>
    /// <param name="modelIds">The IDs of the models to compare.</param>
    /// <param name="dataset">The dataset to use for comparison.</param>
    /// <param name="criteria">The criteria to use for determining the best model.</param>
    /// <returns>The model comparison result.</returns>
    abstract member CompareModels : modelIds: Guid list * dataset: Dataset * ?criteria: string -> Task<ModelComparisonResult>
    
    /// <summary>
    /// Exports a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to export.</param>
    /// <param name="format">The format to export the model in.</param>
    /// <param name="path">The path to export the model to.</param>
    /// <returns>The model export result.</returns>
    abstract member ExportModel : modelId: Guid * format: string * path: string -> Task<ModelExportResult>
    
    /// <summary>
    /// Imports a model.
    /// </summary>
    /// <param name="format">The format of the model to import.</param>
    /// <param name="path">The path to import the model from.</param>
    /// <param name="name">The name to give the imported model.</param>
    /// <returns>The model import result.</returns>
    abstract member ImportModel : format: string * path: string * ?name: string -> Task<ModelImportResult>
    
    /// <summary>
    /// Deletes a model.
    /// </summary>
    /// <param name="modelId">The ID of the model to delete.</param>
    /// <returns>Whether the model was deleted.</returns>
    abstract member DeleteModel : modelId: Guid -> Task<bool>
    
    /// <summary>
    /// Creates a dataset.
    /// </summary>
    /// <param name="name">The name of the dataset.</param>
    /// <param name="features">The features of the dataset.</param>
    /// <param name="label">The label of the dataset, if any.</param>
    /// <param name="dataPoints">The data points of the dataset.</param>
    /// <returns>The created dataset.</returns>
    abstract member CreateDataset : name: string * features: Feature list * ?label: Label * ?dataPoints: DataPoint list -> Task<Dataset>
    
    /// <summary>
    /// Splits a dataset.
    /// </summary>
    /// <param name="dataset">The dataset to split.</param>
    /// <param name="trainRatio">The ratio of data to use for training.</param>
    /// <param name="testRatio">The ratio of data to use for testing.</param>
    /// <param name="validationRatio">The ratio of data to use for validation, if any.</param>
    /// <param name="stratified">Whether to use stratified sampling.</param>
    /// <returns>The dataset split.</returns>
    abstract member SplitDataset : dataset: Dataset * trainRatio: float * testRatio: float * ?validationRatio: float * ?stratified: bool -> Task<DatasetSplit>
    
    /// <summary>
    /// Loads a dataset from a file.
    /// </summary>
    /// <param name="path">The path to load the dataset from.</param>
    /// <param name="format">The format of the dataset.</param>
    /// <param name="name">The name to give the dataset.</param>
    /// <returns>The loaded dataset.</returns>
    abstract member LoadDataset : path: string * format: string * ?name: string -> Task<Dataset>
    
    /// <summary>
    /// Saves a dataset to a file.
    /// </summary>
    /// <param name="dataset">The dataset to save.</param>
    /// <param name="path">The path to save the dataset to.</param>
    /// <param name="format">The format to save the dataset in.</param>
    /// <returns>Whether the dataset was saved.</returns>
    abstract member SaveDataset : dataset: Dataset * path: string * format: string -> Task<bool>
