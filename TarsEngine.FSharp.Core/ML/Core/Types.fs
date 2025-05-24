namespace TarsEngine.FSharp.Core.ML.Core

open System
open System.Collections.Generic

/// <summary>
/// Represents the type of a machine learning model.
/// </summary>
type ModelType =
    | Classification
    | Regression
    | Clustering
    | Reinforcement
    | NeuralNetwork
    | Transformer
    | Custom of string

/// <summary>
/// Represents the status of a machine learning model.
/// </summary>
type ModelStatus =
    | NotTrained
    | Training
    | Trained
    | Failed
    | Cancelled

/// <summary>
/// Represents a feature in a machine learning model.
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
    /// The importance of the feature.
    /// </summary>
    Importance: float option
    
    /// <summary>
    /// Whether the feature is categorical.
    /// </summary>
    IsCategorical: bool
    
    /// <summary>
    /// The possible values of the feature, if categorical.
    /// </summary>
    PossibleValues: obj list option
    
    /// <summary>
    /// Additional metadata about the feature.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a label in a machine learning model.
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
    PossibleValues: obj list option
    
    /// <summary>
    /// Additional metadata about the label.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a prediction from a machine learning model.
/// </summary>
type Prediction = {
    /// <summary>
    /// The predicted value.
    /// </summary>
    Value: obj
    
    /// <summary>
    /// The confidence of the prediction.
    /// </summary>
    Confidence: float option
    
    /// <summary>
    /// The probability distribution of the prediction, if applicable.
    /// </summary>
    ProbabilityDistribution: Map<obj, float> option
    
    /// <summary>
    /// Additional metadata about the prediction.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a data point for a machine learning model.
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
    /// The weight of the data point.
    /// </summary>
    Weight: float option
    
    /// <summary>
    /// Additional metadata about the data point.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a dataset for a machine learning model.
/// </summary>
type Dataset = {
    /// <summary>
    /// The name of the dataset.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The features in the dataset.
    /// </summary>
    Features: Feature list
    
    /// <summary>
    /// The label in the dataset, if any.
    /// </summary>
    Label: Label option
    
    /// <summary>
    /// The data points in the dataset.
    /// </summary>
    DataPoints: DataPoint list
    
    /// <summary>
    /// Additional metadata about the dataset.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a split of a dataset into training and testing sets.
/// </summary>
type DatasetSplit = {
    /// <summary>
    /// The training dataset.
    /// </summary>
    TrainingSet: Dataset
    
    /// <summary>
    /// The testing dataset.
    /// </summary>
    TestingSet: Dataset
    
    /// <summary>
    /// The validation dataset, if any.
    /// </summary>
    ValidationSet: Dataset option
    
    /// <summary>
    /// The split ratio (training/testing/validation).
    /// </summary>
    SplitRatio: float * float * float option
    
    /// <summary>
    /// Whether the split was stratified.
    /// </summary>
    IsStratified: bool
    
    /// <summary>
    /// Additional metadata about the split.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents hyperparameters for a machine learning model.
/// </summary>
type Hyperparameters = {
    /// <summary>
    /// The parameters of the model.
    /// </summary>
    Parameters: Map<string, obj>
    
    /// <summary>
    /// Additional metadata about the hyperparameters.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents metrics for evaluating a machine learning model.
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
    ConfusionMatrix: Map<obj, Map<obj, int>> option
    
    /// <summary>
    /// The area under the ROC curve of the model, if applicable.
    /// </summary>
    AUC: float option
    
    /// <summary>
    /// Custom metrics for the model.
    /// </summary>
    CustomMetrics: Map<string, float>
    
    /// <summary>
    /// Additional metadata about the metrics.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a machine learning model.
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
    /// The features used by the model.
    /// </summary>
    Features: Feature list
    
    /// <summary>
    /// The label predicted by the model, if any.
    /// </summary>
    Label: Label option
    
    /// <summary>
    /// The hyperparameters of the model.
    /// </summary>
    Hyperparameters: Hyperparameters
    
    /// <summary>
    /// The metrics of the model, if trained.
    /// </summary>
    Metrics: ModelMetrics option
    
    /// <summary>
    /// The creation time of the model.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The last training time of the model, if trained.
    /// </summary>
    LastTrainingTime: DateTime option
    
    /// <summary>
    /// The version of the model.
    /// </summary>
    Version: string
    
    /// <summary>
    /// Additional metadata about the model.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a training configuration for a machine learning model.
/// </summary>
type TrainingConfig = {
    /// <summary>
    /// The dataset to use for training.
    /// </summary>
    Dataset: Dataset
    
    /// <summary>
    /// The split to use for training, if any.
    /// </summary>
    Split: DatasetSplit option
    
    /// <summary>
    /// The hyperparameters to use for training.
    /// </summary>
    Hyperparameters: Hyperparameters
    
    /// <summary>
    /// The maximum number of iterations for training.
    /// </summary>
    MaxIterations: int option
    
    /// <summary>
    /// The maximum time for training.
    /// </summary>
    MaxTime: TimeSpan option
    
    /// <summary>
    /// The early stopping criteria for training.
    /// </summary>
    EarlyStoppingCriteria: Map<string, obj> option
    
    /// <summary>
    /// Whether to use cross-validation.
    /// </summary>
    UseCrossValidation: bool
    
    /// <summary>
    /// The number of folds for cross-validation, if used.
    /// </summary>
    CrossValidationFolds: int option
    
    /// <summary>
    /// Additional metadata about the training configuration.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a training result for a machine learning model.
/// </summary>
type TrainingResult = {
    /// <summary>
    /// The model that was trained.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The metrics of the model.
    /// </summary>
    Metrics: ModelMetrics
    
    /// <summary>
    /// The training time.
    /// </summary>
    TrainingTime: TimeSpan
    
    /// <summary>
    /// The number of iterations performed.
    /// </summary>
    Iterations: int
    
    /// <summary>
    /// Whether the training was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The error message, if training failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata about the training result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a prediction request for a machine learning model.
/// </summary>
type PredictionRequest = {
    /// <summary>
    /// The features to use for prediction.
    /// </summary>
    Features: Map<string, obj>
    
    /// <summary>
    /// Additional metadata about the prediction request.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a prediction result from a machine learning model.
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
    /// The prediction request.
    /// </summary>
    Request: PredictionRequest
    
    /// <summary>
    /// The prediction time.
    /// </summary>
    PredictionTime: TimeSpan
    
    /// <summary>
    /// Whether the prediction was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The error message, if prediction failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata about the prediction result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a feature importance result for a machine learning model.
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
    /// Additional metadata about the feature importance result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a model comparison result.
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
    /// Additional metadata about the model comparison result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a model export result.
/// </summary>
type ModelExportResult = {
    /// <summary>
    /// The model that was exported.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The format of the export.
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
    /// The error message, if export failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata about the model export result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a model import result.
/// </summary>
type ModelImportResult = {
    /// <summary>
    /// The imported model.
    /// </summary>
    Model: Model
    
    /// <summary>
    /// The format of the import.
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
    /// The error message, if import failed.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional metadata about the model import result.
    /// </summary>
    Metadata: Map<string, obj>
}
