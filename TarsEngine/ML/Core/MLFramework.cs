using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Data;
using TarsEngine.Models;

namespace TarsEngine.ML.Core;

/// <summary>
/// Core machine learning framework for TARS intelligence
/// </summary>
public class MLFramework
{
    private readonly ILogger<MLFramework> _logger;
    private readonly MLContext _mlContext;
    private readonly string _modelBasePath;
    private readonly Dictionary<string, ITransformer> _loadedModels = new();
    private readonly Dictionary<string, PredictionEngine<object, object>> _predictionEngines = new();
    private readonly Dictionary<string, DateTime> _modelLastUpdated = new();
    private readonly Dictionary<string, ModelMetadata> _modelMetadata = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="MLFramework"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public MLFramework(ILogger<MLFramework> logger)
    {
        _logger = logger;
        _mlContext = new MLContext(seed: 42);
        _modelBasePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models");
        
        // Ensure model directory exists
        Directory.CreateDirectory(_modelBasePath);
    }

    /// <summary>
    /// Loads a model from disk or creates a new one if it doesn't exist
    /// </summary>
    /// <typeparam name="TData">The data type</typeparam>
    /// <typeparam name="TPrediction">The prediction type</typeparam>
    /// <param name="modelName">The model name</param>
    /// <param name="createModelPipeline">Function to create the model pipeline if needed</param>
    /// <param name="trainingData">Optional training data to train/retrain the model</param>
    /// <returns>True if the model was loaded or created successfully</returns>
    public async Task<bool> LoadOrCreateModelAsync<TData, TPrediction>(
        string modelName,
        Func<MLContext, IEstimator<ITransformer>> createModelPipeline,
        IEnumerable<TData>? trainingData = null)
        where TData : class
        where TPrediction : class, new()
    {
        try
        {
            var modelPath = Path.Combine(_modelBasePath, $"{modelName}.zip");
            
            // Check if model exists and is up to date
            if (File.Exists(modelPath))
            {
                var fileInfo = new FileInfo(modelPath);
                
                // If model is already loaded and hasn't changed, return
                if (_loadedModels.ContainsKey(modelName) && 
                    _modelLastUpdated.TryGetValue(modelName, out var lastUpdated) && 
                    lastUpdated >= fileInfo.LastWriteTimeUtc)
                {
                    return true;
                }
                
                // Load model
                _logger.LogInformation("Loading model: {ModelName}", modelName);
                var model = _mlContext.Model.Load(modelPath, out var modelSchema);
                
                // Store model
                _loadedModels[modelName] = model;
                _modelLastUpdated[modelName] = fileInfo.LastWriteTimeUtc;
                
                // Create prediction engine
                var predictionEngine = _mlContext.Model.CreatePredictionEngine<TData, TPrediction>(model);
                _predictionEngines[modelName] = predictionEngine as PredictionEngine<object, object>;
                
                // Load metadata if exists
                await LoadModelMetadataAsync(modelName);
                
                return true;
            }
            
            // If no training data provided, can't create model
            if (trainingData == null || !trainingData.Any())
            {
                _logger.LogWarning("No model exists and no training data provided for: {ModelName}", modelName);
                return false;
            }
            
            // Create and train model
            _logger.LogInformation("Creating new model: {ModelName}", modelName);
            
            // Create data view
            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            
            // Create pipeline
            var pipeline = createModelPipeline(_mlContext);
            
            // Train model
            var model = pipeline.Fit(dataView);
            
            // Save model
            _mlContext.Model.Save(model, dataView.Schema, modelPath);
            
            // Store model
            _loadedModels[modelName] = model;
            _modelLastUpdated[modelName] = DateTime.UtcNow;
            
            // Create prediction engine
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<TData, TPrediction>(model);
            _predictionEngines[modelName] = predictionEngine as PredictionEngine<object, object>;
            
            // Create and save metadata
            var metadata = new ModelMetadata
            {
                ModelName = modelName,
                CreatedAt = DateTime.UtcNow,
                LastUpdatedAt = DateTime.UtcNow,
                DataType = typeof(TData).Name,
                PredictionType = typeof(TPrediction).Name,
                TrainingExamples = trainingData.Count(),
                ModelPath = modelPath,
                Metrics = new Dictionary<string, double>(),
                HyperParameters = new Dictionary<string, string>(),
                Tags = new List<string>()
            };
            
            _modelMetadata[modelName] = metadata;
            await SaveModelMetadataAsync(modelName);
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading or creating model: {ModelName}", modelName);
            return false;
        }
    }

    /// <summary>
    /// Trains or retrains a model
    /// </summary>
    /// <typeparam name="TData">The data type</typeparam>
    /// <typeparam name="TPrediction">The prediction type</typeparam>
    /// <param name="modelName">The model name</param>
    /// <param name="createModelPipeline">Function to create the model pipeline</param>
    /// <param name="trainingData">The training data</param>
    /// <param name="evaluateModel">Optional function to evaluate the model and return metrics</param>
    /// <param name="hyperParameters">Optional hyperparameters for the model</param>
    /// <returns>True if the model was trained successfully</returns>
    public async Task<bool> TrainModelAsync<TData, TPrediction>(
        string modelName,
        Func<MLContext, IEstimator<ITransformer>> createModelPipeline,
        IEnumerable<TData> trainingData,
        Func<ITransformer, IDataView, Dictionary<string, double>>? evaluateModel = null,
        Dictionary<string, string>? hyperParameters = null)
        where TData : class
        where TPrediction : class, new()
    {
        try
        {
            _logger.LogInformation("Training model: {ModelName}", modelName);
            
            var modelPath = Path.Combine(_modelBasePath, $"{modelName}.zip");
            
            // Create data view
            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            
            // Split data for training and evaluation
            var dataSplit = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = dataSplit.TrainSet;
            var testData = dataSplit.TestSet;
            
            // Create pipeline
            var pipeline = createModelPipeline(_mlContext);
            
            // Train model
            var model = pipeline.Fit(trainData);
            
            // Evaluate model if evaluator provided
            Dictionary<string, double> metrics = new();
            if (evaluateModel != null)
            {
                metrics = evaluateModel(model, testData);
                _logger.LogInformation("Model evaluation metrics: {Metrics}", 
                    string.Join(", ", metrics.Select(m => $"{m.Key}={m.Value:F4}")));
            }
            
            // Save model
            _mlContext.Model.Save(model, dataView.Schema, modelPath);
            
            // Store model
            _loadedModels[modelName] = model;
            _modelLastUpdated[modelName] = DateTime.UtcNow;
            
            // Create prediction engine
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<TData, TPrediction>(model);
            _predictionEngines[modelName] = predictionEngine as PredictionEngine<object, object>;
            
            // Update metadata
            var metadata = _modelMetadata.TryGetValue(modelName, out var existingMetadata)
                ? existingMetadata
                : new ModelMetadata
                {
                    ModelName = modelName,
                    CreatedAt = DateTime.UtcNow,
                    DataType = typeof(TData).Name,
                    PredictionType = typeof(TPrediction).Name,
                    ModelPath = modelPath,
                    Metrics = new Dictionary<string, double>(),
                    HyperParameters = new Dictionary<string, string>(),
                    Tags = new List<string>()
                };
            
            metadata.LastUpdatedAt = DateTime.UtcNow;
            metadata.TrainingExamples = trainingData.Count();
            metadata.Metrics = metrics;
            
            if (hyperParameters != null)
            {
                metadata.HyperParameters = hyperParameters;
            }
            
            _modelMetadata[modelName] = metadata;
            await SaveModelMetadataAsync(modelName);
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error training model: {ModelName}", modelName);
            return false;
        }
    }

    /// <summary>
    /// Makes a prediction using a loaded model
    /// </summary>
    /// <typeparam name="TData">The data type</typeparam>
    /// <typeparam name="TPrediction">The prediction type</typeparam>
    /// <param name="modelName">The model name</param>
    /// <param name="data">The input data</param>
    /// <returns>The prediction</returns>
    public TPrediction? Predict<TData, TPrediction>(string modelName, TData data)
        where TData : class
        where TPrediction : class, new()
    {
        try
        {
            if (!_predictionEngines.TryGetValue(modelName, out var engine))
            {
                _logger.LogWarning("Prediction engine not found for model: {ModelName}", modelName);
                return null;
            }
            
            var typedEngine = engine as PredictionEngine<TData, TPrediction>;
            if (typedEngine == null)
            {
                _logger.LogWarning("Invalid prediction engine type for model: {ModelName}", modelName);
                return null;
            }
            
            return typedEngine.Predict(data);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error making prediction with model: {ModelName}", modelName);
            return null;
        }
    }

    /// <summary>
    /// Gets all available models
    /// </summary>
    /// <returns>List of model metadata</returns>
    public List<ModelMetadata> GetAvailableModels()
    {
        return _modelMetadata.Values.ToList();
    }

    /// <summary>
    /// Gets metadata for a specific model
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <returns>The model metadata, or null if not found</returns>
    public ModelMetadata? GetModelMetadata(string modelName)
    {
        return _modelMetadata.TryGetValue(modelName, out var metadata) ? metadata : null;
    }

    /// <summary>
    /// Deletes a model
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <returns>True if the model was deleted successfully</returns>
    public async Task<bool> DeleteModelAsync(string modelName)
    {
        try
        {
            var modelPath = Path.Combine(_modelBasePath, $"{modelName}.zip");
            var metadataPath = Path.Combine(_modelBasePath, $"{modelName}.metadata.json");
            
            // Remove from dictionaries
            _loadedModels.Remove(modelName);
            _predictionEngines.Remove(modelName);
            _modelLastUpdated.Remove(modelName);
            _modelMetadata.Remove(modelName);
            
            // Delete files
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
            }
            
            if (File.Exists(metadataPath))
            {
                File.Delete(metadataPath);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting model: {ModelName}", modelName);
            return false;
        }
    }

    /// <summary>
    /// Loads model metadata from disk
    /// </summary>
    /// <param name="modelName">The model name</param>
    private async Task LoadModelMetadataAsync(string modelName)
    {
        try
        {
            var metadataPath = Path.Combine(_modelBasePath, $"{modelName}.metadata.json");
            if (!File.Exists(metadataPath))
            {
                return;
            }
            
            var json = await File.ReadAllTextAsync(metadataPath);
            var metadata = System.Text.Json.JsonSerializer.Deserialize<ModelMetadata>(json);
            
            if (metadata != null)
            {
                _modelMetadata[modelName] = metadata;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading model metadata: {ModelName}", modelName);
        }
    }

    /// <summary>
    /// Saves model metadata to disk
    /// </summary>
    /// <param name="modelName">The model name</param>
    private async Task SaveModelMetadataAsync(string modelName)
    {
        try
        {
            if (!_modelMetadata.TryGetValue(modelName, out var metadata))
            {
                return;
            }
            
            var metadataPath = Path.Combine(_modelBasePath, $"{modelName}.metadata.json");
            var json = System.Text.Json.JsonSerializer.Serialize(metadata, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
            
            await File.WriteAllTextAsync(metadataPath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving model metadata: {ModelName}", modelName);
        }
    }
}
