namespace TarsEngine.ML.Core;

/// <summary>
/// Types of model improvements
/// </summary>
public enum ImprovementType
{
    /// <summary>
    /// More training data
    /// </summary>
    MoreData,
    
    /// <summary>
    /// Better feature engineering
    /// </summary>
    FeatureEngineering,
    
    /// <summary>
    /// Hyperparameter tuning
    /// </summary>
    HyperparameterTuning,
    
    /// <summary>
    /// Architecture change
    /// </summary>
    ArchitectureChange,
    
    /// <summary>
    /// Algorithm change
    /// </summary>
    AlgorithmChange,
    
    /// <summary>
    /// Ensemble creation
    /// </summary>
    EnsembleCreation,
    
    /// <summary>
    /// Transfer learning
    /// </summary>
    TransferLearning,
    
    /// <summary>
    /// Data cleaning
    /// </summary>
    DataCleaning,
    
    /// <summary>
    /// Regularization
    /// </summary>
    Regularization,
    
    /// <summary>
    /// Optimization
    /// </summary>
    Optimization,
    
    /// <summary>
    /// Knowledge integration
    /// </summary>
    KnowledgeIntegration,
    
    /// <summary>
    /// Meta-learning
    /// </summary>
    MetaLearning,
    
    /// <summary>
    /// Self-supervised learning
    /// </summary>
    SelfSupervisedLearning,
    
    /// <summary>
    /// Reinforcement learning
    /// </summary>
    ReinforcementLearning,
    
    /// <summary>
    /// Unsupervised learning
    /// </summary>
    UnsupervisedLearning,
    
    /// <summary>
    /// Neural architecture search
    /// </summary>
    NeuralArchitectureSearch,
    
    /// <summary>
    /// Automated machine learning
    /// </summary>
    AutoML,
    
    /// <summary>
    /// Other improvement type
    /// </summary>
    Other
}
