namespace TarsEngine.FSharp.Core.Consciousness.Conceptual.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Conceptual
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Interface for conceptual services.
/// </summary>
type IConceptualService =
    /// <summary>
    /// Creates a concept.
    /// </summary>
    /// <param name="name">The name of the concept.</param>
    /// <param name="description">The description of the concept.</param>
    /// <param name="type">The type of the concept.</param>
    /// <param name="complexity">The complexity of the concept.</param>
    /// <param name="attributes">The attributes of the concept.</param>
    /// <param name="examples">The examples of the concept.</param>
    /// <param name="tags">The tags of the concept.</param>
    /// <returns>The created concept.</returns>
    abstract member CreateConcept : name:string * description:string * type':ConceptType * ?complexity:ConceptComplexity * ?attributes:Map<string, obj> * ?examples:string list * ?tags:string list -> Task<Concept>
    
    /// <summary>
    /// Gets a concept by ID.
    /// </summary>
    /// <param name="id">The ID of the concept.</param>
    /// <returns>The concept, if found.</returns>
    abstract member GetConcept : id:Guid -> Task<Concept option>
    
    /// <summary>
    /// Gets all concepts.
    /// </summary>
    /// <returns>The list of all concepts.</returns>
    abstract member GetAllConcepts : unit -> Task<Concept list>
    
    /// <summary>
    /// Updates a concept.
    /// </summary>
    /// <param name="id">The ID of the concept to update.</param>
    /// <param name="description">The new description of the concept.</param>
    /// <param name="type">The new type of the concept.</param>
    /// <param name="complexity">The new complexity of the concept.</param>
    /// <param name="attributes">The new attributes of the concept.</param>
    /// <param name="examples">The new examples of the concept.</param>
    /// <param name="tags">The new tags of the concept.</param>
    /// <returns>The updated concept.</returns>
    abstract member UpdateConcept : id:Guid * ?description:string * ?type':ConceptType * ?complexity:ConceptComplexity * ?attributes:Map<string, obj> * ?examples:string list * ?tags:string list -> Task<Concept>
    
    /// <summary>
    /// Deletes a concept.
    /// </summary>
    /// <param name="id">The ID of the concept to delete.</param>
    /// <returns>Whether the concept was deleted.</returns>
    abstract member DeleteConcept : id:Guid -> Task<bool>
    
    /// <summary>
    /// Activates a concept.
    /// </summary>
    /// <param name="id">The ID of the concept to activate.</param>
    /// <param name="activationStrength">The strength of the activation.</param>
    /// <param name="context">The context of the activation.</param>
    /// <param name="trigger">The trigger of the activation.</param>
    /// <returns>The concept activation.</returns>
    abstract member ActivateConcept : id:Guid * ?activationStrength:float * ?context:string * ?trigger:string -> Task<ConceptActivation>
    
    /// <summary>
    /// Adds an emotion to a concept.
    /// </summary>
    /// <param name="conceptId">The ID of the concept.</param>
    /// <param name="emotion">The emotion to add.</param>
    /// <returns>The updated concept.</returns>
    abstract member AddEmotionToConcept : conceptId:Guid * emotion:Emotion -> Task<Concept>
    
    /// <summary>
    /// Relates two concepts.
    /// </summary>
    /// <param name="sourceId">The ID of the source concept.</param>
    /// <param name="targetId">The ID of the target concept.</param>
    /// <param name="strength">The strength of the relation.</param>
    /// <returns>The updated source concept.</returns>
    abstract member RelateConcepts : sourceId:Guid * targetId:Guid * strength:float -> Task<Concept>
    
    /// <summary>
    /// Creates a concept hierarchy.
    /// </summary>
    /// <param name="name">The name of the hierarchy.</param>
    /// <param name="description">The description of the hierarchy.</param>
    /// <param name="rootConcepts">The root concepts of the hierarchy.</param>
    /// <returns>The created hierarchy.</returns>
    abstract member CreateHierarchy : name:string * ?description:string * ?rootConcepts:Guid list -> Task<ConceptHierarchy>
    
    /// <summary>
    /// Gets a concept hierarchy by ID.
    /// </summary>
    /// <param name="id">The ID of the hierarchy.</param>
    /// <returns>The hierarchy, if found.</returns>
    abstract member GetHierarchy : id:Guid -> Task<ConceptHierarchy option>
    
    /// <summary>
    /// Gets all concept hierarchies.
    /// </summary>
    /// <returns>The list of all hierarchies.</returns>
    abstract member GetAllHierarchies : unit -> Task<ConceptHierarchy list>
    
    /// <summary>
    /// Adds a concept to a hierarchy.
    /// </summary>
    /// <param name="hierarchyId">The ID of the hierarchy.</param>
    /// <param name="conceptId">The ID of the concept.</param>
    /// <param name="parentId">The ID of the parent concept, if any.</param>
    /// <returns>The updated hierarchy.</returns>
    abstract member AddConceptToHierarchy : hierarchyId:Guid * conceptId:Guid * ?parentId:Guid -> Task<ConceptHierarchy>
    
    /// <summary>
    /// Removes a concept from a hierarchy.
    /// </summary>
    /// <param name="hierarchyId">The ID of the hierarchy.</param>
    /// <param name="conceptId">The ID of the concept.</param>
    /// <returns>The updated hierarchy.</returns>
    abstract member RemoveConceptFromHierarchy : hierarchyId:Guid * conceptId:Guid -> Task<ConceptHierarchy>
    
    /// <summary>
    /// Deletes a concept hierarchy.
    /// </summary>
    /// <param name="id">The ID of the hierarchy to delete.</param>
    /// <returns>Whether the hierarchy was deleted.</returns>
    abstract member DeleteHierarchy : id:Guid -> Task<bool>
    
    /// <summary>
    /// Finds concepts.
    /// </summary>
    /// <param name="query">The concept query.</param>
    /// <returns>The concept query result.</returns>
    abstract member FindConcepts : query:ConceptQuery -> Task<ConceptQueryResult>
    
    /// <summary>
    /// Suggests concepts.
    /// </summary>
    /// <param name="text">The text to suggest concepts for.</param>
    /// <param name="maxSuggestions">The maximum number of suggestions.</param>
    /// <returns>The list of concept suggestions.</returns>
    abstract member SuggestConcepts : text:string * ?maxSuggestions:int -> Task<ConceptSuggestion list>
    
    /// <summary>
    /// Learns concepts from text.
    /// </summary>
    /// <param name="text">The text to learn from.</param>
    /// <returns>The concept learning result.</returns>
    abstract member LearnConceptsFromText : text:string -> Task<ConceptLearningResult>
    
    /// <summary>
    /// Exports a concept hierarchy.
    /// </summary>
    /// <param name="hierarchyId">The ID of the hierarchy to export.</param>
    /// <param name="format">The format to export in.</param>
    /// <param name="path">The path to export to.</param>
    /// <returns>Whether the export was successful.</returns>
    abstract member ExportHierarchy : hierarchyId:Guid * format:string * path:string -> Task<bool>
    
    /// <summary>
    /// Imports a concept hierarchy.
    /// </summary>
    /// <param name="format">The format to import from.</param>
    /// <param name="path">The path to import from.</param>
    /// <returns>The imported hierarchy.</returns>
    abstract member ImportHierarchy : format:string * path:string -> Task<ConceptHierarchy>
