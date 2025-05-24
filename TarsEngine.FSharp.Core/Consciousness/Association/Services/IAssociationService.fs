namespace TarsEngine.FSharp.Core.Consciousness.Association.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Association

/// <summary>
/// Interface for association services.
/// </summary>
type IAssociationService =
    /// <summary>
    /// Creates an association.
    /// </summary>
    /// <param name="source">The source concept.</param>
    /// <param name="target">The target concept.</param>
    /// <param name="type">The type of the association.</param>
    /// <param name="strength">The strength of the association.</param>
    /// <param name="description">The description of the association.</param>
    /// <param name="isBidirectional">Whether the association is bidirectional.</param>
    /// <returns>The created association.</returns>
    abstract member CreateAssociation : source:string * target:string * type':AssociationType * ?strength:AssociationStrength * ?description:string * ?isBidirectional:bool -> Task<Association>
    
    /// <summary>
    /// Gets an association by ID.
    /// </summary>
    /// <param name="id">The ID of the association.</param>
    /// <returns>The association, if found.</returns>
    abstract member GetAssociation : id:Guid -> Task<Association option>
    
    /// <summary>
    /// Gets all associations.
    /// </summary>
    /// <returns>The list of all associations.</returns>
    abstract member GetAllAssociations : unit -> Task<Association list>
    
    /// <summary>
    /// Updates an association.
    /// </summary>
    /// <param name="id">The ID of the association to update.</param>
    /// <param name="type">The new type of the association.</param>
    /// <param name="strength">The new strength of the association.</param>
    /// <param name="description">The new description of the association.</param>
    /// <param name="isBidirectional">Whether the association is bidirectional.</param>
    /// <returns>The updated association.</returns>
    abstract member UpdateAssociation : id:Guid * ?type':AssociationType * ?strength:AssociationStrength * ?description:string * ?isBidirectional:bool -> Task<Association>
    
    /// <summary>
    /// Deletes an association.
    /// </summary>
    /// <param name="id">The ID of the association to delete.</param>
    /// <returns>Whether the association was deleted.</returns>
    abstract member DeleteAssociation : id:Guid -> Task<bool>
    
    /// <summary>
    /// Activates an association.
    /// </summary>
    /// <param name="id">The ID of the association to activate.</param>
    /// <param name="activationStrength">The strength of the activation.</param>
    /// <param name="context">The context of the activation.</param>
    /// <param name="trigger">The trigger of the activation.</param>
    /// <returns>The association activation.</returns>
    abstract member ActivateAssociation : id:Guid * ?activationStrength:float * ?context:string * ?trigger:string -> Task<AssociationActivation>
    
    /// <summary>
    /// Creates an association network.
    /// </summary>
    /// <param name="name">The name of the network.</param>
    /// <param name="description">The description of the network.</param>
    /// <returns>The created network.</returns>
    abstract member CreateNetwork : name:string * ?description:string -> Task<AssociationNetwork>
    
    /// <summary>
    /// Gets an association network by ID.
    /// </summary>
    /// <param name="id">The ID of the network.</param>
    /// <returns>The network, if found.</returns>
    abstract member GetNetwork : id:Guid -> Task<AssociationNetwork option>
    
    /// <summary>
    /// Gets all association networks.
    /// </summary>
    /// <returns>The list of all networks.</returns>
    abstract member GetAllNetworks : unit -> Task<AssociationNetwork list>
    
    /// <summary>
    /// Adds an association to a network.
    /// </summary>
    /// <param name="networkId">The ID of the network.</param>
    /// <param name="associationId">The ID of the association.</param>
    /// <returns>The updated network.</returns>
    abstract member AddAssociationToNetwork : networkId:Guid * associationId:Guid -> Task<AssociationNetwork>
    
    /// <summary>
    /// Removes an association from a network.
    /// </summary>
    /// <param name="networkId">The ID of the network.</param>
    /// <param name="associationId">The ID of the association.</param>
    /// <returns>The updated network.</returns>
    abstract member RemoveAssociationFromNetwork : networkId:Guid * associationId:Guid -> Task<AssociationNetwork>
    
    /// <summary>
    /// Deletes an association network.
    /// </summary>
    /// <param name="id">The ID of the network to delete.</param>
    /// <returns>Whether the network was deleted.</returns>
    abstract member DeleteNetwork : id:Guid -> Task<bool>
    
    /// <summary>
    /// Finds associations between concepts.
    /// </summary>
    /// <param name="query">The association query.</param>
    /// <returns>The association query result.</returns>
    abstract member FindAssociations : query:AssociationQuery -> Task<AssociationQueryResult>
    
    /// <summary>
    /// Finds paths between concepts.
    /// </summary>
    /// <param name="source">The source concept.</param>
    /// <param name="target">The target concept.</param>
    /// <param name="maxPathLength">The maximum path length.</param>
    /// <param name="maxResults">The maximum number of results.</param>
    /// <returns>The list of association paths.</returns>
    abstract member FindPaths : source:string * target:string * ?maxPathLength:int * ?maxResults:int -> Task<AssociationPath list>
    
    /// <summary>
    /// Suggests associations.
    /// </summary>
    /// <param name="concept">The concept to suggest associations for.</param>
    /// <param name="maxSuggestions">The maximum number of suggestions.</param>
    /// <returns>The list of association suggestions.</returns>
    abstract member SuggestAssociations : concept:string * ?maxSuggestions:int -> Task<AssociationSuggestion list>
    
    /// <summary>
    /// Learns associations from text.
    /// </summary>
    /// <param name="text">The text to learn from.</param>
    /// <returns>The association learning result.</returns>
    abstract member LearnAssociationsFromText : text:string -> Task<AssociationLearningResult>
    
    /// <summary>
    /// Exports an association network.
    /// </summary>
    /// <param name="networkId">The ID of the network to export.</param>
    /// <param name="format">The format to export in.</param>
    /// <param name="path">The path to export to.</param>
    /// <returns>Whether the export was successful.</returns>
    abstract member ExportNetwork : networkId:Guid * format:string * path:string -> Task<bool>
    
    /// <summary>
    /// Imports an association network.
    /// </summary>
    /// <param name="format">The format to import from.</param>
    /// <param name="path">The path to import from.</param>
    /// <returns>The imported network.</returns>
    abstract member ImportNetwork : format:string * path:string -> Task<AssociationNetwork>
