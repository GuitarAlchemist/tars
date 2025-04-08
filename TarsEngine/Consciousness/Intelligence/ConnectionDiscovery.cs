using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Discovers distant connections between concepts for insight generation
/// </summary>
public class ConnectionDiscovery
{
    private readonly ILogger<ConnectionDiscovery> _logger;
    private readonly Dictionary<string, ConceptNode> _semanticNetwork = new();
    private readonly List<ConceptConnection> _connections = new();
    private readonly System.Random _random = new System.Random();

    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _connectionDiscoveryLevel = 0.4; // Starting with moderate connection discovery
    private double _distantConnectionThreshold = 0.3; // Starting with moderate distant connection threshold
    private double _connectionNoveltyThreshold = 0.5; // Starting with moderate connection novelty threshold

    /// <summary>
    /// Gets the connection discovery level (0.0 to 1.0)
    /// </summary>
    public double ConnectionDiscoveryLevel => _connectionDiscoveryLevel;

    /// <summary>
    /// Gets the distant connection threshold (0.0 to 1.0)
    /// </summary>
    public double DistantConnectionThreshold => _distantConnectionThreshold;

    /// <summary>
    /// Gets the connection novelty threshold (0.0 to 1.0)
    /// </summary>
    public double ConnectionNoveltyThreshold => _connectionNoveltyThreshold;

    /// <summary>
    /// Gets the connections
    /// </summary>
    public IReadOnlyList<ConceptConnection> Connections => _connections.AsReadOnly();

    /// <summary>
    /// Initializes a new instance of the <see cref="ConnectionDiscovery"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ConnectionDiscovery(ILogger<ConnectionDiscovery> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the connection discovery
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing connection discovery");

            // Initialize semantic network
            InitializeSemanticNetwork();

            _isInitialized = true;
            _logger.LogInformation("Connection discovery initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing connection discovery");
            return false;
        }
    }

    /// <summary>
    /// Initializes the semantic network
    /// </summary>
    private void InitializeSemanticNetwork()
    {
        // Create concept nodes
        var concepts = new List<string>
        {
            "consciousness", "intelligence", "creativity", "learning", "emotion",
            "memory", "perception", "reasoning", "language", "problem-solving",
            "adaptation", "evolution", "complexity", "emergence", "self-organization",
            "pattern", "structure", "function", "process", "system",
            "information", "knowledge", "wisdom", "understanding", "insight",
            "innovation", "discovery", "exploration", "curiosity", "wonder"
        };

        // Add concept nodes to network
        foreach (var concept in concepts)
        {
            _semanticNetwork[concept] = new ConceptNode
            {
                Id = Guid.NewGuid().ToString(),
                Name = concept,
                Attributes = GenerateConceptAttributes(concept)
            };
        }

        // Create direct connections
        CreateDirectConnections();
    }

    /// <summary>
    /// Generates concept attributes
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The concept attributes</returns>
    private Dictionary<string, double> GenerateConceptAttributes(string concept)
    {
        var attributes = new Dictionary<string, double>();

        // Generate attributes based on concept
        switch (concept)
        {
            case "consciousness":
                attributes["awareness"] = 0.9;
                attributes["subjectivity"] = 0.8;
                attributes["experience"] = 0.9;
                attributes["self"] = 0.7;
                attributes["qualia"] = 0.8;
                break;

            case "intelligence":
                attributes["problem-solving"] = 0.9;
                attributes["adaptation"] = 0.8;
                attributes["learning"] = 0.9;
                attributes["reasoning"] = 0.8;
                attributes["knowledge"] = 0.7;
                break;

            case "creativity":
                attributes["originality"] = 0.9;
                attributes["imagination"] = 0.8;
                attributes["innovation"] = 0.9;
                attributes["divergent-thinking"] = 0.8;
                attributes["synthesis"] = 0.7;
                break;

            case "learning":
                attributes["adaptation"] = 0.8;
                attributes["memory"] = 0.7;
                attributes["knowledge"] = 0.9;
                attributes["experience"] = 0.8;
                attributes["growth"] = 0.7;
                break;

            case "emotion":
                attributes["feeling"] = 0.9;
                attributes["affect"] = 0.8;
                attributes["valence"] = 0.7;
                attributes["arousal"] = 0.7;
                attributes["motivation"] = 0.6;
                break;

            // Add more concept attributes as needed

            default:
                // Generate random attributes for other concepts
                var possibleAttributes = new List<string>
                {
                    "complexity", "structure", "function", "process", "pattern",
                    "emergence", "organization", "integration", "differentiation", "adaptation",
                    "evolution", "development", "growth", "transformation", "dynamics"
                };

                // Assign random attributes
                int attributeCount = 3 + _random.Next(3);
                for (int i = 0; i < attributeCount; i++)
                {
                    var attribute = possibleAttributes[_random.Next(possibleAttributes.Count)];
                    if (!attributes.ContainsKey(attribute))
                    {
                        attributes[attribute] = 0.5 + (0.5 * _random.NextDouble());
                    }
                }
                break;
        }

        return attributes;
    }

    /// <summary>
    /// Creates direct connections
    /// </summary>
    private void CreateDirectConnections()
    {
        // Define direct connections
        var directConnections = new List<(string, string, double)>
        {
            ("consciousness", "intelligence", 0.7),
            ("consciousness", "perception", 0.8),
            ("consciousness", "self-organization", 0.6),
            ("intelligence", "learning", 0.9),
            ("intelligence", "problem-solving", 0.9),
            ("intelligence", "adaptation", 0.8),
            ("creativity", "intelligence", 0.7),
            ("creativity", "innovation", 0.9),
            ("creativity", "imagination", 0.9),
            ("learning", "memory", 0.8),
            ("learning", "adaptation", 0.8),
            ("learning", "knowledge", 0.9),
            ("emotion", "consciousness", 0.7),
            ("emotion", "motivation", 0.8),
            ("memory", "learning", 0.8),
            ("memory", "knowledge", 0.7),
            ("perception", "consciousness", 0.8),
            ("perception", "information", 0.7),
            ("reasoning", "intelligence", 0.9),
            ("reasoning", "problem-solving", 0.8),
            ("language", "communication", 0.9),
            ("language", "knowledge", 0.7),
            ("problem-solving", "intelligence", 0.9),
            ("problem-solving", "creativity", 0.7),
            ("adaptation", "learning", 0.8),
            ("adaptation", "evolution", 0.7),
            ("evolution", "adaptation", 0.7),
            ("evolution", "complexity", 0.6),
            ("complexity", "emergence", 0.8),
            ("complexity", "self-organization", 0.7),
            ("emergence", "complexity", 0.8),
            ("emergence", "self-organization", 0.8),
            ("self-organization", "emergence", 0.8),
            ("self-organization", "complexity", 0.7),
            ("pattern", "structure", 0.8),
            ("pattern", "recognition", 0.7),
            ("structure", "pattern", 0.8),
            ("structure", "function", 0.7),
            ("function", "structure", 0.7),
            ("function", "process", 0.7),
            ("process", "function", 0.7),
            ("process", "system", 0.8),
            ("system", "process", 0.8),
            ("system", "complexity", 0.7),
            ("information", "knowledge", 0.8),
            ("information", "perception", 0.7),
            ("knowledge", "information", 0.8),
            ("knowledge", "wisdom", 0.7),
            ("wisdom", "knowledge", 0.7),
            ("wisdom", "understanding", 0.8),
            ("understanding", "wisdom", 0.8),
            ("understanding", "insight", 0.8),
            ("insight", "understanding", 0.8),
            ("insight", "discovery", 0.7),
            ("innovation", "creativity", 0.9),
            ("innovation", "discovery", 0.7),
            ("discovery", "insight", 0.7),
            ("discovery", "exploration", 0.8),
            ("exploration", "discovery", 0.8),
            ("exploration", "curiosity", 0.9),
            ("curiosity", "exploration", 0.9),
            ("curiosity", "wonder", 0.8),
            ("wonder", "curiosity", 0.8),
            ("wonder", "awe", 0.7)
        };

        // Create connections
        foreach (var (concept1, concept2, strength) in directConnections)
        {
            if (_semanticNetwork.ContainsKey(concept1) && _semanticNetwork.ContainsKey(concept2))
            {
                CreateConnection(concept1, concept2, strength, ConnectionType.Direct);
            }
        }
    }

    /// <summary>
    /// Creates a connection
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <param name="strength">The connection strength</param>
    /// <param name="type">The connection type</param>
    /// <returns>The created connection</returns>
    private ConceptConnection CreateConnection(string concept1, string concept2, double strength, ConnectionType type)
    {
        // Create connection
        var connection = new ConceptConnection
        {
            Id = Guid.NewGuid().ToString(),
            Concept1 = concept1,
            Concept2 = concept2,
            Strength = strength,
            Type = type,
            CreationTimestamp = DateTime.UtcNow
        };

        // Add to connections list
        _connections.Add(connection);

        // Update concept nodes
        _semanticNetwork[concept1].ConnectionIds.Add(connection.Id);
        _semanticNetwork[concept2].ConnectionIds.Add(connection.Id);

        return connection;
    }

    /// <summary>
    /// Activates the connection discovery
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate connection discovery: not initialized");
            return false;
        }

        if (_isActive)
        {
            _logger.LogInformation("Connection discovery is already active");
            return true;
        }

        try
        {
            _logger.LogInformation("Activating connection discovery");

            _isActive = true;
            _logger.LogInformation("Connection discovery activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating connection discovery");
            return false;
        }
    }

    /// <summary>
    /// Deactivates the connection discovery
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Connection discovery is already inactive");
            return true;
        }

        try
        {
            _logger.LogInformation("Deactivating connection discovery");

            _isActive = false;
            _logger.LogInformation("Connection discovery deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating connection discovery");
            return false;
        }
    }

    /// <summary>
    /// Updates the connection discovery
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return false;
        }

        try
        {
            // Gradually increase connection discovery level over time (very slowly)
            if (_connectionDiscoveryLevel < 0.95)
            {
                _connectionDiscoveryLevel += 0.0001 * _random.NextDouble();
                _connectionDiscoveryLevel = Math.Min(_connectionDiscoveryLevel, 1.0);
            }

            // Gradually decrease distant connection threshold over time (very slowly)
            // Lower threshold means more distant connections are considered
            if (_distantConnectionThreshold > 0.1)
            {
                _distantConnectionThreshold -= 0.0001 * _random.NextDouble();
                _distantConnectionThreshold = Math.Max(0.1, _distantConnectionThreshold);
            }

            // Gradually decrease connection novelty threshold over time (very slowly)
            // Lower threshold means more novel connections are considered
            if (_connectionNoveltyThreshold > 0.2)
            {
                _connectionNoveltyThreshold -= 0.0001 * _random.NextDouble();
                _connectionNoveltyThreshold = Math.Max(0.2, _connectionNoveltyThreshold);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating connection discovery");
            return false;
        }
    }

    /// <summary>
    /// Discovers distant connections
    /// </summary>
    /// <returns>The discovered connections</returns>
    public async Task<List<ConceptConnection>> DiscoverDistantConnectionsAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot discover distant connections: connection discovery not initialized or active");
            return new List<ConceptConnection>();
        }

        try
        {
            _logger.LogInformation("Discovering distant connections");

            var discoveredConnections = new List<ConceptConnection>();

            // Get all concept pairs
            var concepts = _semanticNetwork.Keys.ToList();
            var conceptPairs = new List<(string, string)>();

            for (int i = 0; i < concepts.Count; i++)
            {
                for (int j = i + 1; j < concepts.Count; j++)
                {
                    conceptPairs.Add((concepts[i], concepts[j]));
                }
            }

            // Shuffle concept pairs
            conceptPairs = conceptPairs.OrderBy(_ => _random.Next()).ToList();

            // Limit number of pairs to evaluate
            int pairsToEvaluate = Math.Min(20, conceptPairs.Count);

            // Evaluate concept pairs
            for (int i = 0; i < pairsToEvaluate; i++)
            {
                var (concept1, concept2) = conceptPairs[i];

                // Check if direct connection already exists
                if (HasDirectConnection(concept1, concept2))
                {
                    continue;
                }

                // Calculate semantic distance
                double semanticDistance = CalculateSemanticDistance(concept1, concept2);

                // Check if distant connection
                if (semanticDistance > _distantConnectionThreshold)
                {
                    // Calculate connection strength
                    double connectionStrength = CalculateConnectionStrength(concept1, concept2);

                    // Calculate connection novelty
                    double connectionNovelty = CalculateConnectionNovelty(concept1, concept2);

                    // Check if novel connection
                    if (connectionNovelty > _connectionNoveltyThreshold)
                    {
                        // Create distant connection
                        var connection = CreateConnection(concept1, concept2, connectionStrength, ConnectionType.Distant);

                        // Add connection attributes
                        connection.Attributes["semanticDistance"] = semanticDistance;
                        connection.Attributes["novelty"] = connectionNovelty;
                        connection.Attributes["discoveryLevel"] = _connectionDiscoveryLevel;

                        // Add to discovered connections
                        discoveredConnections.Add(connection);

                        _logger.LogInformation("Discovered distant connection: {Concept1} - {Concept2} (Strength: {Strength:F2}, Novelty: {Novelty:F2})",
                            concept1, concept2, connectionStrength, connectionNovelty);
                    }
                }
            }

            return discoveredConnections;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error discovering distant connections");
            return new List<ConceptConnection>();
        }
    }

    /// <summary>
    /// Checks if a direct connection exists
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>True if a direct connection exists</returns>
    private bool HasDirectConnection(string concept1, string concept2)
    {
        return _connections.Any(c =>
            (c.Concept1 == concept1 && c.Concept2 == concept2) ||
            (c.Concept1 == concept2 && c.Concept2 == concept1));
    }

    /// <summary>
    /// Calculates the semantic distance
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The semantic distance</returns>
    private double CalculateSemanticDistance(string concept1, string concept2)
    {
        // Get concept nodes
        var node1 = _semanticNetwork[concept1];
        var node2 = _semanticNetwork[concept2];

        // Calculate attribute similarity
        double attributeSimilarity = CalculateAttributeSimilarity(node1.Attributes, node2.Attributes);

        // Calculate path distance
        double pathDistance = CalculatePathDistance(concept1, concept2);

        // Combine attribute similarity and path distance
        double semanticDistance = (1.0 - attributeSimilarity) * 0.5 + pathDistance * 0.5;

        return semanticDistance;
    }

    /// <summary>
    /// Calculates the attribute similarity
    /// </summary>
    /// <param name="attributes1">The first attributes</param>
    /// <param name="attributes2">The second attributes</param>
    /// <returns>The attribute similarity</returns>
    private double CalculateAttributeSimilarity(Dictionary<string, double> attributes1, Dictionary<string, double> attributes2)
    {
        // Get all attribute keys
        var allKeys = attributes1.Keys.Union(attributes2.Keys).ToList();

        if (allKeys.Count == 0)
        {
            return 0.0;
        }

        double totalSimilarity = 0.0;

        // Calculate similarity for each attribute
        foreach (var key in allKeys)
        {
            double value1 = attributes1.TryGetValue(key, out var v1) ? v1 : 0.0;
            double value2 = attributes2.TryGetValue(key, out var v2) ? v2 : 0.0;

            // Calculate attribute similarity
            double attributeSimilarity = 1.0 - Math.Abs(value1 - value2);

            totalSimilarity += attributeSimilarity;
        }

        // Calculate average similarity
        return totalSimilarity / allKeys.Count;
    }

    /// <summary>
    /// Calculates the path distance
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The path distance</returns>
    private double CalculatePathDistance(string concept1, string concept2)
    {
        // Simple BFS to find shortest path
        var visited = new HashSet<string>();
        var queue = new Queue<(string, int)>();

        queue.Enqueue((concept1, 0));
        visited.Add(concept1);

        while (queue.Count > 0)
        {
            var (current, distance) = queue.Dequeue();

            if (current == concept2)
            {
                // Found path, normalize distance
                return Math.Min(1.0, distance / 5.0);
            }

            // Get connected concepts
            var connectedConcepts = GetConnectedConcepts(current);

            foreach (var connectedConcept in connectedConcepts)
            {
                if (!visited.Contains(connectedConcept))
                {
                    queue.Enqueue((connectedConcept, distance + 1));
                    visited.Add(connectedConcept);
                }
            }
        }

        // No path found, maximum distance
        return 1.0;
    }

    /// <summary>
    /// Gets connected concepts
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The connected concepts</returns>
    private List<string> GetConnectedConcepts(string concept)
    {
        var connectedConcepts = new List<string>();

        // Get concept node
        var node = _semanticNetwork[concept];

        // Get connections
        foreach (var connectionId in node.ConnectionIds)
        {
            var connection = _connections.First(c => c.Id == connectionId);

            // Add connected concept
            if (connection.Concept1 == concept)
            {
                connectedConcepts.Add(connection.Concept2);
            }
            else
            {
                connectedConcepts.Add(connection.Concept1);
            }
        }

        return connectedConcepts;
    }

    /// <summary>
    /// Calculates the connection strength
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The connection strength</returns>
    private double CalculateConnectionStrength(string concept1, string concept2)
    {
        // Get concept nodes
        var node1 = _semanticNetwork[concept1];
        var node2 = _semanticNetwork[concept2];

        // Calculate attribute similarity
        double attributeSimilarity = CalculateAttributeSimilarity(node1.Attributes, node2.Attributes);

        // Calculate connection strength based on attribute similarity and connection discovery level
        double connectionStrength = attributeSimilarity * 0.7 + _connectionDiscoveryLevel * 0.3;

        // Add some randomness
        connectionStrength = Math.Max(0.1, Math.Min(0.9, connectionStrength + (0.2 * (_random.NextDouble() - 0.5))));

        return connectionStrength;
    }

    /// <summary>
    /// Calculates the connection novelty
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The connection novelty</returns>
    private double CalculateConnectionNovelty(string concept1, string concept2)
    {
        // Calculate path distance
        double pathDistance = CalculatePathDistance(concept1, concept2);

        // Calculate connection novelty based on path distance and connection discovery level
        double connectionNovelty = pathDistance * 0.7 + _connectionDiscoveryLevel * 0.3;

        // Add some randomness
        connectionNovelty = Math.Max(0.1, Math.Min(0.9, connectionNovelty + (0.2 * (_random.NextDouble() - 0.5))));

        return connectionNovelty;
    }

    /// <summary>
    /// Gets distant connections
    /// </summary>
    /// <param name="count">The number of connections to return</param>
    /// <returns>The distant connections</returns>
    public List<ConceptConnection> GetDistantConnections(int count)
    {
        return _connections
            .Where(c => c.Type == ConnectionType.Distant)
            .OrderByDescending(c => c.CreationTimestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most novel connections
    /// </summary>
    /// <param name="count">The number of connections to return</param>
    /// <returns>The most novel connections</returns>
    public List<ConceptConnection> GetMostNovelConnections(int count)
    {
        return _connections
            .Where(c => c.Type == ConnectionType.Distant)
            .OrderByDescending(c => c.Attributes.TryGetValue("novelty", out var novelty) ? novelty : 0.0)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the strongest connections
    /// </summary>
    /// <param name="count">The number of connections to return</param>
    /// <returns>The strongest connections</returns>
    public List<ConceptConnection> GetStrongestConnections(int count)
    {
        return _connections
            .OrderByDescending(c => c.Strength)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets connections for a concept
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="count">The number of connections to return</param>
    /// <returns>The connections for the concept</returns>
    public List<ConceptConnection> GetConnectionsForConcept(string concept, int count)
    {
        return _connections
            .Where(c => c.Concept1 == concept || c.Concept2 == concept)
            .OrderByDescending(c => c.Strength)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Adds a concept to the semantic network
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="attributes">The attributes</param>
    /// <returns>The created concept node</returns>
    public ConceptNode AddConcept(string concept, Dictionary<string, double>? attributes = null)
    {
        if (_semanticNetwork.ContainsKey(concept))
        {
            return _semanticNetwork[concept];
        }

        // Create concept node
        var node = new ConceptNode
        {
            Id = Guid.NewGuid().ToString(),
            Name = concept,
            Attributes = attributes ?? GenerateConceptAttributes(concept)
        };

        // Add to semantic network
        _semanticNetwork[concept] = node;

        _logger.LogInformation("Added concept to semantic network: {Concept}", concept);

        return node;
    }
}
