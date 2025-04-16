using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence.Insight;

/// <summary>
/// Implements connection discovery capabilities for insight generation
/// </summary>
public class ConnectionDiscovery
{
    private readonly ILogger<ConnectionDiscovery> _logger;
    private readonly System.Random _random = new();
    private double _connectionDiscoveryLevel = 0.5; // Starting with moderate connection discovery
    private readonly Dictionary<string, List<string>> _conceptRelations = new();
    private readonly List<DiscoveredConnection> _discoveredConnections = [];

    /// <summary>
    /// Gets the connection discovery level (0.0 to 1.0)
    /// </summary>
    public double ConnectionDiscoveryLevel => _connectionDiscoveryLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConnectionDiscovery"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ConnectionDiscovery(ILogger<ConnectionDiscovery> logger)
    {
        _logger = logger;
        InitializeConceptRelations();
    }

    /// <summary>
    /// Initializes the concept relations
    /// </summary>
    private void InitializeConceptRelations()
    {
        // Add programming concepts and relations
        AddConceptRelation("Algorithm", ["Efficiency", "Complexity", "Optimization", "Problem Solving", "Data Structure"
        ]);
        AddConceptRelation("Data Structure", ["Algorithm", "Memory", "Organization", "Efficiency", "Access Pattern"]);
        AddConceptRelation("Design Pattern", ["Architecture", "Reusability", "Abstraction", "Modularity", "Problem Solving"
        ]);
        AddConceptRelation("Functional Programming", ["Immutability", "Higher-Order Functions", "Recursion", "Purity", "Type System"
        ]);
        AddConceptRelation("Object-Oriented Programming", ["Inheritance", "Encapsulation", "Polymorphism", "Class", "Interface"
        ]);

        // Add AI concepts and relations
        AddConceptRelation("Neural Network", ["Deep Learning", "Weights", "Activation Function", "Backpropagation", "Layer"
        ]);
        AddConceptRelation("Machine Learning", ["Training", "Model", "Feature", "Prediction", "Dataset"]);
        AddConceptRelation("Natural Language Processing", ["Tokenization", "Embedding", "Semantics", "Parsing", "Generation"
        ]);
        AddConceptRelation("Reinforcement Learning", ["Agent", "Environment", "Reward", "Policy", "State"]);
        AddConceptRelation("Computer Vision", ["Image Processing", "Recognition", "Feature Extraction", "Convolution", "Segmentation"
        ]);

        // Add philosophical concepts and relations
        AddConceptRelation("Consciousness", ["Awareness", "Experience", "Qualia", "Self", "Mind"]);
        AddConceptRelation("Intelligence", ["Problem Solving", "Learning", "Adaptation", "Knowledge", "Reasoning"]);
        AddConceptRelation("Emergence", ["Complexity", "System", "Property", "Interaction", "Unpredictability"]);
        AddConceptRelation("Epistemology", ["Knowledge", "Belief", "Justification", "Truth", "Skepticism"]);
        AddConceptRelation("Ethics", ["Morality", "Value", "Principle", "Virtue", "Consequence"]);

        // Add scientific concepts and relations
        AddConceptRelation("Quantum Mechanics", ["Superposition", "Entanglement", "Wave Function", "Measurement", "Uncertainty"
        ]);
        AddConceptRelation("Complexity Theory", ["Emergence", "Chaos", "Self-Organization", "Network", "Adaptation"]);
        AddConceptRelation("Information Theory", ["Entropy", "Compression", "Channel", "Redundancy", "Noise"]);
        AddConceptRelation("Systems Theory", ["Feedback", "Homeostasis", "Boundary", "Hierarchy", "Emergence"]);
        AddConceptRelation("Evolution", ["Selection", "Adaptation", "Fitness", "Mutation", "Diversity"]);

        // Add cross-domain relations
        AddConceptRelation("Algorithm", ["Problem Solving", "Intelligence", "Evolution"]);
        AddConceptRelation("Neural Network", ["Brain", "Learning", "Adaptation", "Consciousness"]);
        AddConceptRelation("Emergence", ["Complexity Theory", "Systems Theory", "Consciousness", "Evolution"]);
        AddConceptRelation("Information Theory", ["Communication", "Knowledge", "Entropy", "Compression"]);
        AddConceptRelation("Quantum Mechanics", ["Uncertainty", "Probability", "Measurement", "Observer"]);

        _logger.LogInformation("Initialized concept relations for {ConceptCount} concepts", _conceptRelations.Count);
    }

    /// <summary>
    /// Updates the connection discovery level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase connection discovery level over time (very slowly)
            if (_connectionDiscoveryLevel < 0.95)
            {
                _connectionDiscoveryLevel += 0.0001 * _random.NextDouble();
                _connectionDiscoveryLevel = Math.Min(_connectionDiscoveryLevel, 1.0);
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
    /// Adds a concept relation
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="relatedConcepts">The related concepts</param>
    public void AddConceptRelation(string concept, string[] relatedConcepts)
    {
        if (!_conceptRelations.ContainsKey(concept))
        {
            _conceptRelations[concept] = [];
        }

        foreach (var relatedConcept in relatedConcepts)
        {
            if (!_conceptRelations[concept].Contains(relatedConcept))
            {
                _conceptRelations[concept].Add(relatedConcept);
            }

            // Add reverse relation
            if (!_conceptRelations.ContainsKey(relatedConcept))
            {
                _conceptRelations[relatedConcept] = [];
            }

            if (!_conceptRelations[relatedConcept].Contains(concept))
            {
                _conceptRelations[relatedConcept].Add(concept);
            }
        }

        _logger.LogDebug("Added concept relation: {Concept} -> {RelatedConcepts}",
            concept, string.Join(", ", relatedConcepts));
    }

    /// <summary>
    /// Gets related concepts
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The related concepts</returns>
    public List<string> GetRelatedConcepts(string concept)
    {
        if (_conceptRelations.TryGetValue(concept, out var relatedConcepts))
        {
            return relatedConcepts;
        }

        return [];
    }

    /// <summary>
    /// Discovers connections between concepts
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <param name="maxDepth">The maximum search depth</param>
    /// <returns>The discovered connections</returns>
    public List<DiscoveredConnection> DiscoverConnections(string concept1, string concept2, int maxDepth = 3)
    {
        var connections = new List<DiscoveredConnection>();

        try
        {
            _logger.LogDebug("Discovering connections between {Concept1} and {Concept2} with max depth {MaxDepth}",
                concept1, concept2, maxDepth);

            // Check if concepts exist
            if (!_conceptRelations.ContainsKey(concept1) || !_conceptRelations.ContainsKey(concept2))
            {
                _logger.LogWarning("One or both concepts not found in relations: {Concept1}, {Concept2}", concept1, concept2);
                return connections;
            }

            // Check for direct connection
            if (_conceptRelations[concept1].Contains(concept2))
            {
                var directConnection = new DiscoveredConnection
                {
                    Concept1 = concept1,
                    Concept2 = concept2,
                    Path = [concept1, concept2],
                    Strength = 1.0,
                    Description = $"Direct connection between {concept1} and {concept2}",
                    Timestamp = DateTime.UtcNow
                };

                connections.Add(directConnection);

                // Record the discovered connection
                _discoveredConnections.Add(directConnection);

                return connections;
            }

            // Perform breadth-first search for connections
            var paths = FindPaths(concept1, concept2, maxDepth);

            foreach (var path in paths)
            {
                // Calculate connection strength based on path length
                var strength = 1.0 / path.Count;

                // Apply connection discovery level
                strength *= _connectionDiscoveryLevel;

                var connection = new DiscoveredConnection
                {
                    Concept1 = concept1,
                    Concept2 = concept2,
                    Path = path,
                    Strength = strength,
                    Description = GenerateConnectionDescription(path),
                    Timestamp = DateTime.UtcNow
                };

                connections.Add(connection);

                // Record the discovered connection
                _discoveredConnections.Add(connection);
            }

            _logger.LogInformation("Discovered {ConnectionCount} connections between {Concept1} and {Concept2}",
                connections.Count, concept1, concept2);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error discovering connections");
        }

        return connections;
    }

    /// <summary>
    /// Finds paths between concepts
    /// </summary>
    /// <param name="start">The start concept</param>
    /// <param name="end">The end concept</param>
    /// <param name="maxDepth">The maximum depth</param>
    /// <returns>The paths</returns>
    private List<List<string>> FindPaths(string start, string end, int maxDepth)
    {
        var paths = new List<List<string>>();
        var visited = new HashSet<string>();
        var queue = new Queue<(string Node, List<string> Path, int Depth)>();

        // Start with the initial node
        queue.Enqueue((start, [start], 0));

        while (queue.Count > 0)
        {
            var (node, path, depth) = queue.Dequeue();

            // Skip if we've reached max depth
            if (depth >= maxDepth)
            {
                continue;
            }

            // Get related concepts
            var relatedConcepts = GetRelatedConcepts(node);

            foreach (var relatedConcept in relatedConcepts)
            {
                // Skip if already in path (avoid cycles)
                if (path.Contains(relatedConcept))
                {
                    continue;
                }

                // Create new path
                var newPath = new List<string>(path) { relatedConcept };

                // Check if we've reached the end
                if (relatedConcept == end)
                {
                    paths.Add(newPath);
                    continue;
                }

                // Add to queue for further exploration
                queue.Enqueue((relatedConcept, newPath, depth + 1));
            }
        }

        return paths;
    }

    /// <summary>
    /// Generates a connection description
    /// </summary>
    /// <param name="path">The connection path</param>
    /// <returns>The description</returns>
    private string GenerateConnectionDescription(List<string> path)
    {
        if (path.Count <= 2)
        {
            return $"Direct connection between {path[0]} and {path[1]}";
        }

        var intermediates = string.Join(" → ", path.Skip(1).Take(path.Count - 2));
        return $"Connection between {path[0]} and {path[path.Count - 1]} via {intermediates}";
    }

    /// <summary>
    /// Discovers unexpected connections
    /// </summary>
    /// <param name="count">The number of connections to discover</param>
    /// <returns>The discovered connections</returns>
    public List<DiscoveredConnection> DiscoverUnexpectedConnections(int count)
    {
        var connections = new List<DiscoveredConnection>();

        try
        {
            _logger.LogDebug("Discovering {Count} unexpected connections", count);

            // Get all concepts
            var concepts = _conceptRelations.Keys.ToList();

            // Try to find unexpected connections
            var attempts = 0;
            while (connections.Count < count && attempts < count * 3)
            {
                // Choose random concepts
                var concept1 = concepts[_random.Next(concepts.Count)];
                var concept2 = concepts[_random.Next(concepts.Count)];

                // Skip if same concept
                if (concept1 == concept2)
                {
                    attempts++;
                    continue;
                }

                // Skip if directly connected
                if (_conceptRelations[concept1].Contains(concept2))
                {
                    attempts++;
                    continue;
                }

                // Discover connections
                var discoveredConnections = DiscoverConnections(concept1, concept2, 2);

                // Add unexpected connections
                foreach (var connection in discoveredConnections)
                {
                    // Only add if path length is at least 3 (start, intermediate, end)
                    if (connection.Path.Count >= 3)
                    {
                        connections.Add(connection);

                        if (connections.Count >= count)
                        {
                            break;
                        }
                    }
                }

                attempts++;
            }

            _logger.LogInformation("Discovered {ConnectionCount} unexpected connections", connections.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error discovering unexpected connections");
        }

        return connections;
    }

    /// <summary>
    /// Generates an insight based on a connection
    /// </summary>
    /// <param name="connection">The connection</param>
    /// <returns>The generated insight</returns>
    public InsightModel GenerateConnectionInsight(DiscoveredConnection connection)
    {
        try
        {
            _logger.LogDebug("Generating insight for connection: {Description}", connection.Description);

            // Generate insight templates
            var insightTemplates = new List<string>
            {
                "I've discovered an interesting connection between {0} and {1} through {2}. This suggests that {3}.",
                "The link between {0} and {1} via {2} reveals a pattern that might indicate {3}.",
                "By connecting {0} to {1} through {2}, we can see that {3}.",
                "The pathway from {0} to {1} through {2} suggests a principle: {3}.",
                "I've realized that {0} and {1} are connected through {2}, which implies {3}."
            };

            // Choose a random template
            var template = insightTemplates[_random.Next(insightTemplates.Count)];

            // Generate implication
            var implication = GenerateImplication(connection);

            // Format intermediate concepts
            var intermediates = string.Join(" and ", connection.Path.Skip(1).Take(connection.Path.Count - 2));
            if (string.IsNullOrEmpty(intermediates))
            {
                intermediates = "direct association";
            }

            // Generate insight content
            var content = string.Format(template, connection.Concept1, connection.Concept2, intermediates, implication);

            // Calculate significance based on connection strength and unexpectedness
            var unexpectedness = 1.0 - connection.Strength;
            var significance = (0.3 + (0.7 * unexpectedness)) * _connectionDiscoveryLevel;

            // Create insight
            var insight = new InsightModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = content,
                Type = InsightType.ConnectionDiscovery,
                Significance = significance,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>
                {
                    { "Connection", connection },
                    { "Concept1", connection.Concept1 },
                    { "Concept2", connection.Concept2 },
                    { "Path", connection.Path },
                    { "Strength", connection.Strength }
                },
                Tags = [..connection.Path, "connection", "insight"],
                Source = "ConnectionDiscovery"
            };

            _logger.LogInformation("Generated connection insight: {Content} (Significance: {Significance:F2})",
                insight.Content, insight.Significance);

            return insight;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating connection insight");

            // Return basic insight
            return new InsightModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = $"I've noticed a connection between {connection.Concept1} and {connection.Concept2}",
                Type = InsightType.ConnectionDiscovery,
                Significance = 0.3,
                Timestamp = DateTime.UtcNow,
                Source = "ConnectionDiscovery"
            };
        }
    }

    /// <summary>
    /// Generates an implication for a connection
    /// </summary>
    /// <param name="connection">The connection</param>
    /// <returns>The implication</returns>
    private string GenerateImplication(DiscoveredConnection connection)
    {
        // Generate implication templates based on concepts
        var implicationTemplates = new Dictionary<string, List<string>>
        {
            ["Algorithm"] =
            [
                "algorithmic approaches might be applicable in unexpected domains",
                "computational thinking can provide insights into diverse problems",
                "the structure of algorithms reflects deeper patterns in problem-solving",
                "algorithmic efficiency concepts might transfer to other processes"
            ],

            ["Neural Network"] =
            [
                "brain-inspired computing models might have broader applications",
                "learning systems can emerge from simple connected components",
                "distributed processing principles apply across different domains",
                "adaptive networks might be a universal pattern in complex systems"
            ],

            ["Consciousness"] =
            [
                "awareness might emerge from simpler interconnected processes",
                "self-reference could be a key aspect of complex systems",
                "subjective experience might have computational analogues",
                "the boundary between conscious and unconscious processing is blurry"
            ],

            ["Emergence"] =
            [
                "complex behaviors can arise from simple rules and interactions",
                "higher-level properties aren't always predictable from lower-level components",
                "similar emergent patterns appear across vastly different systems",
                "self-organization principles might be universal"
            ],

            ["Quantum Mechanics"] =
            [
                "uncertainty and probability are fundamental to reality",
                "observer effects might be more widespread than we realize",
                "complementary perspectives can both be valid simultaneously",
                "entanglement-like connections might exist in other systems"
            ]
        };

        // Check if we have templates for either concept
        List<string> templates = [];

        if (implicationTemplates.ContainsKey(connection.Concept1))
        {
            templates.AddRange(implicationTemplates[connection.Concept1]);
        }

        if (implicationTemplates.ContainsKey(connection.Concept2))
        {
            templates.AddRange(implicationTemplates[connection.Concept2]);
        }

        // Add generic templates if needed
        if (templates.Count == 0)
        {
            templates.AddRange([
                "there might be underlying principles connecting seemingly disparate domains",
                "cross-disciplinary insights can lead to unexpected breakthroughs",
                "conceptual boundaries are often more fluid than we assume",
                "similar patterns can emerge in different contexts",
                "knowledge transfer between domains can yield novel perspectives"
            ]);
        }

        // Choose a random template
        return templates[_random.Next(templates.Count)];
    }

    /// <summary>
    /// Gets recent discovered connections
    /// </summary>
    /// <param name="count">The number of connections to return</param>
    /// <returns>The recent discovered connections</returns>
    public List<DiscoveredConnection> GetRecentDiscoveredConnections(int count)
    {
        return _discoveredConnections
            .OrderByDescending(c => c.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the strongest discovered connections
    /// </summary>
    /// <param name="count">The number of connections to return</param>
    /// <returns>The strongest discovered connections</returns>
    public List<DiscoveredConnection> GetStrongestDiscoveredConnections(int count)
    {
        return _discoveredConnections
            .OrderByDescending(c => c.Strength)
            .Take(count)
            .ToList();
    }
}

/// <summary>
/// Represents a discovered connection
/// </summary>
public class DiscoveredConnection
{
    /// <summary>
    /// Gets or sets the first concept
    /// </summary>
    public string Concept1 { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the second concept
    /// </summary>
    public string Concept2 { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the connection path
    /// </summary>
    public List<string> Path { get; set; } = [];

    /// <summary>
    /// Gets or sets the connection strength (0.0 to 1.0)
    /// </summary>
    public double Strength { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the connection description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}
