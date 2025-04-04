using Microsoft.Extensions.Configuration;
using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for managing and organizing tutorial content
/// </summary>
public class TutorialOrganizerService
{
    private readonly ILogger<TutorialOrganizerService> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _tutorialsDirectory;
    private readonly string _tutorialCatalogPath;

    public TutorialOrganizerService(
        ILogger<TutorialOrganizerService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;

        // Set up the tutorials directory
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        if (string.IsNullOrEmpty(appDataPath))
        {
            appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars");
        }
        else
        {
            appDataPath = Path.Combine(appDataPath, "TARS");
        }

        _tutorialsDirectory = Path.Combine(appDataPath, "Tutorials");
        _tutorialCatalogPath = Path.Combine(_tutorialsDirectory, "catalog.json");

        if (!Directory.Exists(_tutorialsDirectory))
        {
            Directory.CreateDirectory(_tutorialsDirectory);
        }

        // Initialize the catalog if it doesn't exist
        if (!File.Exists(_tutorialCatalogPath))
        {
            var catalog = new TutorialCatalog
            {
                Categories = new List<TutorialCategory>(),
                LastUpdated = DateTime.UtcNow
            };

            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };

            File.WriteAllText(_tutorialCatalogPath, JsonSerializer.Serialize(catalog, options));
        }
    }

    /// <summary>
    /// Adds a new tutorial to the catalog
    /// </summary>
    public async Task<Tutorial> AddTutorial(
        string title,
        string description,
        string content,
        string category,
        DifficultyLevel difficultyLevel,
        List<string> tags,
        List<string>? prerequisites = null)
    {
        _logger.LogInformation($"Adding tutorial: {title} in category: {category}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Create a new tutorial
            var tutorial = new Tutorial
            {
                Id = Guid.NewGuid().ToString(),
                Title = title,
                Description = description,
                Category = category,
                DifficultyLevel = difficultyLevel,
                Tags = tags ?? new List<string>(),
                Prerequisites = prerequisites ?? new List<string>(),
                CreatedDate = DateTime.UtcNow,
                LastModifiedDate = DateTime.UtcNow
            };

            // Save the tutorial content to a file
            var tutorialFilePath = Path.Combine(_tutorialsDirectory, $"{tutorial.Id}.md");
            await File.WriteAllTextAsync(tutorialFilePath, content);

            // Add the tutorial to the catalog
            var existingCategory = catalog.Categories.FirstOrDefault(c => c.Name.Equals(category, StringComparison.OrdinalIgnoreCase));

            if (existingCategory == null)
            {
                // Create a new category if it doesn't exist
                existingCategory = new TutorialCategory
                {
                    Name = category,
                    Tutorials = new List<Tutorial>()
                };

                catalog.Categories.Add(existingCategory);
            }

            existingCategory.Tutorials.Add(tutorial);

            // Update the catalog
            catalog.LastUpdated = DateTime.UtcNow;
            await SaveCatalog(catalog);

            return tutorial;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error adding tutorial: {title}");
            throw;
        }
    }

    /// <summary>
    /// Gets all tutorials
    /// </summary>
    /// <returns>List of all tutorials</returns>
    public async Task<List<Tutorial>> GetTutorials()
    {
        _logger.LogInformation("Getting all tutorials");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Collect all tutorials from all categories
            var allTutorials = new List<Tutorial>();

            foreach (var category in catalog.Categories)
            {
                allTutorials.AddRange(category.Tutorials);
            }

            return allTutorials;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting all tutorials");
            return new List<Tutorial>();
        }
    }

    /// <summary>
    /// Gets the content of a tutorial
    /// </summary>
    /// <param name="id">Tutorial ID</param>
    /// <returns>Tutorial content</returns>
    public async Task<string> GetTutorialContent(string id)
    {
        _logger.LogInformation($"Getting tutorial content: {id}");

        try
        {
            var result = await GetTutorial(id);

            // Use pattern matching to check if result is not null
            if (result is TutorialWithContent tutorial)
            {
                return tutorial.Content;
            }

            return "Tutorial content not found";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting tutorial content: {id}");
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Categorize tutorials
    /// </summary>
    /// <param name="ids">List of tutorial IDs</param>
    /// <param name="category">Category name</param>
    /// <returns>True if successful</returns>
    public async Task<bool> CategorizeTutorials(List<string> ids, string category)
    {
        _logger.LogInformation($"Categorizing tutorials: {string.Join(", ", ids)} to {category}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Find the category or create it if it doesn't exist
            var targetCategory = catalog.Categories.FirstOrDefault(c => c.Name.Equals(category, StringComparison.OrdinalIgnoreCase));

            if (targetCategory == null)
            {
                targetCategory = new TutorialCategory
                {
                    Name = category,
                    Tutorials = new List<Tutorial>()
                };

                catalog.Categories.Add(targetCategory);
            }

            // Find the tutorials and move them to the target category
            foreach (var id in ids)
            {
                Tutorial? tutorial = null;
                TutorialCategory? sourceCategory = null;

                // Find the tutorial and its source category
                foreach (var cat in catalog.Categories)
                {
                    var t = cat.Tutorials.FirstOrDefault(t => t.Id == id);

                    if (t != null)
                    {
                        tutorial = t;
                        sourceCategory = cat;
                        break;
                    }
                }

                if (tutorial != null && sourceCategory != null)
                {
                    // Remove from source category
                    sourceCategory.Tutorials.Remove(tutorial);

                    // Update the category property
                    tutorial.Category = category;

                    // Add to target category
                    targetCategory.Tutorials.Add(tutorial);

                    _logger.LogInformation($"Moved tutorial {id} from {sourceCategory.Name} to {category}");
                }
                else
                {
                    _logger.LogWarning($"Tutorial {id} not found");
                }
            }

            // Save the catalog
            await SaveCatalog(catalog);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error categorizing tutorials");
            return false;
        }
    }

    /// <summary>
    /// Gets a tutorial by ID with its content
    /// </summary>
    /// <returns>A TutorialWithContent object or null if not found</returns>
    public async Task<TutorialWithContent?> GetTutorial(string id)
    {
        _logger.LogInformation($"Getting tutorial: {id}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Find the tutorial using LINQ and pattern matching
            var tutorial = catalog.Categories
                .SelectMany(c => c.Tutorials)
                .FirstOrDefault(t => t.Id == id);

            if (tutorial is null)
            {
                _logger.LogError($"Tutorial not found: {id}");
                return null;
            }

            // Load the tutorial content
            var tutorialFilePath = Path.Combine(_tutorialsDirectory, $"{tutorial.Id}.md");

            if (!File.Exists(tutorialFilePath))
            {
                _logger.LogError($"Tutorial content file not found: {tutorialFilePath}");
                return null;
            }

            var content = await File.ReadAllTextAsync(tutorialFilePath);

            // Return the tutorial with content using the record constructor
            return new TutorialWithContent(tutorial, content);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting tutorial: {id}");
            return null;
        }
    }

    /// <summary>
    /// Updates an existing tutorial
    /// </summary>
    public async Task<Tutorial> UpdateTutorial(
        string id,
        string? title = null,
        string? description = null,
        string? content = null,
        string? category = null,
        DifficultyLevel? difficultyLevel = null,
        List<string>? tags = null,
        List<string>? prerequisites = null)
    {
        _logger.LogInformation($"Updating tutorial: {id}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Find the tutorial and its category
            Tutorial tutorial = null;
            TutorialCategory tutorialCategory = null;

            foreach (var cat in catalog.Categories)
            {
                var foundTutorial = cat.Tutorials.FirstOrDefault(t => t.Id == id);

                if (foundTutorial != null)
                {
                    tutorial = foundTutorial;
                    tutorialCategory = cat;
                    break;
                }
            }

            if (tutorial == null)
            {
                _logger.LogError($"Tutorial not found: {id}");
                throw new FileNotFoundException($"Tutorial not found: {id}");
            }

            // Update the tutorial properties if provided
            if (!string.IsNullOrEmpty(title))
            {
                tutorial.Title = title;
            }

            if (!string.IsNullOrEmpty(description))
            {
                tutorial.Description = description;
            }

            if (difficultyLevel.HasValue)
            {
                tutorial.DifficultyLevel = difficultyLevel.Value;
            }

            if (tags != null)
            {
                tutorial.Tags = tags;
            }

            if (prerequisites != null)
            {
                tutorial.Prerequisites = prerequisites;
            }

            // Update the tutorial content if provided
            if (!string.IsNullOrEmpty(content))
            {
                var tutorialFilePath = Path.Combine(_tutorialsDirectory, $"{tutorial.Id}.md");
                await File.WriteAllTextAsync(tutorialFilePath, content);
            }

            // Move the tutorial to a different category if specified
            if (!string.IsNullOrEmpty(category) && !category.Equals(tutorialCategory.Name, StringComparison.OrdinalIgnoreCase))
            {
                // Remove the tutorial from the current category
                tutorialCategory.Tutorials.Remove(tutorial);

                // Find or create the new category
                var newCategory = catalog.Categories.FirstOrDefault(c => c.Name.Equals(category, StringComparison.OrdinalIgnoreCase));

                if (newCategory == null)
                {
                    // Create a new category if it doesn't exist
                    newCategory = new TutorialCategory
                    {
                        Name = category,
                        Tutorials = new List<Tutorial>()
                    };

                    catalog.Categories.Add(newCategory);
                }

                // Add the tutorial to the new category
                tutorial.Category = category;
                newCategory.Tutorials.Add(tutorial);

                // Remove empty categories
                catalog.Categories.RemoveAll(c => c.Tutorials.Count == 0);
            }

            // Update the last modified date
            tutorial.LastModifiedDate = DateTime.UtcNow;

            // Update the catalog
            catalog.LastUpdated = DateTime.UtcNow;
            await SaveCatalog(catalog);

            return tutorial;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error updating tutorial: {id}");
            throw;
        }
    }

    /// <summary>
    /// Deletes a tutorial
    /// </summary>
    public async Task DeleteTutorial(string id)
    {
        _logger.LogInformation($"Deleting tutorial: {id}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Find the tutorial and its category
            Tutorial tutorial = null;
            TutorialCategory tutorialCategory = null;

            foreach (var category in catalog.Categories)
            {
                var foundTutorial = category.Tutorials.FirstOrDefault(t => t.Id == id);

                if (foundTutorial != null)
                {
                    tutorial = foundTutorial;
                    tutorialCategory = category;
                    break;
                }
            }

            if (tutorial == null)
            {
                _logger.LogError($"Tutorial not found: {id}");
                throw new FileNotFoundException($"Tutorial not found: {id}");
            }

            // Remove the tutorial from the category
            tutorialCategory.Tutorials.Remove(tutorial);

            // Remove empty categories
            catalog.Categories.RemoveAll(c => c.Tutorials.Count == 0);

            // Delete the tutorial content file
            var tutorialFilePath = Path.Combine(_tutorialsDirectory, $"{tutorial.Id}.md");

            if (File.Exists(tutorialFilePath))
            {
                File.Delete(tutorialFilePath);
            }

            // Update the catalog
            catalog.LastUpdated = DateTime.UtcNow;
            await SaveCatalog(catalog);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error deleting tutorial: {id}");
            throw;
        }
    }

    /// <summary>
    /// Gets all tutorials
    /// </summary>
    public async Task<List<Tutorial>> GetAllTutorials()
    {
        _logger.LogInformation("Getting all tutorials");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Collect all tutorials from all categories
            var tutorials = new List<Tutorial>();

            foreach (var category in catalog.Categories)
            {
                tutorials.AddRange(category.Tutorials);
            }

            return tutorials.OrderBy(t => t.Category).ThenBy(t => t.Title).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting all tutorials");
            throw;
        }
    }

    /// <summary>
    /// Gets all tutorials in a specific category
    /// </summary>
    public async Task<List<Tutorial>> GetTutorialsByCategory(string category)
    {
        _logger.LogInformation($"Getting tutorials in category: {category}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Find the category
            var tutorialCategory = catalog.Categories.FirstOrDefault(c => c.Name.Equals(category, StringComparison.OrdinalIgnoreCase));

            if (tutorialCategory == null)
            {
                return new List<Tutorial>();
            }

            return tutorialCategory.Tutorials.OrderBy(t => t.Title).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting tutorials in category: {category}");
            throw;
        }
    }

    /// <summary>
    /// Gets all tutorials with a specific tag
    /// </summary>
    public async Task<List<Tutorial>> GetTutorialsByTag(string tag)
    {
        _logger.LogInformation($"Getting tutorials with tag: {tag}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Collect all tutorials with the specified tag
            var tutorials = new List<Tutorial>();

            foreach (var category in catalog.Categories)
            {
                tutorials.AddRange(category.Tutorials.Where(t => t.Tags.Any(tag => tag.Equals(tag, StringComparison.OrdinalIgnoreCase))));
            }

            return tutorials.OrderBy(t => t.Category).ThenBy(t => t.Title).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting tutorials with tag: {tag}");
            throw;
        }
    }

    /// <summary>
    /// Gets all tutorials with a specific difficulty level
    /// </summary>
    public async Task<List<Tutorial>> GetTutorialsByDifficulty(DifficultyLevel difficultyLevel)
    {
        _logger.LogInformation($"Getting tutorials with difficulty level: {difficultyLevel}");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Collect all tutorials with the specified difficulty level
            var tutorials = new List<Tutorial>();

            foreach (var category in catalog.Categories)
            {
                tutorials.AddRange(category.Tutorials.Where(t => t.DifficultyLevel == difficultyLevel));
            }

            return tutorials.OrderBy(t => t.Category).ThenBy(t => t.Title).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting tutorials with difficulty level: {difficultyLevel}");
            throw;
        }
    }

    /// <summary>
    /// Gets all categories
    /// </summary>
    public async Task<List<string>> GetAllCategories()
    {
        _logger.LogInformation("Getting all categories");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            return catalog.Categories.Select(c => c.Name).OrderBy(n => n).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting all categories");
            throw;
        }
    }

    /// <summary>
    /// Gets all tags
    /// </summary>
    public async Task<List<string>> GetAllTags()
    {
        _logger.LogInformation("Getting all tags");

        try
        {
            // Load the catalog
            var catalog = await LoadCatalog();

            // Collect all unique tags
            var tags = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var category in catalog.Categories)
            {
                foreach (var tutorial in category.Tutorials)
                {
                    foreach (var tag in tutorial.Tags)
                    {
                        tags.Add(tag);
                    }
                }
            }

            return tags.OrderBy(t => t).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting all tags");
            throw;
        }
    }

    /// <summary>
    /// Searches for tutorials matching the specified criteria
    /// </summary>
    public async Task<List<Tutorial>> SearchTutorials(
        string? searchTerm = null,
        string? category = null,
        DifficultyLevel? difficultyLevel = null,
        List<string>? tags = null,
        List<string>? prerequisites = null)
    {
        _logger.LogInformation($"Searching tutorials with term: {searchTerm}, category: {category}, difficulty: {difficultyLevel}");

        try
        {
            // Get all tutorials
            var allTutorials = await GetAllTutorials();

            // Filter by search term
            if (!string.IsNullOrEmpty(searchTerm))
            {
                allTutorials = allTutorials.Where(t =>
                    t.Title.Contains(searchTerm, StringComparison.OrdinalIgnoreCase) ||
                    t.Description.Contains(searchTerm, StringComparison.OrdinalIgnoreCase) ||
                    t.Tags.Any(tag => tag.Contains(searchTerm, StringComparison.OrdinalIgnoreCase))
                ).ToList();
            }

            // Filter by category
            if (!string.IsNullOrEmpty(category))
            {
                allTutorials = allTutorials.Where(t => t.Category.Equals(category, StringComparison.OrdinalIgnoreCase)).ToList();
            }

            // Filter by difficulty level
            if (difficultyLevel.HasValue)
            {
                allTutorials = allTutorials.Where(t => t.DifficultyLevel == difficultyLevel.Value).ToList();
            }

            // Filter by tags
            if (tags != null && tags.Count > 0)
            {
                allTutorials = allTutorials.Where(t =>
                    tags.All(tag => t.Tags.Any(tutorialTag => tutorialTag.Equals(tag, StringComparison.OrdinalIgnoreCase)))
                ).ToList();
            }

            // Filter by prerequisites
            if (prerequisites != null && prerequisites.Count > 0)
            {
                allTutorials = allTutorials.Where(t =>
                    prerequisites.All(prereq => t.Prerequisites.Any(tutorialPrereq => tutorialPrereq.Equals(prereq, StringComparison.OrdinalIgnoreCase)))
                ).ToList();
            }

            return allTutorials;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error searching tutorials");
            throw;
        }
    }

    /// <summary>
    /// Loads the tutorial catalog from disk
    /// </summary>
    private async Task<TutorialCatalog> LoadCatalog()
    {
        try
        {
            if (File.Exists(_tutorialCatalogPath))
            {
                var json = await File.ReadAllTextAsync(_tutorialCatalogPath);
                return JsonSerializer.Deserialize<TutorialCatalog>(json);
            }
            else
            {
                // Create a new catalog if it doesn't exist
                var catalog = new TutorialCatalog
                {
                    Categories = new List<TutorialCategory>(),
                    LastUpdated = DateTime.UtcNow
                };

                await SaveCatalog(catalog);

                return catalog;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading tutorial catalog");
            throw;
        }
    }

    /// <summary>
    /// Saves the tutorial catalog to disk
    /// </summary>
    private async Task SaveCatalog(TutorialCatalog catalog)
    {
        try
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };

            var json = JsonSerializer.Serialize(catalog, options);
            await File.WriteAllTextAsync(_tutorialCatalogPath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving tutorial catalog");
            throw;
        }
    }
}

/// <summary>
/// Represents a tutorial catalog
/// </summary>
public class TutorialCatalog
{
    public List<TutorialCategory> Categories { get; set; } = new List<TutorialCategory>();
    public DateTime LastUpdated { get; set; }
}

/// <summary>
/// Represents a tutorial category
/// </summary>
public class TutorialCategory
{
    public required string Name { get; set; }
    public List<Tutorial> Tutorials { get; set; } = new List<Tutorial>();
}

/// <summary>
/// Represents a tutorial
/// </summary>
public class Tutorial
{
    public required string Id { get; set; }
    public required string Title { get; set; }
    public required string Description { get; set; }
    public required string Category { get; set; }
    public DifficultyLevel DifficultyLevel { get; set; }
    public List<string> Tags { get; set; } = new List<string>();
    public List<string> Prerequisites { get; set; } = new List<string>();
    public DateTime CreatedDate { get; set; }
    public DateTime LastModifiedDate { get; set; }
}

/// <summary>
/// Represents a tutorial with its content
/// </summary>
public record TutorialWithContent(Tutorial Tutorial, string Content);