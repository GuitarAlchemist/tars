using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.Workflow;

namespace TarsCli.Services.SelfCoding
{
    /// <summary>
    /// Service for processing files in the self-coding workflow
    /// </summary>
    public class FileProcessor
    {
        private readonly ILogger<FileProcessor> _logger;
        private readonly TaskPrioritizer _taskPrioritizer;

        /// <summary>
        /// Initializes a new instance of the FileProcessor class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="taskPrioritizer">Task prioritizer</param>
        public FileProcessor(ILogger<FileProcessor> logger, TaskPrioritizer taskPrioritizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _taskPrioritizer = taskPrioritizer ?? throw new ArgumentNullException(nameof(taskPrioritizer));
        }

        /// <summary>
        /// Selects files for improvement
        /// </summary>
        /// <param name="targetDirectory">Target directory</param>
        /// <param name="filePatterns">File patterns to match</param>
        /// <param name="maxFiles">Maximum number of files to select</param>
        /// <param name="excludePatterns">Patterns to exclude</param>
        /// <returns>List of selected files</returns>
        public async Task<List<string>> SelectFilesAsync(string targetDirectory, IEnumerable<string> filePatterns, int maxFiles = 10, IEnumerable<string> excludePatterns = null)
        {
            _logger.LogInformation($"Selecting files from {targetDirectory} with patterns: {string.Join(", ", filePatterns)}");

            try
            {
                // Validate parameters
                if (string.IsNullOrEmpty(targetDirectory))
                {
                    throw new ArgumentException("Target directory is required", nameof(targetDirectory));
                }

                if (filePatterns == null || !filePatterns.Any())
                {
                    throw new ArgumentException("File patterns are required", nameof(filePatterns));
                }

                if (!Directory.Exists(targetDirectory))
                {
                    throw new DirectoryNotFoundException($"Directory not found: {targetDirectory}");
                }

                // Find files matching the patterns
                var selectedFiles = new List<string>();
                foreach (var pattern in filePatterns)
                {
                    var files = Directory.GetFiles(targetDirectory, pattern, SearchOption.AllDirectories);
                    selectedFiles.AddRange(files);
                }

                // Remove duplicates
                selectedFiles = selectedFiles.Distinct().ToList();

                // Apply exclude patterns
                if (excludePatterns != null && excludePatterns.Any())
                {
                    foreach (var excludePattern in excludePatterns)
                    {
                        var excludeFiles = Directory.GetFiles(targetDirectory, excludePattern, SearchOption.AllDirectories);
                        selectedFiles = selectedFiles.Except(excludeFiles).ToList();
                    }
                }

                // Limit the number of files
                if (selectedFiles.Count > maxFiles)
                {
                    selectedFiles = selectedFiles.Take(maxFiles).ToList();
                }

                _logger.LogInformation($"Selected {selectedFiles.Count} files");
                return await Task.FromResult(selectedFiles);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error selecting files from {targetDirectory}");
                throw;
            }
        }

        /// <summary>
        /// Extracts file metadata
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <returns>File metadata</returns>
        public async Task<FileMetadata> ExtractFileMetadataAsync(string filePath)
        {
            _logger.LogInformation($"Extracting metadata for {filePath}");

            try
            {
                // Validate parameters
                if (string.IsNullOrEmpty(filePath))
                {
                    throw new ArgumentException("File path is required", nameof(filePath));
                }

                if (!File.Exists(filePath))
                {
                    throw new FileNotFoundException($"File not found: {filePath}");
                }

                // Get file info
                var fileInfo = new FileInfo(filePath);

                // Read the file content
                var fileContent = await File.ReadAllTextAsync(filePath);

                // Create the metadata
                var metadata = new FileMetadata
                {
                    FilePath = filePath,
                    FileName = fileInfo.Name,
                    FileExtension = fileInfo.Extension,
                    FileSize = fileInfo.Length,
                    CreationTime = fileInfo.CreationTime,
                    LastWriteTime = fileInfo.LastWriteTime,
                    LineCount = CountLines(fileContent),
                    CharacterCount = fileContent.Length
                };

                _logger.LogInformation($"Extracted metadata for {filePath}");
                return metadata;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error extracting metadata for {filePath}");
                throw;
            }
        }

        /// <summary>
        /// Prioritizes files for improvement
        /// </summary>
        /// <param name="files">List of files</param>
        /// <param name="analysisResults">Analysis results</param>
        /// <param name="maxFiles">Maximum number of files to prioritize</param>
        /// <returns>Prioritized list of files</returns>
        public async Task<List<string>> PrioritizeFilesAsync(List<string> files, List<CodeAnalysis.CodeAnalysisResult> analysisResults, int maxFiles = 10)
        {
            _logger.LogInformation($"Prioritizing {files.Count} files for improvement");

            try
            {
                // Validate parameters
                if (files == null || !files.Any())
                {
                    throw new ArgumentException("Files are required", nameof(files));
                }

                if (analysisResults == null)
                {
                    throw new ArgumentException("Analysis results are required", nameof(analysisResults));
                }

                // Prioritize files
                var prioritizedFiles = _taskPrioritizer.PrioritizeFiles(analysisResults, maxFiles);

                _logger.LogInformation($"Prioritized {prioritizedFiles.Count} files for improvement");
                return await Task.FromResult(prioritizedFiles);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error prioritizing files for improvement");
                throw;
            }
        }

        /// <summary>
        /// Counts the number of lines in a string
        /// </summary>
        /// <param name="content">String content</param>
        /// <returns>Number of lines</returns>
        private int CountLines(string content)
        {
            if (string.IsNullOrEmpty(content))
            {
                return 0;
            }

            return content.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None).Length;
        }
    }

    /// <summary>
    /// Metadata for a file
    /// </summary>
    public class FileMetadata
    {
        /// <summary>
        /// Path to the file
        /// </summary>
        public string FilePath { get; set; }

        /// <summary>
        /// Name of the file
        /// </summary>
        public string FileName { get; set; }

        /// <summary>
        /// Extension of the file
        /// </summary>
        public string FileExtension { get; set; }

        /// <summary>
        /// Size of the file in bytes
        /// </summary>
        public long FileSize { get; set; }

        /// <summary>
        /// Creation time of the file
        /// </summary>
        public DateTime CreationTime { get; set; }

        /// <summary>
        /// Last write time of the file
        /// </summary>
        public DateTime LastWriteTime { get; set; }

        /// <summary>
        /// Number of lines in the file
        /// </summary>
        public int LineCount { get; set; }

        /// <summary>
        /// Number of characters in the file
        /// </summary>
        public int CharacterCount { get; set; }

        /// <summary>
        /// Additional metadata
        /// </summary>
        public Dictionary<string, object> AdditionalMetadata { get; set; } = new Dictionary<string, object>();
    }
}
