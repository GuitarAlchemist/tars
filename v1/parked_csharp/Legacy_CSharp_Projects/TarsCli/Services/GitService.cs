using Microsoft.Extensions.Configuration;
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace TarsCli.Services;

/// <summary>
/// Service for Git integration
/// </summary>
public class GitService
{
    private readonly ILogger<GitService> _logger;
    private readonly string _repositoryPath;

    public GitService(
        ILogger<GitService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _repositoryPath = configuration["Tars:RepositoryPath"] ?? Directory.GetCurrentDirectory();
    }

    /// <summary>
    /// Get the commit history for a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="maxCommits">Maximum number of commits to return</param>
    /// <returns>List of commits</returns>
    public async Task<List<GitCommit>> GetFileHistoryAsync(string filePath, int maxCommits = 10)
    {
        _logger.LogInformation($"Getting commit history for: {Path.GetFullPath(filePath)}");

        try
        {
            // Run git log on the file
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = $"log --pretty=format:\"%H|%an|%ad|%s\" --date=iso -n {maxCommits} -- \"{filePath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _repositoryPath
                }
            };

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            process.OutputDataReceived += (sender, args) =>
            {
                if (args.Data != null)
                {
                    outputBuilder.AppendLine(args.Data);
                }
            };

            process.ErrorDataReceived += (sender, args) =>
            {
                if (args.Data != null)
                {
                    errorBuilder.AppendLine(args.Data);
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();

            var output = outputBuilder.ToString();
            var error = errorBuilder.ToString();

            if (process.ExitCode != 0)
            {
                _logger.LogWarning($"Git log failed: {error}");
                return new List<GitCommit>();
            }

            // Parse the commit history
            var commits = new List<GitCommit>();
            var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            foreach (var line in lines)
            {
                var parts = line.Split('|');
                if (parts.Length >= 4)
                {
                    var commit = new GitCommit
                    {
                        Hash = parts[0],
                        Author = parts[1],
                        Date = DateTime.Parse(parts[2]),
                        Message = parts[3]
                    };

                    commits.Add(commit);
                }
            }

            _logger.LogInformation($"Found {commits.Count} commits for {filePath}");
            return commits;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting commit history for {filePath}");
            return new List<GitCommit>();
        }
    }

    /// <summary>
    /// Commit changes to a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="message">Commit message</param>
    /// <returns>True if the commit succeeded, false otherwise</returns>
    public async Task<bool> CommitFileAsync(string filePath, string message)
    {
        _logger.LogInformation($"Committing changes to: {Path.GetFullPath(filePath)}");

        try
        {
            // Stage the file
            var stageProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = $"add \"{filePath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _repositoryPath
                }
            };

            stageProcess.Start();
            await stageProcess.WaitForExitAsync();

            if (stageProcess.ExitCode != 0)
            {
                _logger.LogWarning($"Git add failed for {filePath}");
                return false;
            }

            // Commit the file
            var commitProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = $"commit -m \"{message}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _repositoryPath
                }
            };

            commitProcess.Start();
            await commitProcess.WaitForExitAsync();

            if (commitProcess.ExitCode != 0)
            {
                _logger.LogWarning($"Git commit failed for {filePath}");
                return false;
            }

            _logger.LogInformation($"Successfully committed changes to {filePath}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error committing changes to {filePath}");
            return false;
        }
    }

    /// <summary>
    /// Create a branch
    /// </summary>
    /// <param name="branchName">Name of the branch</param>
    /// <returns>True if the branch was created, false otherwise</returns>
    public async Task<bool> CreateBranchAsync(string branchName)
    {
        _logger.LogInformation($"Creating branch: {branchName}");

        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = $"checkout -b {branchName}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _repositoryPath
                }
            };

            process.Start();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogWarning($"Git branch creation failed for {branchName}");
                return false;
            }

            _logger.LogInformation($"Successfully created branch {branchName}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error creating branch {branchName}");
            return false;
        }
    }

    /// <summary>
    /// Push changes to the remote repository
    /// </summary>
    /// <param name="branchName">Name of the branch to push</param>
    /// <returns>True if the push succeeded, false otherwise</returns>
    public async Task<bool> PushChangesAsync(string branchName)
    {
        _logger.LogInformation($"Pushing changes to branch: {branchName}");

        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = $"push -u origin {branchName}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _repositoryPath
                }
            };

            process.Start();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogWarning($"Git push failed for branch {branchName}");
                return false;
            }

            _logger.LogInformation($"Successfully pushed changes to branch {branchName}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error pushing changes to branch {branchName}");
            return false;
        }
    }

    /// <summary>
    /// Create a pull request
    /// </summary>
    /// <param name="branchName">Name of the branch</param>
    /// <param name="title">Title of the pull request</param>
    /// <param name="description">Description of the pull request</param>
    /// <returns>URL of the pull request if successful, null otherwise</returns>
    public async Task<string> CreatePullRequestAsync(string branchName, string title, string description)
    {
        _logger.LogInformation($"Creating pull request for branch: {branchName}");

        try
        {
            // Get the remote URL
            var remoteUrlProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = "config --get remote.origin.url",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _repositoryPath
                }
            };

            remoteUrlProcess.Start();
            var remoteUrl = await remoteUrlProcess.StandardOutput.ReadToEndAsync();
            await remoteUrlProcess.WaitForExitAsync();

            // Parse the remote URL to get the owner and repo
            var match = Regex.Match(remoteUrl, @"github\.com[:/]([^/]+)/([^/\.]+)");
            if (!match.Success)
            {
                _logger.LogWarning($"Could not parse remote URL: {remoteUrl}");
                return null;
            }

            var owner = match.Groups[1].Value;
            var repo = match.Groups[2].Value.EndsWith(".git") ? match.Groups[2].Value.Substring(0, match.Groups[2].Value.Length - 4) : match.Groups[2].Value;

            // Create the pull request using the GitHub CLI
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "gh",
                    Arguments = $"pr create --title \"{title}\" --body \"{description}\" --base main --head {branchName}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _repositoryPath
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                _logger.LogWarning($"GitHub CLI pull request creation failed for branch {branchName}");
                return null;
            }

            // Extract the pull request URL from the output
            var urlMatch = Regex.Match(output, @"https://github\.com/[^\s]+");
            if (!urlMatch.Success)
            {
                _logger.LogWarning($"Could not extract pull request URL from output: {output}");
                return null;
            }

            var pullRequestUrl = urlMatch.Value;
            _logger.LogInformation($"Successfully created pull request: {pullRequestUrl}");
            return pullRequestUrl;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error creating pull request for branch {branchName}");
            return null;
        }
    }
}

/// <summary>
/// Git commit information
/// </summary>
public class GitCommit
{
    /// <summary>
    /// Commit hash
    /// </summary>
    public string Hash { get; set; }

    /// <summary>
    /// Commit author
    /// </summary>
    public string Author { get; set; }

    /// <summary>
    /// Commit date
    /// </summary>
    public DateTime Date { get; set; }

    /// <summary>
    /// Commit message
    /// </summary>
    public string Message { get; set; }
}
