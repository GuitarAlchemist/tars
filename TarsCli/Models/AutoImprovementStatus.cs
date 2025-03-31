using System;
using System.Collections.Generic;

namespace TarsCli.Models
{
    /// <summary>
    /// Represents the status of the autonomous improvement process
    /// </summary>
    public class AutoImprovementStatus
    {
        /// <summary>
        /// Whether the autonomous improvement process is running
        /// </summary>
        public bool IsRunning { get; set; }

        /// <summary>
        /// The time the autonomous improvement process started
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// The time limit for the autonomous improvement process
        /// </summary>
        public TimeSpan TimeLimit { get; set; }

        /// <summary>
        /// The elapsed time since the autonomous improvement process started
        /// </summary>
        public TimeSpan ElapsedTime { get; set; }

        /// <summary>
        /// The remaining time until the autonomous improvement process stops
        /// </summary>
        public TimeSpan RemainingTime { get; set; }

        /// <summary>
        /// The number of files that have been processed
        /// </summary>
        public int FilesProcessed { get; set; }

        /// <summary>
        /// The number of files remaining to be processed
        /// </summary>
        public int FilesRemaining { get; set; }

        /// <summary>
        /// The current file being processed
        /// </summary>
        public string? CurrentFile { get; set; }

        /// <summary>
        /// The last file that was improved
        /// </summary>
        public string? LastImprovedFile { get; set; }

        /// <summary>
        /// The total number of improvements made
        /// </summary>
        public int TotalImprovements { get; set; }

        /// <summary>
        /// The top priority files (up to 5)
        /// </summary>
        public List<FilePriorityInfo> TopPriorityFiles { get; set; } = new List<FilePriorityInfo>();

        /// <summary>
        /// The recent improvements (up to 5)
        /// </summary>
        public List<ImprovementInfo> RecentImprovements { get; set; } = new List<ImprovementInfo>();
    }

    /// <summary>
    /// Represents information about a file's priority
    /// </summary>
    public class FilePriorityInfo
    {
        /// <summary>
        /// The path of the file
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// The priority score of the file
        /// </summary>
        public double Score { get; set; }

        /// <summary>
        /// A description of why the file has this priority
        /// </summary>
        public string Reason { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents information about an improvement
    /// </summary>
    public class ImprovementInfo
    {
        /// <summary>
        /// The path of the file that was improved
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// The time the improvement was made
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// A description of the improvement
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// The score improvement
        /// </summary>
        public double ScoreImprovement { get; set; }
    }
}
