using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TarsCli.Models
{
    /// <summary>
    /// Represents the state of the autonomous improvement process
    /// </summary>
    public class AutoImprovementState
    {
        /// <summary>
        /// The list of files pending processing
        /// </summary>
        public List<string> PendingFiles { get; set; } = new List<string>();

        /// <summary>
        /// The list of files that have been processed
        /// </summary>
        public List<string> ProcessedFiles { get; set; } = new List<string>();

        /// <summary>
        /// The list of files that have been improved
        /// </summary>
        public List<string> ImprovedFiles { get; set; } = new List<string>();

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
        /// The file priority scores
        /// </summary>
        [JsonIgnore] // Don't serialize this property as it can be recalculated
        public Dictionary<string, FilePriorityScore> FilePriorityScores { get; set; } = new Dictionary<string, FilePriorityScore>();

        /// <summary>
        /// The last time the state was updated
        /// </summary>
        public DateTime LastUpdated { get; set; } = DateTime.Now;

        /// <summary>
        /// The history of improvements made
        /// </summary>
        public List<ImprovementRecord> ImprovementHistory { get; set; } = new List<ImprovementRecord>();
    }

    /// <summary>
    /// Represents a record of an improvement made to a file
    /// </summary>
    public class ImprovementRecord
    {
        /// <summary>
        /// The path of the file that was improved
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// The time the improvement was made
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.Now;

        /// <summary>
        /// A description of the improvement
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// The score of the file before improvement
        /// </summary>
        public double ScoreBefore { get; set; }

        /// <summary>
        /// The score of the file after improvement
        /// </summary>
        public double ScoreAfter { get; set; }

        /// <summary>
        /// The model used for the improvement
        /// </summary>
        public string Model { get; set; } = string.Empty;
    }
}
