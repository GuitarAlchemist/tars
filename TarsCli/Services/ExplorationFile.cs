using System;
using System.IO;

namespace TarsCli.Services
{
    /// <summary>
    /// Represents an exploration file
    /// </summary>
    public class ExplorationFile
    {
        /// <summary>
        /// The file path of the exploration
        /// </summary>
        public string FilePath { get; set; }
        
        /// <summary>
        /// The file name of the exploration
        /// </summary>
        public string FileName => Path.GetFileName(FilePath);
        
        /// <summary>
        /// The title of the exploration (file name without extension and prefix)
        /// </summary>
        public string Title
        {
            get
            {
                var fileName = Path.GetFileNameWithoutExtension(FilePath);
                
                // Remove common prefixes
                if (fileName.StartsWith("ChatGPT-"))
                {
                    fileName = fileName.Substring(8);
                }
                else if (fileName.StartsWith("DeepThinking-"))
                {
                    fileName = fileName.Substring(13);
                }
                
                return fileName;
            }
        }
        
        /// <summary>
        /// The version of the exploration (e.g., v1, v2, etc.)
        /// </summary>
        public string Version
        {
            get
            {
                var directory = Path.GetDirectoryName(FilePath);
                if (directory == null)
                {
                    return string.Empty;
                }
                
                var parts = directory.Split(Path.DirectorySeparatorChar);
                foreach (var part in parts)
                {
                    if (part.StartsWith("v") && part.Length > 1 && int.TryParse(part.Substring(1), out _))
                    {
                        return part;
                    }
                }
                
                return string.Empty;
            }
        }
        
        /// <summary>
        /// The creation date of the exploration
        /// </summary>
        public DateTime CreationDate => File.GetCreationTime(FilePath);
        
        /// <summary>
        /// The last write date of the exploration
        /// </summary>
        public DateTime LastWriteDate => File.GetLastWriteTime(FilePath);
        
        /// <summary>
        /// Constructor
        /// </summary>
        public ExplorationFile(string filePath)
        {
            FilePath = filePath;
        }
        
        /// <summary>
        /// Returns a string representation of the exploration file
        /// </summary>
        public override string ToString()
        {
            return $"{Title} ({Version})";
        }
    }
}
