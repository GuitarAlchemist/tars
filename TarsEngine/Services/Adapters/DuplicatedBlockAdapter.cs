using System.Collections.Generic;
using System.Linq;
using ModelDuplicatedBlock = TarsEngine.Models.Metrics.DuplicatedBlock;
using InterfaceDuplicatedBlock = TarsEngine.Services.Interfaces.DuplicatedBlock;
using ModelDuplicateLocation = TarsEngine.Models.Unified.DuplicateLocation;
using InterfaceDuplicateLocation = TarsEngine.Services.Interfaces.DuplicateLocation;

namespace TarsEngine.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between TarsEngine.Models.Metrics.DuplicatedBlock and TarsEngine.Services.Interfaces.DuplicatedBlock
    /// </summary>
    public static class DuplicatedBlockAdapter
    {
        /// <summary>
        /// Converts from TarsEngine.Models.Metrics.DuplicatedBlock to TarsEngine.Services.Interfaces.DuplicatedBlock
        /// </summary>
        /// <param name="modelBlock">The model block to convert</param>
        /// <returns>The interface block</returns>
        public static InterfaceDuplicatedBlock ToInterfaceBlock(ModelDuplicatedBlock modelBlock)
        {
            if (modelBlock == null)
            {
                return new InterfaceDuplicatedBlock();
            }

            var result = new InterfaceDuplicatedBlock
            {
                FilePath = modelBlock.SourceFilePath,
                StartLine = modelBlock.SourceStartLine,
                EndLine = modelBlock.SourceEndLine,
                Content = modelBlock.DuplicatedCode
            };

            // Add duplicate location for the target
            if (!string.IsNullOrEmpty(modelBlock.TargetFilePath))
            {
                result.DuplicateLocations.Add(new InterfaceDuplicateLocation
                {
                    FilePath = modelBlock.TargetFilePath,
                    StartLine = modelBlock.TargetStartLine,
                    EndLine = modelBlock.TargetEndLine
                });
            }

            return result;
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Interfaces.DuplicatedBlock to TarsEngine.Models.Metrics.DuplicatedBlock
        /// </summary>
        /// <param name="interfaceBlock">The interface block to convert</param>
        /// <returns>The model block</returns>
        public static ModelDuplicatedBlock ToModelBlock(InterfaceDuplicatedBlock interfaceBlock)
        {
            if (interfaceBlock == null)
            {
                return new ModelDuplicatedBlock();
            }

            var result = new ModelDuplicatedBlock
            {
                SourceFilePath = interfaceBlock.FilePath,
                SourceStartLine = interfaceBlock.StartLine,
                SourceEndLine = interfaceBlock.EndLine,
                DuplicatedCode = interfaceBlock.Content
            };

            // Get the first duplicate location as the target
            if (interfaceBlock.DuplicateLocations.Any())
            {
                var firstLocation = interfaceBlock.DuplicateLocations.First();
                result.TargetFilePath = firstLocation.FilePath;
                result.TargetStartLine = firstLocation.StartLine;
                result.TargetEndLine = firstLocation.EndLine;
            }
            else
            {
                // If no duplicate locations, use the source as the target
                result.TargetFilePath = interfaceBlock.FilePath;
                result.TargetStartLine = interfaceBlock.StartLine;
                result.TargetEndLine = interfaceBlock.EndLine;
            }

            // Set similarity to 100% by default
            result.SimilarityPercentage = 100.0;

            return result;
        }

        /// <summary>
        /// Converts a list of TarsEngine.Models.Metrics.DuplicatedBlock to a list of TarsEngine.Services.Interfaces.DuplicatedBlock
        /// </summary>
        /// <param name="modelBlocks">The model blocks to convert</param>
        /// <returns>The interface blocks</returns>
        public static List<InterfaceDuplicatedBlock> ToInterfaceBlocks(IEnumerable<ModelDuplicatedBlock> modelBlocks)
        {
            return modelBlocks?.Select(ToInterfaceBlock).ToList() ?? new List<InterfaceDuplicatedBlock>();
        }

        /// <summary>
        /// Converts a list of TarsEngine.Services.Interfaces.DuplicatedBlock to a list of TarsEngine.Models.Metrics.DuplicatedBlock
        /// </summary>
        /// <param name="interfaceBlocks">The interface blocks to convert</param>
        /// <returns>The model blocks</returns>
        public static List<ModelDuplicatedBlock> ToModelBlocks(IEnumerable<InterfaceDuplicatedBlock> interfaceBlocks)
        {
            return interfaceBlocks?.Select(ToModelBlock).ToList() ?? new List<ModelDuplicatedBlock>();
        }
    }
}
