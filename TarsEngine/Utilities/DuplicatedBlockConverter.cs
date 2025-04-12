using System.Linq;
using TarsEngine.Models.Unified;
using ModelDuplicatedBlock = TarsEngine.Models.Metrics.DuplicatedBlock;
using InterfaceDuplicatedBlock = TarsEngine.Services.Interfaces.DuplicatedBlock;

namespace TarsEngine.Utilities
{
    /// <summary>
    /// Provides conversion methods between different DuplicatedBlock classes
    /// </summary>
    public static class DuplicatedBlockConverter
    {
        /// <summary>
        /// Converts from Models.Metrics.DuplicatedBlock to DuplicatedBlockUnified
        /// </summary>
        public static DuplicatedBlockUnified ToUnified(ModelDuplicatedBlock block)
        {
            if (block == null)
            {
                return new DuplicatedBlockUnified();
            }

            return new DuplicatedBlockUnified
            {
                SourceFilePath = block.SourceFilePath,
                SourceStartLine = block.SourceStartLine,
                SourceEndLine = block.SourceEndLine,
                TargetFilePath = block.TargetFilePath,
                TargetStartLine = block.TargetStartLine,
                TargetEndLine = block.TargetEndLine,
                DuplicatedLines = block.DuplicatedLines,
                SimilarityPercentage = block.SimilarityPercentage,
                DuplicatedCode = block.DuplicatedCode
            };
        }

        /// <summary>
        /// Converts from Services.Interfaces.DuplicatedBlock to DuplicatedBlockUnified
        /// </summary>
        public static DuplicatedBlockUnified ToUnified(InterfaceDuplicatedBlock block)
        {
            if (block == null)
            {
                return new DuplicatedBlockUnified();
            }

            // Get start and end lines from the interface block
            int sourceStartLine = 0;
            int sourceEndLine = 0;
            int targetStartLine = 0;
            int targetEndLine = 0;
            string sourceMethod = string.Empty;
            string targetMethod = string.Empty;

            // Check if the properties exist using reflection
            var type = block.GetType();
            var sourceStartProp = type.GetProperty("SourceStartLine");
            var sourceEndProp = type.GetProperty("SourceEndLine");
            var targetStartProp = type.GetProperty("TargetStartLine");
            var targetEndProp = type.GetProperty("TargetEndLine");
            var sourceMethodProp = type.GetProperty("SourceMethod");
            var targetMethodProp = type.GetProperty("TargetMethod");

            if (sourceStartProp != null) sourceStartLine = (int)sourceStartProp.GetValue(block)!;
            if (sourceEndProp != null) sourceEndLine = (int)sourceEndProp.GetValue(block)!;
            if (targetStartProp != null) targetStartLine = (int)targetStartProp.GetValue(block)!;
            if (targetEndProp != null) targetEndLine = (int)targetEndProp.GetValue(block)!;
            if (sourceMethodProp != null) sourceMethod = (string)sourceMethodProp.GetValue(block)!;
            if (targetMethodProp != null) targetMethod = (string)targetMethodProp.GetValue(block)!;

            // Calculate duplicated lines
            int duplicatedLines = targetEndLine - targetStartLine + 1;
            if (duplicatedLines < 0) duplicatedLines = 0;

            return new DuplicatedBlockUnified
            {
                SourceFilePath = block.FilePath,
                SourceStartLine = sourceStartLine,
                SourceEndLine = sourceEndLine,
                TargetFilePath = block.FilePath,
                TargetStartLine = targetStartLine,
                TargetEndLine = targetEndLine,
                SourceMethod = sourceMethod,
                TargetMethod = targetMethod,
                DuplicatedCode = block.Content,
                DuplicatedLines = duplicatedLines,
                SimilarityPercentage = 100.0
            };
        }

        /// <summary>
        /// Converts from DuplicatedBlockUnified to Models.Metrics.DuplicatedBlock
        /// </summary>
        public static ModelDuplicatedBlock ToModelBlock(DuplicatedBlockUnified block)
        {
            if (block == null)
            {
                return new ModelDuplicatedBlock();
            }

            return new ModelDuplicatedBlock
            {
                SourceFilePath = block.SourceFilePath,
                SourceStartLine = block.SourceStartLine,
                SourceEndLine = block.SourceEndLine,
                TargetFilePath = block.TargetFilePath,
                TargetStartLine = block.TargetStartLine,
                TargetEndLine = block.TargetEndLine,
                SimilarityPercentage = block.SimilarityPercentage,
                DuplicatedCode = block.DuplicatedCode
                // Note: DuplicatedLines is a calculated property in ModelDuplicatedBlock
            };
        }

        /// <summary>
        /// Converts from DuplicatedBlockUnified to Services.Interfaces.DuplicatedBlock
        /// </summary>
        public static InterfaceDuplicatedBlock ToInterfaceBlock(DuplicatedBlockUnified block)
        {
            if (block == null)
            {
                return new InterfaceDuplicatedBlock();
            }

            var result = new InterfaceDuplicatedBlock
            {
                FilePath = block.SourceFilePath,
                Content = block.DuplicatedCode
            };

            // Check if the properties exist using reflection
            var type = result.GetType();
            var sourceStartProp = type.GetProperty("SourceStartLine");
            var sourceEndProp = type.GetProperty("SourceEndLine");
            var targetStartProp = type.GetProperty("TargetStartLine");
            var targetEndProp = type.GetProperty("TargetEndLine");
            var sourceMethodProp = type.GetProperty("SourceMethod");
            var targetMethodProp = type.GetProperty("TargetMethod");

            if (sourceStartProp != null) sourceStartProp.SetValue(result, block.SourceStartLine);
            if (sourceEndProp != null) sourceEndProp.SetValue(result, block.SourceEndLine);
            if (targetStartProp != null) targetStartProp.SetValue(result, block.TargetStartLine);
            if (targetEndProp != null) targetEndProp.SetValue(result, block.TargetEndLine);
            if (sourceMethodProp != null) sourceMethodProp.SetValue(result, block.SourceMethod);
            if (targetMethodProp != null) targetMethodProp.SetValue(result, block.TargetMethod);

            return result;
        }

        /// <summary>
        /// Direct conversion from Models.Metrics.DuplicatedBlock to Services.Interfaces.DuplicatedBlock
        /// </summary>
        public static InterfaceDuplicatedBlock ToInterfaceBlock(ModelDuplicatedBlock block)
        {
            return ToInterfaceBlock(ToUnified(block));
        }

        /// <summary>
        /// Direct conversion from Services.Interfaces.DuplicatedBlock to Models.Metrics.DuplicatedBlock
        /// </summary>
        public static ModelDuplicatedBlock ToModelBlock(InterfaceDuplicatedBlock block)
        {
            return ToModelBlock(ToUnified(block));
        }
    }
}
