using System;
using ModelComplexityType = TarsEngine.Models.Metrics.ComplexityType;
using InterfaceComplexityType = TarsEngine.Services.Interfaces.ComplexityType;
using TarsEngine.Models.Unified;

namespace TarsEngine.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between TarsEngine.Services.Interfaces.ComplexityType and TarsEngine.Models.Metrics.ComplexityType
    /// </summary>
    public static class ComplexityTypeAdapter
    {
        // Check if the enum values exist using reflection
        private static readonly bool _modelHasMaintainability;
        private static readonly bool _modelHasMaintainabilityIndex;
        private static readonly bool _interfaceHasMaintainability;
        private static readonly bool _interfaceHasMaintainabilityIndex;
        private static readonly bool _interfaceHasMethodLength;
        private static readonly bool _interfaceHasClassLength;
        private static readonly bool _interfaceHasParameterCount;
        private static readonly bool _interfaceHasNestingDepth;

        static ComplexityTypeAdapter()
        {
            // Check if the enum values exist in the model enum
            _modelHasMaintainability = Enum.IsDefined(typeof(ModelComplexityType), "Maintainability");
            _modelHasMaintainabilityIndex = Enum.IsDefined(typeof(ModelComplexityType), "MaintainabilityIndex");

            // Check if the enum values exist in the interface enum
            _interfaceHasMaintainability = Enum.IsDefined(typeof(InterfaceComplexityType), "Maintainability");
            _interfaceHasMaintainabilityIndex = Enum.IsDefined(typeof(InterfaceComplexityType), "MaintainabilityIndex");
            _interfaceHasMethodLength = Enum.IsDefined(typeof(InterfaceComplexityType), "MethodLength");
            _interfaceHasClassLength = Enum.IsDefined(typeof(InterfaceComplexityType), "ClassLength");
            _interfaceHasParameterCount = Enum.IsDefined(typeof(InterfaceComplexityType), "ParameterCount");
            _interfaceHasNestingDepth = Enum.IsDefined(typeof(InterfaceComplexityType), "NestingDepth");
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Interfaces.ComplexityType to TarsEngine.Models.Metrics.ComplexityType
        /// </summary>
        /// <param name="interfaceType">The interface type to convert</param>
        /// <returns>The model type</returns>
        public static ModelComplexityType ToModelType(InterfaceComplexityType interfaceType)
        {
            // Use reflection to safely convert the enum value
            string enumName = Enum.GetName(typeof(InterfaceComplexityType), interfaceType) ?? "Cyclomatic";

            return enumName switch
            {
                "Cyclomatic" => ModelComplexityType.Cyclomatic,
                "Cognitive" => ModelComplexityType.Cognitive,
                "Halstead" => ModelComplexityType.Halstead,
                "Maintainability" => _modelHasMaintainability ? ModelComplexityType.Maintainability : ModelComplexityType.Other,
                "MaintainabilityIndex" => _modelHasMaintainabilityIndex ? (ModelComplexityType)Enum.Parse(typeof(ModelComplexityType), "MaintainabilityIndex") : (_modelHasMaintainability ? ModelComplexityType.Maintainability : ModelComplexityType.Other),
                "MethodLength" => ModelComplexityType.Other, // No direct mapping
                "ClassLength" => ModelComplexityType.Other, // No direct mapping
                "ParameterCount" => ModelComplexityType.Other, // No direct mapping
                "NestingDepth" => ModelComplexityType.Other, // No direct mapping
                _ => ModelComplexityType.Other
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Models.Metrics.ComplexityType to TarsEngine.Services.Interfaces.ComplexityType
        /// </summary>
        /// <param name="modelType">The model type to convert</param>
        /// <returns>The interface type</returns>
        public static InterfaceComplexityType ToInterfaceType(ModelComplexityType modelType)
        {
            try
            {
                // Use reflection to safely convert the enum value
                string enumName = Enum.GetName(typeof(ModelComplexityType), modelType) ?? "Cyclomatic";

                return enumName switch
                {
                    "Cyclomatic" => InterfaceComplexityType.Cyclomatic,
                    "Cognitive" => InterfaceComplexityType.Cognitive,
                    "Halstead" => InterfaceComplexityType.Halstead,
                    "Maintainability" => _interfaceHasMaintainability ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "Maintainability") : InterfaceComplexityType.Cyclomatic,
                    "MaintainabilityIndex" => _interfaceHasMaintainabilityIndex ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "MaintainabilityIndex") : (_interfaceHasMaintainability ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "Maintainability") : InterfaceComplexityType.Cyclomatic),
                    "Structural" => InterfaceComplexityType.Cyclomatic, // No direct mapping
                    "Algorithmic" => InterfaceComplexityType.Cyclomatic, // No direct mapping
                    "Other" => InterfaceComplexityType.Cyclomatic, // No direct mapping
                    _ => InterfaceComplexityType.Cyclomatic // Default to Cyclomatic as a fallback
                };
            }
            catch
            {
                // If any error occurs, return the default value
                return InterfaceComplexityType.Cyclomatic;
            }
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Interfaces.ComplexityType to ComplexityTypeUnified
        /// </summary>
        /// <param name="interfaceType">The interface type to convert</param>
        /// <returns>The unified type</returns>
        public static ComplexityTypeUnified ToUnified(InterfaceComplexityType interfaceType)
        {
            // Use reflection to safely convert the enum value
            string enumName = Enum.GetName(typeof(InterfaceComplexityType), interfaceType) ?? "Cyclomatic";

            return enumName switch
            {
                "Cyclomatic" => ComplexityTypeUnified.Cyclomatic,
                "Cognitive" => ComplexityTypeUnified.Cognitive,
                "Halstead" => ComplexityTypeUnified.Halstead,
                "Maintainability" => ComplexityTypeUnified.Maintainability,
                "MaintainabilityIndex" => ComplexityTypeUnified.MaintainabilityIndex,
                "MethodLength" => ComplexityTypeUnified.MethodLength,
                "ClassLength" => ComplexityTypeUnified.ClassLength,
                "ParameterCount" => ComplexityTypeUnified.ParameterCount,
                "NestingDepth" => ComplexityTypeUnified.NestingDepth,
                _ => ComplexityTypeUnified.Other
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Models.Metrics.ComplexityType to ComplexityTypeUnified
        /// </summary>
        /// <param name="modelType">The model type to convert</param>
        /// <returns>The unified type</returns>
        public static ComplexityTypeUnified ToUnified(ModelComplexityType modelType)
        {
            // Use reflection to safely convert the enum value
            string enumName = Enum.GetName(typeof(ModelComplexityType), modelType) ?? "Other";

            return enumName switch
            {
                "Cyclomatic" => ComplexityTypeUnified.Cyclomatic,
                "Cognitive" => ComplexityTypeUnified.Cognitive,
                "Halstead" => ComplexityTypeUnified.Halstead,
                "Maintainability" => ComplexityTypeUnified.Maintainability,
                "MaintainabilityIndex" => ComplexityTypeUnified.MaintainabilityIndex,
                "Structural" => ComplexityTypeUnified.Structural,
                "Algorithmic" => ComplexityTypeUnified.Algorithmic,
                _ => ComplexityTypeUnified.Other
            };
        }

        /// <summary>
        /// Converts from ComplexityTypeUnified to TarsEngine.Services.Interfaces.ComplexityType
        /// </summary>
        /// <param name="unifiedType">The unified type to convert</param>
        /// <returns>The interface type</returns>
        public static InterfaceComplexityType FromUnified(ComplexityTypeUnified unifiedType)
        {
            try
            {
                return unifiedType switch
                {
                    ComplexityTypeUnified.Cyclomatic => InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.Cognitive => InterfaceComplexityType.Cognitive,
                    ComplexityTypeUnified.Halstead => InterfaceComplexityType.Halstead,
                    ComplexityTypeUnified.Maintainability => _interfaceHasMaintainability ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "Maintainability") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.MaintainabilityIndex => _interfaceHasMaintainabilityIndex ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "MaintainabilityIndex") : (_interfaceHasMaintainability ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "Maintainability") : InterfaceComplexityType.Cyclomatic),
                    ComplexityTypeUnified.MethodLength => _interfaceHasMethodLength ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "MethodLength") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.ClassLength => _interfaceHasClassLength ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "ClassLength") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.ParameterCount => _interfaceHasParameterCount ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "ParameterCount") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.NestingDepth => _interfaceHasNestingDepth ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "NestingDepth") : InterfaceComplexityType.Cyclomatic,
                    _ => InterfaceComplexityType.Cyclomatic // Default to Cyclomatic as a fallback
                };
            }
            catch
            {
                // If any error occurs, return the default value
                return InterfaceComplexityType.Cyclomatic;
            }
        }

        /// <summary>
        /// Converts from ComplexityTypeUnified to TarsEngine.Models.Metrics.ComplexityType
        /// </summary>
        /// <param name="unifiedType">The unified type to convert</param>
        /// <returns>The model type</returns>
        public static ModelComplexityType ModelFromUnified(ComplexityTypeUnified unifiedType)
        {
            return unifiedType switch
            {
                ComplexityTypeUnified.Cyclomatic => ModelComplexityType.Cyclomatic,
                ComplexityTypeUnified.Cognitive => ModelComplexityType.Cognitive,
                ComplexityTypeUnified.Halstead => ModelComplexityType.Halstead,
                ComplexityTypeUnified.Maintainability => _modelHasMaintainability ? ModelComplexityType.Maintainability : ModelComplexityType.Other,
                ComplexityTypeUnified.MaintainabilityIndex => _modelHasMaintainabilityIndex ? (ModelComplexityType)Enum.Parse(typeof(ModelComplexityType), "MaintainabilityIndex") : (_modelHasMaintainability ? ModelComplexityType.Maintainability : ModelComplexityType.Other),
                ComplexityTypeUnified.Structural => ModelComplexityType.Structural,
                ComplexityTypeUnified.Algorithmic => ModelComplexityType.Algorithmic,
                _ => ModelComplexityType.Other
            };
        }
    }
}
