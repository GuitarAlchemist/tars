using System.Reflection;
using TarsEngine.Models.Unified;
using ModelComplexityType = TarsEngine.Models.Metrics.ComplexityType;
using InterfaceComplexityType = TarsEngine.Services.Interfaces.ComplexityType;

namespace TarsEngine.Utilities
{
    /// <summary>
    /// Provides conversion methods between different ComplexityType enums
    /// </summary>
    public static class ComplexityTypeConverter
    {
        // Check if the enum values exist using reflection
        private static readonly bool _modelHasMaintainability;
        private static readonly bool _modelHasMaintainabilityIndex;
        private static readonly bool _interfaceHasMaintainability;
        private static readonly bool _interfaceHasMaintainabilityIndex;

        static ComplexityTypeConverter()
        {
            // Check if the enum values exist in the model enum
            _modelHasMaintainability = Enum.IsDefined(typeof(ModelComplexityType), "Maintainability");
            _modelHasMaintainabilityIndex = Enum.IsDefined(typeof(ModelComplexityType), "MaintainabilityIndex");

            // Check if the enum values exist in the interface enum
            _interfaceHasMaintainability = Enum.IsDefined(typeof(InterfaceComplexityType), "Maintainability");
            _interfaceHasMaintainabilityIndex = Enum.IsDefined(typeof(InterfaceComplexityType), "MaintainabilityIndex");
        }

        /// <summary>
        /// Converts from Models.Metrics.ComplexityType to ComplexityTypeUnified
        /// </summary>
        public static ComplexityTypeUnified ToUnified(ModelComplexityType complexityType)
        {
            // Use reflection to safely convert the enum value
            string enumName = Enum.GetName(typeof(ModelComplexityType), complexityType) ?? "Other";

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
        /// Converts from Services.Interfaces.ComplexityType to ComplexityTypeUnified
        /// </summary>
        public static ComplexityTypeUnified ToUnified(InterfaceComplexityType complexityType)
        {
            // Use reflection to safely convert the enum value
            string enumName = Enum.GetName(typeof(InterfaceComplexityType), complexityType) ?? "Cyclomatic";

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
        /// Converts from ComplexityTypeUnified to Models.Metrics.ComplexityType
        /// </summary>
        public static ModelComplexityType ToModelType(ComplexityTypeUnified complexityType)
        {
            return complexityType switch
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

        /// <summary>
        /// Converts from ComplexityTypeUnified to Services.Interfaces.ComplexityType
        /// </summary>
        public static InterfaceComplexityType ToInterfaceType(ComplexityTypeUnified complexityType)
        {
            try
            {
                return complexityType switch
                {
                    ComplexityTypeUnified.Cyclomatic => InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.Cognitive => InterfaceComplexityType.Cognitive,
                    ComplexityTypeUnified.Halstead => InterfaceComplexityType.Halstead,
                    ComplexityTypeUnified.Maintainability => _interfaceHasMaintainability ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "Maintainability") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.MaintainabilityIndex => _interfaceHasMaintainabilityIndex ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "MaintainabilityIndex") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.MethodLength => Enum.IsDefined(typeof(InterfaceComplexityType), "MethodLength") ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "MethodLength") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.ClassLength => Enum.IsDefined(typeof(InterfaceComplexityType), "ClassLength") ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "ClassLength") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.ParameterCount => Enum.IsDefined(typeof(InterfaceComplexityType), "ParameterCount") ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "ParameterCount") : InterfaceComplexityType.Cyclomatic,
                    ComplexityTypeUnified.NestingDepth => Enum.IsDefined(typeof(InterfaceComplexityType), "NestingDepth") ? (InterfaceComplexityType)Enum.Parse(typeof(InterfaceComplexityType), "NestingDepth") : InterfaceComplexityType.Cyclomatic,
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
        /// Direct conversion from Models.Metrics.ComplexityType to Services.Interfaces.ComplexityType
        /// </summary>
        public static InterfaceComplexityType ToInterfaceType(ModelComplexityType complexityType)
        {
            return ToInterfaceType(ToUnified(complexityType));
        }

        /// <summary>
        /// Direct conversion from Services.Interfaces.ComplexityType to Models.Metrics.ComplexityType
        /// </summary>
        public static ModelComplexityType ToModelType(InterfaceComplexityType complexityType)
        {
            return ToModelType(ToUnified(complexityType));
        }
    }
}
