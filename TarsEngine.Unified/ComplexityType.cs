using System;

namespace TarsEngine.Unified
{
    /// <summary>
    /// Represents a complexity type
    /// </summary>
    public enum ComplexityType
    {
        /// <summary>
        /// Cyclomatic complexity
        /// </summary>
        Cyclomatic,

        /// <summary>
        /// Cognitive complexity
        /// </summary>
        Cognitive,

        /// <summary>
        /// Halstead complexity
        /// </summary>
        Halstead,

        /// <summary>
        /// Maintainability index
        /// </summary>
        Maintainability,

        /// <summary>
        /// Maintainability index (alternative name)
        /// </summary>
        MaintainabilityIndex,

        /// <summary>
        /// Method length
        /// </summary>
        MethodLength,

        /// <summary>
        /// Class length
        /// </summary>
        ClassLength,

        /// <summary>
        /// Parameter count
        /// </summary>
        ParameterCount,

        /// <summary>
        /// Nesting depth
        /// </summary>
        NestingDepth,

        /// <summary>
        /// Structural complexity
        /// </summary>
        Structural,

        /// <summary>
        /// Algorithmic complexity
        /// </summary>
        Algorithmic,

        /// <summary>
        /// Other complexity
        /// </summary>
        Other
    }
}
