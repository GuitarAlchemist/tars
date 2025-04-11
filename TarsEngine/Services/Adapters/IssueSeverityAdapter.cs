using ModelIssueSeverity = TarsEngine.Models.IssueSeverity;
using InterfaceIssueSeverity = TarsEngine.Services.Interfaces.IssueSeverity;
using ServicesModelIssueSeverity = TarsEngine.Services.Models.IssueSeverity;
using ModelValidationSeverity = TarsEngine.Models.ValidationSeverity;

namespace TarsEngine.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between different IssueSeverity enums
    /// </summary>
    public static class IssueSeverityAdapter
    {
        /// <summary>
        /// Converts from TarsEngine.Models.IssueSeverity to TarsEngine.Services.Interfaces.IssueSeverity
        /// </summary>
        /// <param name="severity">The model severity to convert</param>
        /// <returns>The interface severity</returns>
        public static InterfaceIssueSeverity ToInterfaceSeverity(ModelIssueSeverity severity)
        {
            return severity switch
            {
                ModelIssueSeverity.Blocker => InterfaceIssueSeverity.Critical,
                ModelIssueSeverity.Critical => InterfaceIssueSeverity.Critical,
                ModelIssueSeverity.Major => InterfaceIssueSeverity.Major,
                ModelIssueSeverity.Error => InterfaceIssueSeverity.Error,
                ModelIssueSeverity.Minor => InterfaceIssueSeverity.Minor,
                ModelIssueSeverity.Trivial => InterfaceIssueSeverity.Trivial,
                ModelIssueSeverity.Info => InterfaceIssueSeverity.Trivial,
                _ => InterfaceIssueSeverity.Trivial
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Interfaces.IssueSeverity to TarsEngine.Models.IssueSeverity
        /// </summary>
        /// <param name="severity">The interface severity to convert</param>
        /// <returns>The model severity</returns>
        public static ModelIssueSeverity ToModelSeverity(InterfaceIssueSeverity severity)
        {
            return severity switch
            {
                InterfaceIssueSeverity.Critical => ModelIssueSeverity.Critical,
                InterfaceIssueSeverity.Major => ModelIssueSeverity.Major,
                InterfaceIssueSeverity.Error => ModelIssueSeverity.Error,
                InterfaceIssueSeverity.Minor => ModelIssueSeverity.Minor,
                InterfaceIssueSeverity.Trivial => ModelIssueSeverity.Trivial,
                InterfaceIssueSeverity.Warning => ModelIssueSeverity.Minor,
                _ => ModelIssueSeverity.Info
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Models.IssueSeverity to TarsEngine.Services.Interfaces.IssueSeverity
        /// </summary>
        /// <param name="severity">The services model severity to convert</param>
        /// <returns>The interface severity</returns>
        public static InterfaceIssueSeverity ToInterfaceSeverity(ServicesModelIssueSeverity severity)
        {
            return severity switch
            {
                ServicesModelIssueSeverity.Critical => InterfaceIssueSeverity.Critical,
                ServicesModelIssueSeverity.Error => InterfaceIssueSeverity.Error,
                ServicesModelIssueSeverity.Warning => InterfaceIssueSeverity.Warning,
                ServicesModelIssueSeverity.Suggestion => InterfaceIssueSeverity.Trivial,
                ServicesModelIssueSeverity.Info => InterfaceIssueSeverity.Trivial,
                _ => InterfaceIssueSeverity.Trivial
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Interfaces.IssueSeverity to TarsEngine.Services.Models.IssueSeverity
        /// </summary>
        /// <param name="severity">The interface severity to convert</param>
        /// <returns>The services model severity</returns>
        public static ServicesModelIssueSeverity ToServicesModelSeverity(InterfaceIssueSeverity severity)
        {
            return severity switch
            {
                InterfaceIssueSeverity.Critical => ServicesModelIssueSeverity.Critical,
                InterfaceIssueSeverity.Error => ServicesModelIssueSeverity.Error,
                InterfaceIssueSeverity.Major => ServicesModelIssueSeverity.Error,
                InterfaceIssueSeverity.Warning => ServicesModelIssueSeverity.Warning,
                InterfaceIssueSeverity.Minor => ServicesModelIssueSeverity.Warning,
                InterfaceIssueSeverity.Trivial => ServicesModelIssueSeverity.Info,
                _ => ServicesModelIssueSeverity.Info
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Models.IssueSeverity to TarsEngine.Services.Models.IssueSeverity
        /// </summary>
        /// <param name="severity">The model severity to convert</param>
        /// <returns>The services model severity</returns>
        public static ServicesModelIssueSeverity ToServicesModelSeverity(ModelIssueSeverity severity)
        {
            return severity switch
            {
                ModelIssueSeverity.Blocker => ServicesModelIssueSeverity.Critical,
                ModelIssueSeverity.Critical => ServicesModelIssueSeverity.Critical,
                ModelIssueSeverity.Major => ServicesModelIssueSeverity.Error,
                ModelIssueSeverity.Error => ServicesModelIssueSeverity.Error,
                ModelIssueSeverity.Minor => ServicesModelIssueSeverity.Warning,
                ModelIssueSeverity.Trivial => ServicesModelIssueSeverity.Info,
                ModelIssueSeverity.Info => ServicesModelIssueSeverity.Info,
                _ => ServicesModelIssueSeverity.Info
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Models.IssueSeverity to TarsEngine.Models.IssueSeverity
        /// </summary>
        /// <param name="severity">The services model severity to convert</param>
        /// <returns>The model severity</returns>
        public static ModelIssueSeverity ToModelSeverity(ServicesModelIssueSeverity severity)
        {
            return severity switch
            {
                ServicesModelIssueSeverity.Critical => ModelIssueSeverity.Critical,
                ServicesModelIssueSeverity.Error => ModelIssueSeverity.Error,
                ServicesModelIssueSeverity.Warning => ModelIssueSeverity.Minor,
                ServicesModelIssueSeverity.Suggestion => ModelIssueSeverity.Trivial,
                ServicesModelIssueSeverity.Info => ModelIssueSeverity.Info,
                _ => ModelIssueSeverity.Info
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Models.ValidationSeverity to TarsEngine.Services.Interfaces.IssueSeverity
        /// </summary>
        /// <param name="severity">The validation severity to convert</param>
        /// <returns>The interface severity</returns>
        public static InterfaceIssueSeverity ValidationSeverityToInterfaceSeverity(ModelValidationSeverity severity)
        {
            return severity switch
            {
                ModelValidationSeverity.Critical => InterfaceIssueSeverity.Critical,
                ModelValidationSeverity.Error => InterfaceIssueSeverity.Error,
                ModelValidationSeverity.Warning => InterfaceIssueSeverity.Warning,
                ModelValidationSeverity.Info => InterfaceIssueSeverity.Trivial,
                _ => InterfaceIssueSeverity.Trivial
            };
        }

        /// <summary>
        /// Converts from TarsEngine.Services.Interfaces.IssueSeverity to TarsEngine.Models.ValidationSeverity
        /// </summary>
        /// <param name="severity">The interface severity to convert</param>
        /// <returns>The validation severity</returns>
        public static ModelValidationSeverity InterfaceSeverityToValidationSeverity(InterfaceIssueSeverity severity)
        {
            return severity switch
            {
                InterfaceIssueSeverity.Critical => ModelValidationSeverity.Critical,
                InterfaceIssueSeverity.Error => ModelValidationSeverity.Error,
                InterfaceIssueSeverity.Major => ModelValidationSeverity.Error,
                InterfaceIssueSeverity.Warning => ModelValidationSeverity.Warning,
                InterfaceIssueSeverity.Minor => ModelValidationSeverity.Warning,
                InterfaceIssueSeverity.Trivial => ModelValidationSeverity.Info,
                _ => ModelValidationSeverity.Info
            };
        }
    }
}
