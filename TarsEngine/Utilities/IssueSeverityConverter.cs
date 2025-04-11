using TarsEngine.Models;

namespace TarsEngine.Utilities
{
    /// <summary>
    /// Provides conversion methods between different IssueSeverity enums
    /// </summary>
    public static class IssueSeverityConverter
    {
        /// <summary>
        /// Converts from Models.IssueSeverity to IssueSeverityUnified
        /// </summary>
        public static IssueSeverityUnified ToUnified(Models.IssueSeverity severity)
        {
            return severity switch
            {
                Models.IssueSeverity.Blocker => IssueSeverityUnified.Blocker,
                Models.IssueSeverity.Critical => IssueSeverityUnified.Critical,
                Models.IssueSeverity.Major => IssueSeverityUnified.Major,
                Models.IssueSeverity.Minor => IssueSeverityUnified.Minor,
                Models.IssueSeverity.Trivial => IssueSeverityUnified.Trivial,
                Models.IssueSeverity.Error => IssueSeverityUnified.Error,
                Models.IssueSeverity.Info => IssueSeverityUnified.Info,
                _ => IssueSeverityUnified.Info
            };
        }

        /// <summary>
        /// Converts from IssueSeverityUnified to Models.IssueSeverity
        /// </summary>
        public static Models.IssueSeverity FromUnified(IssueSeverityUnified severity)
        {
            return severity switch
            {
                IssueSeverityUnified.Blocker => Models.IssueSeverity.Blocker,
                IssueSeverityUnified.Critical => Models.IssueSeverity.Critical,
                IssueSeverityUnified.Major => Models.IssueSeverity.Major,
                IssueSeverityUnified.Minor => Models.IssueSeverity.Minor,
                IssueSeverityUnified.Trivial => Models.IssueSeverity.Trivial,
                IssueSeverityUnified.Error => Models.IssueSeverity.Error,
                IssueSeverityUnified.Info => Models.IssueSeverity.Info,
                _ => Models.IssueSeverity.Info
            };
        }

        /// <summary>
        /// Converts from Services.Interfaces.IssueSeverity to IssueSeverityUnified
        /// </summary>
        public static IssueSeverityUnified ToUnified(Services.Interfaces.IssueSeverity severity)
        {
            return severity switch
            {
                Services.Interfaces.IssueSeverity.Critical => IssueSeverityUnified.Critical,
                Services.Interfaces.IssueSeverity.Major => IssueSeverityUnified.Major,
                Services.Interfaces.IssueSeverity.Minor => IssueSeverityUnified.Minor,
                Services.Interfaces.IssueSeverity.Trivial => IssueSeverityUnified.Trivial,
                _ => IssueSeverityUnified.Info
            };
        }

        /// <summary>
        /// Converts from IssueSeverityUnified to Services.Interfaces.IssueSeverity
        /// </summary>
        public static Services.Interfaces.IssueSeverity ToInterfacesSeverity(IssueSeverityUnified severity)
        {
            return severity switch
            {
                IssueSeverityUnified.Blocker => Services.Interfaces.IssueSeverity.Critical,
                IssueSeverityUnified.Critical => Services.Interfaces.IssueSeverity.Critical,
                IssueSeverityUnified.Major => Services.Interfaces.IssueSeverity.Major,
                IssueSeverityUnified.Error => Services.Interfaces.IssueSeverity.Major,
                IssueSeverityUnified.Warning => Services.Interfaces.IssueSeverity.Minor,
                IssueSeverityUnified.Minor => Services.Interfaces.IssueSeverity.Minor,
                IssueSeverityUnified.Trivial => Services.Interfaces.IssueSeverity.Trivial,
                IssueSeverityUnified.Suggestion => Services.Interfaces.IssueSeverity.Trivial,
                IssueSeverityUnified.Info => Services.Interfaces.IssueSeverity.Trivial,
                _ => Services.Interfaces.IssueSeverity.Trivial
            };
        }

        /// <summary>
        /// Converts from Services.Models.IssueSeverity to IssueSeverityUnified
        /// </summary>
        public static IssueSeverityUnified ToUnified(Services.Models.IssueSeverity severity)
        {
            return severity switch
            {
                Services.Models.IssueSeverity.Critical => IssueSeverityUnified.Critical,
                Services.Models.IssueSeverity.Error => IssueSeverityUnified.Error,
                Services.Models.IssueSeverity.Warning => IssueSeverityUnified.Warning,
                Services.Models.IssueSeverity.Suggestion => IssueSeverityUnified.Suggestion,
                Services.Models.IssueSeverity.Info => IssueSeverityUnified.Info,
                _ => IssueSeverityUnified.Info
            };
        }

        /// <summary>
        /// Converts from IssueSeverityUnified to Services.Models.IssueSeverity
        /// </summary>
        public static Services.Models.IssueSeverity ToModelsSeverity(IssueSeverityUnified severity)
        {
            return severity switch
            {
                IssueSeverityUnified.Blocker => Services.Models.IssueSeverity.Critical,
                IssueSeverityUnified.Critical => Services.Models.IssueSeverity.Critical,
                IssueSeverityUnified.Major => Services.Models.IssueSeverity.Error,
                IssueSeverityUnified.Error => Services.Models.IssueSeverity.Error,
                IssueSeverityUnified.Warning => Services.Models.IssueSeverity.Warning,
                IssueSeverityUnified.Minor => Services.Models.IssueSeverity.Warning,
                IssueSeverityUnified.Trivial => Services.Models.IssueSeverity.Info,
                IssueSeverityUnified.Suggestion => Services.Models.IssueSeverity.Suggestion,
                IssueSeverityUnified.Info => Services.Models.IssueSeverity.Info,
                _ => Services.Models.IssueSeverity.Info
            };
        }

        /// <summary>
        /// Direct conversion from Models.IssueSeverity to Services.Interfaces.IssueSeverity
        /// </summary>
        public static Services.Interfaces.IssueSeverity ToInterfacesSeverity(Models.IssueSeverity severity)
        {
            return ToInterfacesSeverity(ToUnified(severity));
        }

        /// <summary>
        /// Direct conversion from Services.Interfaces.IssueSeverity to Models.IssueSeverity
        /// </summary>
        public static Models.IssueSeverity ToModelsSeverity(Services.Interfaces.IssueSeverity severity)
        {
            return FromUnified(ToUnified(severity));
        }

        /// <summary>
        /// Direct conversion from Models.IssueSeverity to Services.Models.IssueSeverity
        /// </summary>
        public static Services.Models.IssueSeverity ToServiceModelsSeverity(Models.IssueSeverity severity)
        {
            return ToModelsSeverity(ToUnified(severity));
        }

        /// <summary>
        /// Direct conversion from Services.Models.IssueSeverity to Models.IssueSeverity
        /// </summary>
        public static Models.IssueSeverity ToModelsSeverity(Services.Models.IssueSeverity severity)
        {
            return FromUnified(ToUnified(severity));
        }

        /// <summary>
        /// Direct conversion from Services.Interfaces.IssueSeverity to Services.Models.IssueSeverity
        /// </summary>
        public static Services.Models.IssueSeverity ToServiceModelsSeverity(Services.Interfaces.IssueSeverity severity)
        {
            return ToModelsSeverity(ToUnified(severity));
        }

        /// <summary>
        /// Direct conversion from Services.Models.IssueSeverity to Services.Interfaces.IssueSeverity
        /// </summary>
        public static Services.Interfaces.IssueSeverity ToInterfacesSeverity(Services.Models.IssueSeverity severity)
        {
            return ToInterfacesSeverity(ToUnified(severity));
        }
    }
}
