using TarsEngine.Models;
using TarsEngine.Services.Interfaces;
using TarsEngine.Services.Models;
using TarsEngine.Utilities;
using ModelIssueSeverity = TarsEngine.Models.IssueSeverity;
using InterfaceIssueSeverity = TarsEngine.Services.Interfaces.IssueSeverity;
using ServicesModelIssueSeverity = TarsEngine.Services.Models.IssueSeverity;

namespace TarsEngine.Services.Adapters
{
    /// <summary>
    /// Converter between different IssueSeverity types
    /// </summary>
    public static class IssueSeverityConverter
    {
        /// <summary>
        /// Converts from Models.IssueSeverity to Services.Interfaces.IssueSeverity
        /// </summary>
        public static InterfaceIssueSeverity ToServiceSeverity(this ModelIssueSeverity severity)
        {
            // Use the existing converter in TarsEngine.Utilities
            return IssueSeverityAdapter.ToInterfaceSeverity(severity);
        }

        /// <summary>
        /// Converts from Services.Interfaces.IssueSeverity to Models.IssueSeverity
        /// </summary>
        public static ModelIssueSeverity ToModelSeverity(this InterfaceIssueSeverity severity)
        {
            // Use the existing converter in TarsEngine.Utilities
            return IssueSeverityAdapter.ToModelSeverity(severity);
        }

        /// <summary>
        /// Converts from Services.Models.IssueSeverity to Services.Interfaces.IssueSeverity
        /// </summary>
        public static InterfaceIssueSeverity ToServiceSeverity(this ServicesModelIssueSeverity severity)
        {
            // Use the existing converter in TarsEngine.Utilities
            return IssueSeverityAdapter.ToInterfaceSeverity(severity);
        }

        /// <summary>
        /// Converts from Services.Interfaces.IssueSeverity to Services.Models.IssueSeverity
        /// </summary>
        public static ServicesModelIssueSeverity ToServicesModelSeverity(this InterfaceIssueSeverity severity)
        {
            // Use the existing converter in TarsEngine.Utilities
            return IssueSeverityAdapter.ToServicesModelSeverity(severity);
        }
    }
}
