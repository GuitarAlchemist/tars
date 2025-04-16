using TarsEngine.Models;

namespace TarsEngine.Utilities;

/// <summary>
/// Provides conversion methods between different IssueSeverity enums
/// </summary>
public static class IssueSeverityConverter
{
    /// <summary>
    /// Converts from Models.IssueSeverity to IssueSeverityUnified
    /// </summary>
    public static IssueSeverityUnified ToUnified(IssueSeverity severity)
    {
        return severity switch
        {
            IssueSeverity.Blocker => IssueSeverityUnified.Blocker,
            IssueSeverity.Critical => IssueSeverityUnified.Critical,
            IssueSeverity.Major => IssueSeverityUnified.Major,
            IssueSeverity.Minor => IssueSeverityUnified.Minor,
            IssueSeverity.Trivial => IssueSeverityUnified.Trivial,
            IssueSeverity.Error => IssueSeverityUnified.Error,
            IssueSeverity.Info => IssueSeverityUnified.Info,
            _ => IssueSeverityUnified.Info
        };
    }

    /// <summary>
    /// Converts from IssueSeverityUnified to Models.IssueSeverity
    /// </summary>
    public static IssueSeverity FromUnified(IssueSeverityUnified severity)
    {
        return severity switch
        {
            IssueSeverityUnified.Blocker => IssueSeverity.Blocker,
            IssueSeverityUnified.Critical => IssueSeverity.Critical,
            IssueSeverityUnified.Major => IssueSeverity.Major,
            IssueSeverityUnified.Minor => IssueSeverity.Minor,
            IssueSeverityUnified.Trivial => IssueSeverity.Trivial,
            IssueSeverityUnified.Error => IssueSeverity.Error,
            IssueSeverityUnified.Info => IssueSeverity.Info,
            _ => IssueSeverity.Info
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
    public static Services.Interfaces.IssueSeverity ToInterfacesSeverity(IssueSeverity severity)
    {
        return ToInterfacesSeverity(ToUnified(severity));
    }

    /// <summary>
    /// Direct conversion from Services.Interfaces.IssueSeverity to Models.IssueSeverity
    /// </summary>
    public static IssueSeverity ToModelsSeverity(Services.Interfaces.IssueSeverity severity)
    {
        return FromUnified(ToUnified(severity));
    }

    /// <summary>
    /// Direct conversion from Models.IssueSeverity to Services.Models.IssueSeverity
    /// </summary>
    public static Services.Models.IssueSeverity ToServiceModelsSeverity(IssueSeverity severity)
    {
        return ToModelsSeverity(ToUnified(severity));
    }

    /// <summary>
    /// Direct conversion from Services.Models.IssueSeverity to Models.IssueSeverity
    /// </summary>
    public static IssueSeverity ToModelsSeverity(Services.Models.IssueSeverity severity)
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