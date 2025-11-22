using ModelValidationIssue = TarsEngine.Models.ValidationIssue;
using InterfaceValidationIssue = TarsEngine.Services.Interfaces.ValidationIssue;
using ModelValidationSeverity = TarsEngine.Models.ValidationSeverity;
using InterfaceIssueSeverity = TarsEngine.Services.Interfaces.IssueSeverity;
using ServicesModelValidationSeverity = TarsEngine.Services.Models.ValidationSeverity;
using ServicesModelIssueSeverity = TarsEngine.Services.Models.IssueSeverity;

namespace TarsEngine.Services.Adapters;

/// <summary>
/// Adapter for converting between different ValidationIssue types
/// </summary>
public static class ValidationIssueAdapter
{
    /// <summary>
    /// Converts a model ValidationIssue to an interface ValidationIssue
    /// </summary>
    public static InterfaceValidationIssue ToInterface(ModelValidationIssue model)
    {
        if (model == null)
            return null;

        return new InterfaceValidationIssue
        {
            Description = model.Message,
            Severity = ConvertSeverityToInterface(model.Severity),
            Location = model.RuleName,
            SuggestedFix = null
        };
    }

    /// <summary>
    /// Converts a model ValidationIssue to an interface ValidationIssue
    /// </summary>
    public static InterfaceValidationIssue ConvertToInterface(ModelValidationIssue model)
    {
        return ToInterface(model);
    }

    /// <summary>
    /// Converts an interface ValidationIssue to a model ValidationIssue
    /// </summary>
    public static ModelValidationIssue ToModel(InterfaceValidationIssue interfaceIssue)
    {
        if (interfaceIssue == null)
            return null;

        return new ModelValidationIssue
        {
            Message = interfaceIssue.Description,
            Severity = ConvertSeverityToModel(interfaceIssue.Severity),
            RuleName = interfaceIssue.Location,
            RuleId = Guid.NewGuid().ToString()
        };
    }

    /// <summary>
    /// Converts an interface ValidationIssue to a model ValidationIssue
    /// </summary>
    public static ModelValidationIssue ConvertToModel(InterfaceValidationIssue interfaceIssue)
    {
        return ToModel(interfaceIssue);
    }

    /// <summary>
    /// Converts InterfaceIssueSeverity to ModelValidationSeverity
    /// </summary>
    public static ModelValidationSeverity ConvertIssueSeverityToValidationSeverity(InterfaceIssueSeverity severity)
    {
        switch (severity)
        {
            case InterfaceIssueSeverity.Critical:
                return ModelValidationSeverity.Critical;
            case InterfaceIssueSeverity.Major:
                return ModelValidationSeverity.Error;
            case InterfaceIssueSeverity.Minor:
                return ModelValidationSeverity.Warning;
            case InterfaceIssueSeverity.Trivial:
                return ModelValidationSeverity.Info;
            default:
                return ModelValidationSeverity.Info;
        }
    }

    /// <summary>
    /// Converts ModelValidationSeverity to InterfaceIssueSeverity
    /// </summary>
    public static InterfaceIssueSeverity ConvertValidationSeverityToIssueSeverity(ModelValidationSeverity severity)
    {
        switch (severity)
        {
            case ModelValidationSeverity.Critical:
                return InterfaceIssueSeverity.Critical;
            case ModelValidationSeverity.Error:
                return InterfaceIssueSeverity.Major;
            case ModelValidationSeverity.Warning:
                return InterfaceIssueSeverity.Minor;
            case ModelValidationSeverity.Info:
                return InterfaceIssueSeverity.Trivial;
            default:
                return InterfaceIssueSeverity.Trivial;
        }
    }

    // Helper methods for converting between different severity types
    private static InterfaceIssueSeverity ConvertSeverityToInterface(ModelValidationSeverity severity)
    {
        return ConvertValidationSeverityToIssueSeverity(severity);
    }

    private static ModelValidationSeverity ConvertSeverityToModel(InterfaceIssueSeverity severity)
    {
        return ConvertIssueSeverityToValidationSeverity(severity);
    }

    /// <summary>
    /// Converts ServicesModelValidationSeverity to InterfaceIssueSeverity
    /// </summary>
    public static InterfaceIssueSeverity ConvertServicesValidationSeverityToInterface(ServicesModelValidationSeverity severity)
    {
        switch (severity)
        {
            case ServicesModelValidationSeverity.Critical:
                return InterfaceIssueSeverity.Critical;
            case ServicesModelValidationSeverity.Error:
                return InterfaceIssueSeverity.Major;
            case ServicesModelValidationSeverity.Warning:
                return InterfaceIssueSeverity.Minor;
            case ServicesModelValidationSeverity.Info:
                return InterfaceIssueSeverity.Trivial;
            default:
                return InterfaceIssueSeverity.Trivial;
        }
    }

    /// <summary>
    /// Converts InterfaceIssueSeverity to ServicesModelValidationSeverity
    /// </summary>
    public static ServicesModelValidationSeverity ConvertInterfaceToServicesValidationSeverity(InterfaceIssueSeverity severity)
    {
        switch (severity)
        {
            case InterfaceIssueSeverity.Critical:
                return ServicesModelValidationSeverity.Critical;
            case InterfaceIssueSeverity.Major:
                return ServicesModelValidationSeverity.Error;
            case InterfaceIssueSeverity.Minor:
                return ServicesModelValidationSeverity.Warning;
            case InterfaceIssueSeverity.Trivial:
                return ServicesModelValidationSeverity.Info;
            default:
                return ServicesModelValidationSeverity.Info;
        }
    }

    /// <summary>
    /// Converts ServicesModelIssueSeverity to InterfaceIssueSeverity
    /// </summary>
    public static InterfaceIssueSeverity ConvertServicesIssueSeverityToInterface(ServicesModelIssueSeverity severity)
    {
        switch (severity)
        {
            case ServicesModelIssueSeverity.Critical:
                return InterfaceIssueSeverity.Critical;
            case ServicesModelIssueSeverity.Error:
                return InterfaceIssueSeverity.Major;
            case ServicesModelIssueSeverity.Warning:
                return InterfaceIssueSeverity.Minor;
            case ServicesModelIssueSeverity.Suggestion:
            case ServicesModelIssueSeverity.Info:
                return InterfaceIssueSeverity.Trivial;
            default:
                return InterfaceIssueSeverity.Trivial;
        }
    }

    /// <summary>
    /// Converts InterfaceIssueSeverity to ServicesModelIssueSeverity
    /// </summary>
    public static ServicesModelIssueSeverity ConvertInterfaceToServicesIssueSeverity(InterfaceIssueSeverity severity)
    {
        switch (severity)
        {
            case InterfaceIssueSeverity.Critical:
                return ServicesModelIssueSeverity.Critical;
            case InterfaceIssueSeverity.Major:
                return ServicesModelIssueSeverity.Error;
            case InterfaceIssueSeverity.Minor:
                return ServicesModelIssueSeverity.Warning;
            case InterfaceIssueSeverity.Trivial:
                return ServicesModelIssueSeverity.Info;
            default:
                return ServicesModelIssueSeverity.Info;
        }
    }
}