namespace TarsCli.Models;

/// <summary>
/// Defines constants and utilities for the TARS-Augment collaboration protocol
/// </summary>
public static class CollaborationProtocol
{
    /// <summary>
    /// Message types for collaboration
    /// </summary>
    public static class MessageTypes
    {
        /// <summary>
        /// Knowledge transfer message
        /// </summary>
        public const string Knowledge = "knowledge";

        /// <summary>
        /// Code improvement message
        /// </summary>
        public const string CodeImprovement = "code_improvement";

        /// <summary>
        /// Feedback message
        /// </summary>
        public const string Feedback = "feedback";

        /// <summary>
        /// Status update message
        /// </summary>
        public const string StatusUpdate = "status_update";

        /// <summary>
        /// Control message
        /// </summary>
        public const string Control = "control";

        /// <summary>
        /// Error message
        /// </summary>
        public const string Error = "error";
    }

    /// <summary>
    /// Operations for knowledge messages
    /// </summary>
    public static class KnowledgeOperations
    {
        /// <summary>
        /// Extract knowledge from a file
        /// </summary>
        public const string Extract = "extract";

        /// <summary>
        /// Apply knowledge to a file
        /// </summary>
        public const string Apply = "apply";

        /// <summary>
        /// Generate a knowledge report
        /// </summary>
        public const string Report = "report";

        /// <summary>
        /// Run a knowledge improvement cycle
        /// </summary>
        public const string Cycle = "cycle";

        /// <summary>
        /// Transfer knowledge between systems
        /// </summary>
        public const string Transfer = "transfer";

        /// <summary>
        /// Request knowledge about a specific topic
        /// </summary>
        public const string Request = "request";
    }

    /// <summary>
    /// Operations for code improvement messages
    /// </summary>
    public static class CodeImprovementOperations
    {
        /// <summary>
        /// Suggest improvements for a file
        /// </summary>
        public const string Suggest = "suggest";

        /// <summary>
        /// Apply suggested improvements
        /// </summary>
        public const string Apply = "apply";

        /// <summary>
        /// Review improvements
        /// </summary>
        public const string Review = "review";

        /// <summary>
        /// Validate improvements
        /// </summary>
        public const string Validate = "validate";
    }

    /// <summary>
    /// Operations for feedback messages
    /// </summary>
    public static class FeedbackOperations
    {
        /// <summary>
        /// Provide feedback on an improvement
        /// </summary>
        public const string Provide = "provide";

        /// <summary>
        /// Request feedback on an improvement
        /// </summary>
        public const string Request = "request";

        /// <summary>
        /// Acknowledge feedback
        /// </summary>
        public const string Acknowledge = "acknowledge";
    }

    /// <summary>
    /// Operations for control messages
    /// </summary>
    public static class ControlOperations
    {
        /// <summary>
        /// Start a collaboration session
        /// </summary>
        public const string Start = "start";

        /// <summary>
        /// End a collaboration session
        /// </summary>
        public const string End = "end";

        /// <summary>
        /// Pause a collaboration session
        /// </summary>
        public const string Pause = "pause";

        /// <summary>
        /// Resume a collaboration session
        /// </summary>
        public const string Resume = "resume";

        /// <summary>
        /// Cancel a specific operation
        /// </summary>
        public const string Cancel = "cancel";
    }

    /// <summary>
    /// Message status values
    /// </summary>
    public static class MessageStatus
    {
        /// <summary>
        /// Request status
        /// </summary>
        public const string Request = "request";

        /// <summary>
        /// In progress status
        /// </summary>
        public const string InProgress = "in_progress";

        /// <summary>
        /// Completed status
        /// </summary>
        public const string Completed = "completed";

        /// <summary>
        /// Error status
        /// </summary>
        public const string Error = "error";

        /// <summary>
        /// Cancelled status
        /// </summary>
        public const string Cancelled = "cancelled";
    }

    /// <summary>
    /// Roles in the collaboration
    /// </summary>
    public static class Roles
    {
        /// <summary>
        /// TARS role
        /// </summary>
        public const string Tars = "TARS";

        /// <summary>
        /// Augment role
        /// </summary>
        public const string Augment = "Augment";
    }

    /// <summary>
    /// Collaboration phases
    /// </summary>
    public static class Phases
    {
        /// <summary>
        /// Knowledge extraction phase
        /// </summary>
        public const string KnowledgeExtraction = "knowledge_extraction";

        /// <summary>
        /// Knowledge application phase
        /// </summary>
        public const string KnowledgeApplication = "knowledge_application";

        /// <summary>
        /// Code improvement phase
        /// </summary>
        public const string CodeImprovement = "code_improvement";

        /// <summary>
        /// Feedback phase
        /// </summary>
        public const string Feedback = "feedback";

        /// <summary>
        /// Learning phase
        /// </summary>
        public const string Learning = "learning";
    }

    /// <summary>
    /// Creates a new collaboration message
    /// </summary>
    /// <param name="type">The message type</param>
    /// <param name="operation">The operation</param>
    /// <param name="content">The message content</param>
    /// <param name="sender">The sender</param>
    /// <param name="recipient">The recipient</param>
    /// <returns>A new collaboration message</returns>
    public static CollaborationMessage CreateMessage(
        string type,
        string operation,
        object content,
        string sender = Roles.Tars,
        string recipient = Roles.Augment)
    {
        return new CollaborationMessage
        {
            Type = type,
            Operation = operation,
            Content = content,
            Sender = sender,
            Recipient = recipient,
            Status = MessageStatus.Request
        };
    }

    /// <summary>
    /// Creates a response message
    /// </summary>
    /// <param name="originalMessage">The original message</param>
    /// <param name="content">The response content</param>
    /// <param name="status">The response status</param>
    /// <returns>A response message</returns>
    public static CollaborationMessage CreateResponse(
        CollaborationMessage originalMessage,
        object content,
        string status = MessageStatus.Completed)
    {
        return new CollaborationMessage
        {
            Type = originalMessage.Type,
            Operation = originalMessage.Operation,
            Content = content,
            Sender = originalMessage.Recipient,
            Recipient = originalMessage.Sender,
            Status = status,
            InResponseTo = originalMessage.Id,
            CorrelationId = originalMessage.CorrelationId
        };
    }

    /// <summary>
    /// Creates a progress update message
    /// </summary>
    /// <param name="originalMessage">The original message</param>
    /// <param name="percentage">The progress percentage</param>
    /// <param name="currentStep">The current step</param>
    /// <param name="totalSteps">The total number of steps</param>
    /// <param name="statusMessage">The status message</param>
    /// <returns>A progress update message</returns>
    public static CollaborationMessage CreateProgressUpdate(
        CollaborationMessage originalMessage,
        int percentage,
        string currentStep,
        int totalSteps,
        string statusMessage)
    {
        return new CollaborationMessage
        {
            Type = MessageTypes.StatusUpdate,
            Operation = originalMessage.Operation,
            Content = null,
            Sender = originalMessage.Recipient,
            Recipient = originalMessage.Sender,
            Status = MessageStatus.InProgress,
            InResponseTo = originalMessage.Id,
            CorrelationId = originalMessage.CorrelationId,
            Progress = new ProgressInfo
            {
                Percentage = percentage,
                CurrentStep = currentStep,
                TotalSteps = totalSteps,
                StatusMessage = statusMessage
            }
        };
    }

    /// <summary>
    /// Creates an error message
    /// </summary>
    /// <param name="originalMessage">The original message</param>
    /// <param name="errorMessage">The error message</param>
    /// <param name="errorDetails">Additional error details</param>
    /// <returns>An error message</returns>
    public static CollaborationMessage CreateErrorMessage(
        CollaborationMessage originalMessage,
        string errorMessage,
        object errorDetails = null)
    {
        return new CollaborationMessage
        {
            Type = MessageTypes.Error,
            Operation = originalMessage.Operation,
            Content = new
            {
                ErrorMessage = errorMessage,
                ErrorDetails = errorDetails
            },
            Sender = originalMessage.Recipient,
            Recipient = originalMessage.Sender,
            Status = MessageStatus.Error,
            InResponseTo = originalMessage.Id,
            CorrelationId = originalMessage.CorrelationId
        };
    }
}
