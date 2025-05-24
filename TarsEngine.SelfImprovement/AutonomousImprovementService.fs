namespace TarsEngine.SelfImprovement

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Service for autonomous improvement of code
/// </summary>
type AutonomousImprovementService(logger: ILogger) =
    /// <summary>
    /// Starts the autonomous improvement process
    /// </summary>
    member _.StartImprovementWorkflow(targetDirectories: string list, maxDurationMinutes: int) =
        task {
            logger.LogInformation("Starting autonomous improvement workflow")

            // Create the workflow steps
            let knowledgeExtractionHandler = KnowledgeExtractionStep.getHandler logger 100
            let codeAnalysisHandler = CodeAnalysisStep.getHandler logger
            let improvementApplicationHandler = ImprovementApplicationStep.getHandler logger 10
            let feedbackCollectionHandler = FeedbackCollectionStep.getHandler logger
            let reportingHandler = ReportingStep.getHandler logger

            // Create the workflow handlers list
            let handlers = [
                knowledgeExtractionHandler
                codeAnalysisHandler
                improvementApplicationHandler
                feedbackCollectionHandler
                reportingHandler
            ]

            // Create and execute the workflow
            let timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss")
            let workflowName = "Autonomous Improvement " + timestamp
            let! result = WorkflowEngine.createAndExecuteWorkflow logger workflowName targetDirectories maxDurationMinutes handlers

            return result
        }

    /// <summary>
    /// Gets the status of the current workflow
    /// </summary>
    member _.GetWorkflowStatus() =
        task {
            let! stateOption = WorkflowState.load WorkflowState.defaultStatePath
            return stateOption
        }

    /// <summary>
    /// Runs the retroaction loop to improve pattern recognition
    /// </summary>
    member _.RunRetroactionLoop() =
        task {
            logger.LogInformation("Running retroaction loop")
            let! state = RetroactionLoop.runRetroactionLoop logger |> Async.StartAsTask
            return state
        }
