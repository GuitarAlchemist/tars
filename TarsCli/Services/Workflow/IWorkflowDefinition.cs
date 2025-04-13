using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsCli.Services.Workflow
{
    /// <summary>
    /// Interface for workflow definitions
    /// </summary>
    public interface IWorkflowDefinition
    {
        /// <summary>
        /// Gets the type of the workflow
        /// </summary>
        string Type { get; }

        /// <summary>
        /// Gets the name of the workflow
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the description of the workflow
        /// </summary>
        string Description { get; }

        /// <summary>
        /// Gets the initial state of the workflow
        /// </summary>
        /// <returns>Initial state</returns>
        string GetInitialState();

        /// <summary>
        /// Gets the next state based on the current state and result
        /// </summary>
        /// <param name="currentState">Current state</param>
        /// <param name="result">Result of the current state</param>
        /// <returns>Next state, or null if there is no next state</returns>
        string GetNextState(string currentState, object result);

        /// <summary>
        /// Checks if a state is a final state
        /// </summary>
        /// <param name="state">State to check</param>
        /// <returns>True if the state is a final state, false otherwise</returns>
        bool IsFinalState(string state);

        /// <summary>
        /// Checks if a transition from one state to another is valid
        /// </summary>
        /// <param name="fromState">Source state</param>
        /// <param name="toState">Target state</param>
        /// <returns>True if the transition is valid, false otherwise</returns>
        bool IsValidTransition(string fromState, string toState);

        /// <summary>
        /// Validates workflow parameters
        /// </summary>
        /// <param name="parameters">Parameters to validate</param>
        /// <returns>Validation result</returns>
        ParameterValidationResult ValidateParameters(Dictionary<string, object> parameters);

        /// <summary>
        /// Executes a workflow state
        /// </summary>
        /// <param name="workflow">Workflow instance</param>
        /// <param name="state">State to execute</param>
        /// <returns>Result of the state execution</returns>
        Task<object> ExecuteStateAsync(WorkflowInstance workflow, string state);
    }

    /// <summary>
    /// Result of parameter validation
    /// </summary>
    public class ParameterValidationResult
    {
        /// <summary>
        /// Whether the parameters are valid
        /// </summary>
        public bool IsValid { get; set; }

        /// <summary>
        /// Error message if the parameters are invalid
        /// </summary>
        public string ErrorMessage { get; set; }

        /// <summary>
        /// Creates a successful validation result
        /// </summary>
        /// <returns>Validation result</returns>
        public static ParameterValidationResult Success()
        {
            return new ParameterValidationResult { IsValid = true };
        }

        /// <summary>
        /// Creates a failed validation result
        /// </summary>
        /// <param name="errorMessage">Error message</param>
        /// <returns>Validation result</returns>
        public static ParameterValidationResult Failure(string errorMessage)
        {
            return new ParameterValidationResult { IsValid = false, ErrorMessage = errorMessage };
        }
    }
}
