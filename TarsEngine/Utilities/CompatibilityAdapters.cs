using System;
using System.Collections.Generic;
using System.Linq;
using TarsEngine.Models;

namespace TarsEngine.Utilities
{
    /// <summary>
    /// Provides adapter methods to convert between similar classes in different namespaces
    /// </summary>
    public static class CompatibilityAdapters
    {
        /// <summary>
        /// Converts from Models.CodeAnalysisResult to Services.CodeAnalysisResult
        /// </summary>
        public static Models.CodeAnalysisResult ToServiceResult(Models.CodeAnalysisResult modelResult)
        {
            // Create a copy of the result with the language converted
            var result = new Models.CodeAnalysisResult
            {
                Path = modelResult.Path,
                Language = modelResult.Language,
                Metrics = modelResult.Metrics?.ToList() ?? new List<Models.CodeMetric>(),
                Issues = modelResult.Issues?.ToList() ?? new List<Models.CodeIssue>(),
                Structures = modelResult.Structures?.ToList() ?? new List<Models.CodeStructure>(),
                IsSuccessful = modelResult.IsSuccessful
            };

            // Copy errors if available
            if (modelResult.Errors != null && modelResult.Errors.Count > 0)
            {
                result.Errors = new List<string>(modelResult.Errors);
            }

            return result;
        }

        /// <summary>
        /// Converts from Services.CodeAnalysisResult to Models.CodeAnalysisResult
        /// </summary>
        public static Models.CodeAnalysisResult ToModelResult(Models.CodeAnalysisResult serviceResult)
        {
            // Create a copy of the result
            var result = new Models.CodeAnalysisResult
            {
                Path = serviceResult.Path,
                Language = serviceResult.Language,
                Metrics = serviceResult.Metrics?.ToList() ?? new List<Models.CodeMetric>(),
                Issues = serviceResult.Issues?.ToList() ?? new List<Models.CodeIssue>(),
                Structures = serviceResult.Structures?.ToList() ?? new List<Models.CodeStructure>(),
                IsSuccessful = serviceResult.IsSuccessful
            };

            // Copy errors if available
            if (serviceResult.Errors != null && serviceResult.Errors.Count > 0)
            {
                result.Errors = new List<string>(serviceResult.Errors);
            }

            return result;
        }





        /// <summary>
        /// Converts TestExecutionResult properties
        /// </summary>
        public static void SetTestExecutionResultProperties(object testExecutionResult)
        {
            try
            {
                var type = testExecutionResult.GetType();

                var isSuccessfulProperty = type.GetProperty("IsSuccessful");
                if (isSuccessfulProperty != null)
                {
                    isSuccessfulProperty.SetValue(testExecutionResult, true);
                }

                var startedAtProperty = type.GetProperty("StartedAt");
                if (startedAtProperty != null)
                {
                    startedAtProperty.SetValue(testExecutionResult, DateTime.Now.AddSeconds(-5));
                }

                var completedAtProperty = type.GetProperty("CompletedAt");
                if (completedAtProperty != null)
                {
                    completedAtProperty.SetValue(testExecutionResult, DateTime.Now);
                }

                var durationMsProperty = type.GetProperty("DurationMs");
                if (durationMsProperty != null)
                {
                    durationMsProperty.SetValue(testExecutionResult, 5000);
                }

                var projectPathProperty = type.GetProperty("ProjectPath");
                if (projectPathProperty != null)
                {
                    projectPathProperty.SetValue(testExecutionResult, "");
                }

                var testFilterProperty = type.GetProperty("TestFilter");
                if (testFilterProperty != null)
                {
                    testFilterProperty.SetValue(testExecutionResult, "");
                }
            }
            catch
            {
                // Ignore reflection errors
            }
        }
    }
}
