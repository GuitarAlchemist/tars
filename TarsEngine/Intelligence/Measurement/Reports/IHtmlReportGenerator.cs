using System.Threading.Tasks;

namespace TarsEngine.Intelligence.Measurement.Reports
{
    /// <summary>
    /// Interface for generating HTML reports from benchmark results
    /// </summary>
    public interface IHtmlReportGenerator
    {
        /// <summary>
        /// Generates an HTML report from benchmark results
        /// </summary>
        /// <param name="results">The benchmark results</param>
        /// <param name="outputPath">The output path for the HTML report</param>
        /// <returns>A task representing the asynchronous operation</returns>
        Task GenerateReportAsync(BenchmarkResults results, string outputPath);
        
        /// <summary>
        /// Generates an HTML comparison report from code comparison results
        /// </summary>
        /// <param name="results">The code comparison results</param>
        /// <param name="outputPath">The output path for the HTML report</param>
        /// <returns>A task representing the asynchronous operation</returns>
        Task GenerateComparisonReportAsync(CodeComparisonResults results, string outputPath);
    }
}
