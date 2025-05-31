// TARS Autonomous Project - Generated Code
// Created: 05/29/2025 21:24:13

namespace TarsAutonomousProject
{
    /// <summary>
    /// Demonstrates TARS autonomous project creation
    /// </summary>
    public class AutonomousProjectDemo
    {
        public string ProjectName { get; } = "tars_project_test_20250529_212413";
        public DateTime Created { get; } = DateTime.Now;
        
        public void DisplayInfo()
        {
            Console.WriteLine($"TARS Autonomous Project: {ProjectName}");
            Console.WriteLine($"Created: {Created}");
            Console.WriteLine("Status: Project creation SUCCESSFUL");
        }
    }
}
