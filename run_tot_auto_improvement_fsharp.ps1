# Script to run the Tree-of-Thought auto-improvement pipeline using F#

Write-Host "Running Tree-of-Thought auto-improvement pipeline using F#..."

# Create the Samples directory if it doesn't exist
if (-not (Test-Path -Path "Samples")) {
    New-Item -Path "Samples" -ItemType Directory | Out-Null
}

# Create a sample code file if it doesn't exist
$sampleCodePath = "Samples\SampleCode.cs"
if (-not (Test-Path -Path $sampleCodePath)) {
    $sampleCode = @"
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Samples
{
    public class SampleCode
    {
        public void ProcessData(List<int> data)
        {
            // Process the data
            for (int i = 0; i < data.Count; i++)
            {
                // Inefficient string concatenation in loop
                string result = "";
                for (int j = 0; j < data[i]; j++)
                {
                    result += j.ToString();
                }
                Console.WriteLine(result);
                
                // Potential division by zero
                int divisor = data[i] - 10;
                int quotient = 100 / divisor;
                Console.WriteLine($"Quotient: {quotient}");
                
                // Inefficient LINQ in loop
                var filtered = data.Where(x => x > 10).ToList();
                foreach (var item in filtered)
                {
                    Console.WriteLine(item);
                }
                
                // Magic numbers
                if (data[i] > 42)
                {
                    Console.WriteLine("The answer to life, the universe, and everything");
                }
                
                // Hardcoded credentials
                string password = "p@ssw0rd";
                Console.WriteLine($"Using password: {password}");
            }
        }
        
        public void SearchData(List<int> data, int target)
        {
            // Inefficient collection usage
            bool found = false;
            for (int i = 0; i < data.Count; i++)
            {
                if (data.Contains(target))
                {
                    found = true;
                    break;
                }
            }
            Console.WriteLine($"Found: {found}");
        }
    }
}
"@
    Set-Content -Path $sampleCodePath -Value $sampleCode
    Write-Host "Created sample code file: $sampleCodePath"
}

# Create the output directory if it doesn't exist
if (-not (Test-Path -Path "tot_output_fsharp")) {
    New-Item -Path "tot_output_fsharp" -ItemType Directory | Out-Null
}

# Run the metascript
Write-Host "Running Tree-of-Thought auto-improvement metascript..."
dotnet run --project TarsCli metascript run --file Metascripts/TreeOfThought/AutoImprovement.tars --variables "default_target_file=$sampleCodePath;default_improvement_type=performance;default_output_dir=tot_output_fsharp"

Write-Host "Tree-of-Thought auto-improvement pipeline using F# completed successfully!"
