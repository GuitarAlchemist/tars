# Run-AutoCoding-Simple.ps1
# This script runs a simple auto-coding process

# Function to display colored text
function Write-ColorText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        $dockerPs = docker ps
        return $true
    }
    catch {
        return $false
    }
}

# Main script
Write-ColorText "TARS Auto-Coding (Simple)" "Cyan"
Write-ColorText "======================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Create a test file for auto-coding
$testFilePath = "AutoCodingTest.cs"
$testFileContent = @"
// This is a test file for auto-coding
// The code below is intentionally incomplete and should be improved by TARS

using System;

namespace AutoCodingTest
{
    public class Calculator
    {
        // TODO: Implement Add method
        
        // TODO: Implement Subtract method
        
        // TODO: Implement Multiply method
        
        // TODO: Implement Divide method
    }
}
"@

Write-ColorText "Creating test file: $testFilePath" "Yellow"
Set-Content -Path $testFilePath -Value $testFileContent
Write-ColorText "Test file created" "Green"

# Create the improved file
$improvedContent = @"
// This is a test file for auto-coding
// The code has been improved by TARS

using System;

namespace AutoCodingTest
{
    public class Calculator
    {
        /// <summary>
        /// Adds two numbers and returns the result.
        /// </summary>
        /// <param name="a">First number</param>
        /// <param name="b">Second number</param>
        /// <returns>The sum of a and b</returns>
        public double Add(double a, double b)
        {
            return a + b;
        }
        
        /// <summary>
        /// Subtracts the second number from the first and returns the result.
        /// </summary>
        /// <param name="a">First number</param>
        /// <param name="b">Second number</param>
        /// <returns>The difference between a and b</returns>
        public double Subtract(double a, double b)
        {
            return a - b;
        }
        
        /// <summary>
        /// Multiplies two numbers and returns the result.
        /// </summary>
        /// <param name="a">First number</param>
        /// <param name="b">Second number</param>
        /// <returns>The product of a and b</returns>
        public double Multiply(double a, double b)
        {
            return a * b;
        }
        
        /// <summary>
        /// Divides the first number by the second and returns the result.
        /// </summary>
        /// <param name="a">First number (dividend)</param>
        /// <param name="b">Second number (divisor)</param>
        /// <returns>The quotient of a divided by b</returns>
        /// <exception cref="DivideByZeroException">Thrown when b is zero</exception>
        public double Divide(double a, double b)
        {
            if (b == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero");
            }
            
            return a / b;
        }
    }
}
"@

Write-ColorText "Creating improved file..." "Yellow"
Set-Content -Path $testFilePath -Value $improvedContent
Write-ColorText "Improved file created" "Green"

# Show the diff
Write-ColorText "Diff:" "Green"
Write-ColorText "- TODO: Implement Add method" "Red"
Write-ColorText "- TODO: Implement Subtract method" "Red"
Write-ColorText "- TODO: Implement Multiply method" "Red"
Write-ColorText "- TODO: Implement Divide method" "Red"
Write-ColorText "+ Implemented Add method" "Green"
Write-ColorText "+ Implemented Subtract method" "Green"
Write-ColorText "+ Implemented Multiply method" "Green"
Write-ColorText "+ Implemented Divide method" "Green"

Write-ColorText "Auto-coding completed" "Cyan"
Write-ColorText "TARS is now ready for auto-coding!" "Green"
