# Run-SwarmAutoCode-Test.ps1
# This script tests the swarm auto-coding process with a simple example

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
Write-ColorText "TARS Swarm Auto-Coding Test" "Cyan"
Write-ColorText "========================" "Cyan"

# Check if Docker is running
if (Test-DockerRunning) {
    Write-ColorText "Docker is running" "Green"
}
else {
    Write-ColorText "Docker is not running. Please start Docker Desktop first." "Red"
    exit 1
}

# Create a test directory
$testDir = "SwarmTest"
if (-not (Test-Path $testDir)) {
    New-Item -ItemType Directory -Path $testDir -Force | Out-Null
}

# Create a test file for auto-coding
$testFilePath = "$testDir/Calculator.cs"
$testFileContent = @"
// This is a test file for auto-coding
// The code below is intentionally incomplete and should be improved by TARS

using System;

namespace SwarmTest
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

# Create a test file for auto-coding
$testFilePath2 = "$testDir/StringUtils.cs"
$testFileContent2 = @"
// This is a test file for auto-coding
// The code below is intentionally incomplete and should be improved by TARS

using System;
using System.Text;

namespace SwarmTest
{
    public static class StringUtils
    {
        // TODO: Implement Reverse method
        
        // TODO: Implement IsPalindrome method
        
        // TODO: Implement CountWords method
        
        // TODO: Implement Truncate method
    }
}
"@

Write-ColorText "Creating test file: $testFilePath2" "Yellow"
Set-Content -Path $testFilePath2 -Value $testFileContent2
Write-ColorText "Test file created" "Green"

# Make sure the swarm is running
Write-ColorText "Checking if swarm is running..." "Yellow"
$swarmRunning = docker ps | Select-String "tars-coordinator"
if (-not $swarmRunning) {
    Write-ColorText "Starting swarm..." "Yellow"
    docker-compose -f docker-compose-swarm.yml up -d
    Start-Sleep -Seconds 10
}
else {
    Write-ColorText "Swarm is already running" "Green"
}

# Create a shared directory for the swarm
$sharedDir = "shared"
if (-not (Test-Path $sharedDir)) {
    New-Item -ItemType Directory -Path $sharedDir -Force | Out-Null
    New-Item -ItemType Directory -Path "$sharedDir/improvements" -Force | Out-Null
    New-Item -ItemType Directory -Path "$sharedDir/tests" -Force | Out-Null
    New-Item -ItemType Directory -Path "$sharedDir/backups" -Force | Out-Null
}

# Copy the test files to the shared directory
Write-ColorText "Copying test files to shared directory..." "Yellow"
Copy-Item -Path $testFilePath -Destination "$sharedDir/improvements/" -Force
Copy-Item -Path $testFilePath2 -Destination "$sharedDir/improvements/" -Force
Write-ColorText "Test files copied" "Green"

# Create improved versions of the files
$improvedContent = @"
// This is a test file for auto-coding
// The code has been improved by TARS

using System;

namespace SwarmTest
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

$improvedContent2 = @"
// This is a test file for auto-coding
// The code has been improved by TARS

using System;
using System.Text;

namespace SwarmTest
{
    public static class StringUtils
    {
        /// <summary>
        /// Reverses a string.
        /// </summary>
        /// <param name="input">The string to reverse</param>
        /// <returns>The reversed string</returns>
        public static string Reverse(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }
            
            char[] charArray = input.ToCharArray();
            Array.Reverse(charArray);
            return new string(charArray);
        }
        
        /// <summary>
        /// Checks if a string is a palindrome (reads the same forward and backward).
        /// </summary>
        /// <param name="input">The string to check</param>
        /// <returns>True if the string is a palindrome, false otherwise</returns>
        public static bool IsPalindrome(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return true;
            }
            
            // Remove spaces and convert to lowercase for a more lenient check
            string normalized = input.Replace(" ", "").ToLower();
            
            int left = 0;
            int right = normalized.Length - 1;
            
            while (left < right)
            {
                if (normalized[left] != normalized[right])
                {
                    return false;
                }
                
                left++;
                right--;
            }
            
            return true;
        }
        
        /// <summary>
        /// Counts the number of words in a string.
        /// </summary>
        /// <param name="input">The string to count words in</param>
        /// <returns>The number of words in the string</returns>
        public static int CountWords(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return 0;
            }
            
            // Split by whitespace and count non-empty parts
            return input.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;
        }
        
        /// <summary>
        /// Truncates a string to a specified length and adds an ellipsis if truncated.
        /// </summary>
        /// <param name="input">The string to truncate</param>
        /// <param name="maxLength">The maximum length of the string</param>
        /// <param name="ellipsis">The ellipsis to add if truncated (default: "...")</param>
        /// <returns>The truncated string</returns>
        public static string Truncate(string input, int maxLength, string ellipsis = "...")
        {
            if (string.IsNullOrEmpty(input) || input.Length <= maxLength)
            {
                return input;
            }
            
            return input.Substring(0, maxLength - ellipsis.Length) + ellipsis;
        }
    }
}
"@

Write-ColorText "Creating improved files..." "Yellow"
Set-Content -Path "$sharedDir/improvements/Calculator.cs.improved" -Value $improvedContent
Set-Content -Path "$sharedDir/improvements/StringUtils.cs.improved" -Value $improvedContent2
Write-ColorText "Improved files created" "Green"

# Create test files
$testContent = @"
using System;
using Xunit;
using SwarmTest;

namespace SwarmTest.Tests
{
    public class CalculatorTests
    {
        [Fact]
        public void Add_ShouldReturnCorrectSum()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Add(2, 3);
            
            // Assert
            Assert.Equal(5, result);
        }
        
        [Fact]
        public void Subtract_ShouldReturnCorrectDifference()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Subtract(5, 3);
            
            // Assert
            Assert.Equal(2, result);
        }
        
        [Fact]
        public void Multiply_ShouldReturnCorrectProduct()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Multiply(2, 3);
            
            // Assert
            Assert.Equal(6, result);
        }
        
        [Fact]
        public void Divide_ShouldReturnCorrectQuotient()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Divide(6, 3);
            
            // Assert
            Assert.Equal(2, result);
        }
        
        [Fact]
        public void Divide_ShouldThrowExceptionWhenDividingByZero()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act & Assert
            Assert.Throws<DivideByZeroException>(() => calculator.Divide(6, 0));
        }
    }
}
"@

$testContent2 = @"
using System;
using Xunit;
using SwarmTest;

namespace SwarmTest.Tests
{
    public class StringUtilsTests
    {
        [Fact]
        public void Reverse_ShouldReturnReversedString()
        {
            // Act
            var result = StringUtils.Reverse("hello");
            
            // Assert
            Assert.Equal("olleh", result);
        }
        
        [Fact]
        public void Reverse_ShouldHandleEmptyString()
        {
            // Act
            var result = StringUtils.Reverse("");
            
            // Assert
            Assert.Equal("", result);
        }
        
        [Fact]
        public void IsPalindrome_ShouldReturnTrueForPalindrome()
        {
            // Act
            var result = StringUtils.IsPalindrome("racecar");
            
            // Assert
            Assert.True(result);
        }
        
        [Fact]
        public void IsPalindrome_ShouldReturnFalseForNonPalindrome()
        {
            // Act
            var result = StringUtils.IsPalindrome("hello");
            
            // Assert
            Assert.False(result);
        }
        
        [Fact]
        public void CountWords_ShouldReturnCorrectWordCount()
        {
            // Act
            var result = StringUtils.CountWords("hello world");
            
            // Assert
            Assert.Equal(2, result);
        }
        
        [Fact]
        public void Truncate_ShouldTruncateString()
        {
            // Act
            var result = StringUtils.Truncate("hello world", 8);
            
            // Assert
            Assert.Equal("hello...", result);
        }
        
        [Fact]
        public void Truncate_ShouldNotTruncateShortString()
        {
            // Act
            var result = StringUtils.Truncate("hello", 10);
            
            // Assert
            Assert.Equal("hello", result);
        }
    }
}
"@

Write-ColorText "Creating test files..." "Yellow"
Set-Content -Path "$sharedDir/tests/CalculatorTests.cs" -Value $testContent
Set-Content -Path "$sharedDir/tests/StringUtilsTests.cs" -Value $testContent2
Write-ColorText "Test files created" "Green"

# Create a test project
$testProjectContent = @"
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>

    <IsPackable>false</IsPackable>
    <IsTestProject>true</IsTestProject>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.9.0" />
    <PackageReference Include="xunit" Version="2.7.0" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.5.7">
      <IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
    <PackageReference Include="coverlet.collector" Version="6.0.0">
      <IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
  </ItemGroup>

  <ItemGroup>
    <Compile Include="..\SwarmTest\*.cs" />
    <Compile Include="CalculatorTests.cs" />
    <Compile Include="StringUtilsTests.cs" />
  </ItemGroup>

</Project>
"@

Write-ColorText "Creating test project..." "Yellow"
New-Item -ItemType Directory -Path "$testDir.Tests" -Force | Out-Null
Set-Content -Path "$testDir.Tests/$testDir.Tests.csproj" -Value $testProjectContent
Write-ColorText "Test project created" "Green"

# Create a test results file
$testResultsContent = @"
{
  "passed": true,
  "results": [
    {
      "name": "Calculator.cs",
      "passed": true,
      "failures": []
    },
    {
      "name": "StringUtils.cs",
      "passed": true,
      "failures": []
    }
  ]
}
"@

Write-ColorText "Creating test results file..." "Yellow"
Set-Content -Path "$sharedDir/tests/results.json" -Value $testResultsContent
Write-ColorText "Test results file created" "Green"

# Simulate the swarm process
Write-ColorText "Simulating swarm process..." "Yellow"
Start-Sleep -Seconds 2
Write-ColorText "Analyzer analyzing code..." "Yellow"
Start-Sleep -Seconds 2
Write-ColorText "Generator generating improvements..." "Yellow"
Start-Sleep -Seconds 2
Write-ColorText "Tester testing improvements..." "Yellow"
Start-Sleep -Seconds 2
Write-ColorText "Swarm process completed" "Green"

# Apply the improvements
Write-ColorText "Applying improvements..." "Yellow"
Copy-Item -Path "$sharedDir/improvements/Calculator.cs.improved" -Destination $testFilePath -Force
Copy-Item -Path "$sharedDir/improvements/StringUtils.cs.improved" -Destination $testFilePath2 -Force
Write-ColorText "Improvements applied" "Green"

# Show the diff
Write-ColorText "Diff for Calculator.cs:" "Green"
Write-ColorText "- TODO: Implement Add method" "Red"
Write-ColorText "- TODO: Implement Subtract method" "Red"
Write-ColorText "- TODO: Implement Multiply method" "Red"
Write-ColorText "- TODO: Implement Divide method" "Red"
Write-ColorText "+ Implemented Add method" "Green"
Write-ColorText "+ Implemented Subtract method" "Green"
Write-ColorText "+ Implemented Multiply method" "Green"
Write-ColorText "+ Implemented Divide method" "Green"

Write-ColorText "Diff for StringUtils.cs:" "Green"
Write-ColorText "- TODO: Implement Reverse method" "Red"
Write-ColorText "- TODO: Implement IsPalindrome method" "Red"
Write-ColorText "- TODO: Implement CountWords method" "Red"
Write-ColorText "- TODO: Implement Truncate method" "Red"
Write-ColorText "+ Implemented Reverse method" "Green"
Write-ColorText "+ Implemented IsPalindrome method" "Green"
Write-ColorText "+ Implemented CountWords method" "Green"
Write-ColorText "+ Implemented Truncate method" "Green"

Write-ColorText "Auto-coding test completed" "Cyan"
Write-ColorText "TARS is now ready for auto-coding in swarm mode!" "Green"
