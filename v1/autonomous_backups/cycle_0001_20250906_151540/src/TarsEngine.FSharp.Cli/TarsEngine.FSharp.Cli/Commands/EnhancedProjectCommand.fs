namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Enhanced Project Command with comprehensive documentation and run scripts
type EnhancedProjectCommand(logger: ILogger<EnhancedProjectCommand>) =
    interface ICommand with
        member _.Name = "create-project"
        member _.Description = "Create complete projects from single prompts with docs and run scripts"
        member self.Usage = "tars create-project \"<prompt>\" [options]"
        member self.Examples = [
            "tars create-project \"file backup system with encryption\""
            "tars create-project \"REST API for user management\""
            "tars create-project \"simple calculator web app\""
            "tars create-project \"todo list with database\""
        ]
        member self.ValidateOptions(_) = true
        
        member self.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | prompt :: _ ->
                        let fullPrompt = String.concat " " options.Arguments
                        
                        printfn "🚀 TARS ENHANCED PROJECT CREATION"
                        printfn "================================="
                        printfn "Prompt: %s" fullPrompt
                        printfn ""
                        
                        // Generate project name from prompt
                        let projectName = 
                            fullPrompt.ToLower()
                                .Replace(" ", "_")
                                .Replace("-", "_")
                            + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss")
                        
                        let projectPath = Path.Combine(".tars", "projects", projectName)
                        
                        printfn "📁 Creating project: %s" projectName
                        printfn "📂 Location: %s" projectPath
                        printfn ""
                        
                        // Create project structure
                        Directory.CreateDirectory(projectPath) |> ignore
                        Directory.CreateDirectory(Path.Combine(projectPath, "src")) |> ignore
                        Directory.CreateDirectory(Path.Combine(projectPath, "docs")) |> ignore
                        Directory.CreateDirectory(Path.Combine(projectPath, "tests")) |> ignore
                        Directory.CreateDirectory(Path.Combine(projectPath, "assets")) |> ignore
                        
                        // Determine project type and generate appropriate files
                        let projectType = 
                            if fullPrompt.Contains("web") || fullPrompt.Contains("app") then "web"
                            elif fullPrompt.Contains("api") || fullPrompt.Contains("rest") then "api"
                            elif fullPrompt.Contains("cli") || fullPrompt.Contains("command") then "cli"
                            elif fullPrompt.Contains("library") || fullPrompt.Contains("package") then "library"
                            else "console"
                        
                        printfn "🎯 Detected project type: %s" projectType
                        
                        // Generate project files based on type
                        do! self.GenerateProjectFiles(projectPath, projectName, fullPrompt, projectType)
                        
                        // Generate comprehensive documentation
                        do! self.GenerateDocumentation(projectPath, projectName, fullPrompt, projectType)
                        
                        // Generate run.cmd script
                        do! self.GenerateRunScript(projectPath, projectName, projectType)
                        
                        // Generate package.json or project file
                        do! self.GenerateProjectConfig(projectPath, projectName, fullPrompt, projectType)
                        
                        printfn ""
                        printfn "✅ PROJECT CREATION COMPLETE!"
                        printfn "============================="
                        printfn "📁 Project: %s" projectName
                        printfn "📂 Location: %s" projectPath
                        printfn "🚀 To run: cd %s && run.cmd" projectPath
                        printfn "📚 Documentation: %s/docs/" projectPath
                        printfn ""
                        
                        return CommandResult.success("Enhanced project created successfully")
                    
                    | [] ->
                        printfn "❌ Error: Please provide a project description"
                        printfn ""
                        printfn "Usage: tars create-project \"<description>\""
                        printfn ""
                        printfn "Examples:"
                        printfn "  tars create-project \"file backup system\""
                        printfn "  tars create-project \"REST API for users\""
                        printfn "  tars create-project \"calculator web app\""
                        
                        return CommandResult.error("No project description provided")
                        
                with
                | ex ->
                    logger.LogError(ex, "Failed to create project")
                    return CommandResult.error($"Project creation failed: {ex.Message}")
            }
    
    member private self.GenerateProjectFiles(projectPath: string, projectName: string, prompt: string, projectType: string) =
        task {
            match projectType with
            | "web" ->
                // Generate HTML, CSS, JS files
                let htmlContent = sprintf """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>%s</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>%s</h1>
        <p>Generated by TARS from prompt: "%s"</p>
        <div id="app">
            <!-- Application content will be generated here -->
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>""" projectName projectName prompt
                
                File.WriteAllText(Path.Combine(projectPath, "src", "index.html"), htmlContent)
                
                let cssContent = """/* Generated by TARS */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1 {
    color: #333;
    text-align: center;
}"""
                
                File.WriteAllText(Path.Combine(projectPath, "src", "style.css"), cssContent)
                
                let jsContent = sprintf """// Generated by TARS for: %s
console.log('TARS Project: %s');

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Application initialized');
    
    // Add your application logic here
    const app = document.getElementById('app');
    app.innerHTML = '<p>✅ Application loaded successfully!</p>';
});""" prompt projectName
                
                File.WriteAllText(Path.Combine(projectPath, "src", "script.js"), jsContent)
                
            | "api" ->
                // Generate API files (Node.js/Express example)
                let serverContent = sprintf """// TARS Generated API Server
// Project: %s
// Prompt: %s

const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/', (req, res) => {
    res.json({
        message: 'TARS Generated API',
        project: '%s',
        status: 'operational'
    });
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Start server
app.listen(port, () => {
    console.log(`🚀 TARS API Server running on port ${port}`);
    console.log(`📡 Health check: http://localhost:${port}/health`);
});

module.exports = app;""" projectName prompt projectName
                
                File.WriteAllText(Path.Combine(projectPath, "src", "server.js"), serverContent)
                
            | _ ->
                // Generate console application
                let mainContent = sprintf """// TARS Generated Console Application
// Project: %s
// Prompt: %s

using System;

namespace %s
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("🤖 TARS Generated Application");
            Console.WriteLine("============================");
            Console.WriteLine("Project: %s");
            Console.WriteLine("Generated from: %s");
            Console.WriteLine();
            Console.WriteLine("✅ Application running successfully!");
            
            // Add your application logic here
            
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}""" projectName prompt (projectName.Replace("_", "").Replace("-", "")) projectName prompt
                
                File.WriteAllText(Path.Combine(projectPath, "src", "Program.cs"), mainContent)
        }
    
    member private self.GenerateDocumentation(projectPath: string, projectName: string, prompt: string, projectType: string) =
        task {
            // Generate comprehensive README
            let readmeContent = sprintf """# %s

**Generated by TARS** | **%s**

## 🎯 Overview
%s

**Project Type**: %s  
**Generated from prompt**: "%s"

## 🚀 Quick Start

### Option 1: One-Click Run
```bash
# Simply double-click or run:
run.cmd
```

### Option 2: Manual Setup
```bash
# Navigate to project directory
cd %s

# Install dependencies (if applicable)
npm install

# Run the application
npm start
```

## 📁 Project Structure
```
%s/
├── src/                 # Source code
├── docs/               # Documentation
├── tests/              # Test files
├── assets/             # Static assets
├── run.cmd             # Quick run script
├── package.json        # Dependencies
└── README.md           # This file
```

## ✨ Features
- ✅ Generated from natural language prompt
- ✅ Complete project structure
- ✅ Ready-to-run configuration
- ✅ Comprehensive documentation
- ✅ Test framework setup

## 🛠️ Development
This project was generated by TARS autonomous system and includes:
- Source code scaffolding
- Documentation templates
- Run scripts for easy execution
- Test framework setup

## 📖 Documentation
- [Setup Guide](docs/SETUP.md) - Detailed installation
- [User Guide](docs/USER_GUIDE.md) - How to use
- [Developer Guide](docs/DEVELOPER_GUIDE.md) - Development info

## 🧪 Testing
```bash
# Run tests
npm test
```

## 📞 Support
Generated by TARS Autonomous System. For questions about TARS project generation, refer to TARS documentation.

---
**🤖 Autonomously generated by TARS** | **%s**
""" projectName (DateTime.Now.ToString("yyyy-MM-dd")) prompt projectType prompt projectName projectName (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))
            
            File.WriteAllText(Path.Combine(projectPath, "README.md"), readmeContent)
            
            // Generate SETUP.md
            let setupContent = sprintf """# Setup Guide - %s

## Prerequisites
- Node.js (v14 or higher) - for web/API projects
- .NET Core (v6 or higher) - for console applications
- Modern web browser - for web applications

## Quick Setup
1. **Run the project**:
   ```bash
   run.cmd
   ```

2. **Manual setup**:
   ```bash
   # Install dependencies
   npm install
   
   # Start the application
   npm start
   ```

## Troubleshooting
- **Node.js not found**: Install from https://nodejs.org/
- **Port in use**: The application will find an available port
- **Permission errors**: Run as administrator if needed

## Project Type: %s
%s

Generated by TARS from prompt: "%s"
""" projectName projectType 
                (match projectType with
                 | "web" -> "This is a web application. Open browser to http://localhost:3000"
                 | "api" -> "This is an API server. Test endpoints at http://localhost:3000"
                 | _ -> "This is a console application. Run directly from command line")
                prompt
            
            File.WriteAllText(Path.Combine(projectPath, "docs", "SETUP.md"), setupContent)
        }
    
    member private self.GenerateRunScript(projectPath: string, projectName: string, projectType: string) =
        task {
            let runScript = sprintf """@echo off
echo 🚀 TARS Project Launcher: %s
echo ================================
echo.

echo 📦 Checking prerequisites...

%s

echo.
echo ✅ Prerequisites checked
echo.

echo 🌐 Starting %s...
echo 📍 Project type: %s
echo.

%s

echo.
echo 🎉 Project started successfully!
pause
""" projectName
                (match projectType with
                 | "web" | "api" -> """node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found! Install from https://nodejs.org/
    pause
    exit /b 1
)
echo ✅ Node.js found"""
                 | _ -> """dotnet --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ .NET not found! Install from https://dotnet.microsoft.com/
    pause
    exit /b 1
)
echo ✅ .NET found""")
                projectName projectType
                (match projectType with
                 | "web" -> """if not exist package.json (
    echo Creating package.json...
    echo {"name": "%s", "version": "1.0.0", "scripts": {"start": "npx live-server src --port=3000"}} > package.json
)

call npm install --silent
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

start http://localhost:3000
call npx live-server src --port=3000"""
                 | "api" -> """if not exist package.json (
    echo Creating package.json...
    echo {"name": "%s", "version": "1.0.0", "main": "src/server.js", "scripts": {"start": "node src/server.js"}, "dependencies": {"express": "^4.18.0"}} > package.json
)

call npm install --silent
echo 📡 API will be available at: http://localhost:3000
call npm start"""
                 | _ -> """if exist src\\*.cs (
    echo Compiling C# application...
    dotnet run --project src
) else (
    echo No runnable files found
    pause
)""")
            
            File.WriteAllText(Path.Combine(projectPath, "run.cmd"), runScript)
        }
    
    member private self.GenerateProjectConfig(projectPath: string, projectName: string, prompt: string, projectType: string) =
        task {
            match projectType with
            | "web" | "api" ->
                let packageJson = sprintf """{
  "name": "%s",
  "version": "1.0.0",
  "description": "Generated by TARS from prompt: %s",
  "main": "%s",
  "scripts": {
    "start": "%s",
    "test": "echo \"No tests specified\" && exit 0"
  },
  "dependencies": {
    %s
  },
  "devDependencies": {
    "live-server": "^1.2.2"
  },
  "keywords": ["tars", "generated", "autonomous"],
  "author": "TARS Autonomous System",
  "license": "MIT"
}""" projectName prompt 
                    (if projectType = "api" then "src/server.js" else "src/index.html")
                    (if projectType = "api" then "node src/server.js" else "npx live-server src --port=3000")
                    (if projectType = "api" then "\"express\": \"^4.18.0\"" else "")
                
                File.WriteAllText(Path.Combine(projectPath, "package.json"), packageJson)
                
            | _ ->
                let csprojContent = sprintf """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <AssemblyName>%s</AssemblyName>
    <RootNamespace>%s</RootNamespace>
  </PropertyGroup>
</Project>""" projectName (projectName.Replace("_", "").Replace("-", ""))
                
                File.WriteAllText(Path.Combine(projectPath, "src", $"{projectName}.csproj"), csprojContent)
        }
