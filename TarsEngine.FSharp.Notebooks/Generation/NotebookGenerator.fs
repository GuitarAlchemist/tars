namespace TarsEngine.FSharp.Notebooks.Generation

open System
open System.IO
open TarsEngine.FSharp.Notebooks.Types
open TarsEngine.FSharp.Notebooks.Generation.MetascriptAnalyzer

/// <summary>
/// Generates Jupyter notebooks from TARS metascript analysis
/// </summary>
module NotebookGenerator =
    
    /// Generate notebook from metascript analysis
    let generateNotebook (analysis: MetascriptAnalysis) (strategy: NotebookGenerationStrategy) : Async<JupyterNotebook> = async {
        try
            let requirements = generateNotebookRequirements analysis strategy
            
            // Create kernel specification
            let kernelSpec = createKernelSpecification requirements.KernelType
            
            // Create language information
            let languageInfo = createLanguageInformation requirements.KernelType
            
            // Create metadata
            let metadata = {
                KernelSpec = Some kernelSpec
                LanguageInfo = Some languageInfo
                Title = Some requirements.Title
                Authors = ["TARS Notebook Generator"]
                Created = Some DateTime.UtcNow
                Modified = Some DateTime.UtcNow
                Custom = Map [
                    ("tars_metascript", box analysis.FilePath)
                    ("generation_strategy", box (strategy.ToString()))
                    ("complexity_score", box analysis.Complexity.ComplexityScore)
                ]
            }
            
            // Generate cells based on strategy
            let cells = generateCells analysis strategy requirements
            
            let notebook = {
                Metadata = metadata
                Cells = cells
                NbFormat = 4
                NbFormatMinor = 5
            }
            
            return notebook
            
        with
        | ex ->
            failwith $"Failed to generate notebook: {ex.Message}"
    }
    
    /// Create kernel specification from kernel type
    let private createKernelSpecification (kernelType: SupportedKernel) : KernelSpecification =
        match kernelType with
        | Python config ->
            {
                Name = "python3"
                DisplayName = $"Python {config.Version}"
                Language = "python"
                InterruptMode = Some "signal"
                Env = Map.empty
                Argv = ["python"; "-m"; "ipykernel_launcher"; "-f"; "{connection_file}"]
            }
        | FSharp config ->
            {
                Name = "fsharp"
                DisplayName = $"F# {config.DotNetVersion}"
                Language = "fsharp"
                InterruptMode = Some "message"
                Env = Map.empty
                Argv = ["dotnet"; "fsi"; "--kernel"]
            }
        | CSharp config ->
            {
                Name = "csharp"
                DisplayName = $"C# {config.DotNetVersion}"
                Language = "csharp"
                InterruptMode = Some "message"
                Env = Map.empty
                Argv = ["dotnet"; "interactive"; "csharp"]
            }
        | JavaScript config ->
            {
                Name = "javascript"
                DisplayName = $"JavaScript (Node.js {config.NodeVersion})"
                Language = "javascript"
                InterruptMode = Some "signal"
                Env = Map.empty
                Argv = ["node"; "--kernel"]
            }
        | SQL config ->
            {
                Name = "sql"
                DisplayName = $"SQL ({config.DatabaseType})"
                Language = "sql"
                InterruptMode = None
                Env = Map [("CONNECTION_STRING", config.ConnectionString)]
                Argv = ["sql-kernel"]
            }
        | R config ->
            {
                Name = "ir"
                DisplayName = $"R {config.Version}"
                Language = "R"
                InterruptMode = Some "signal"
                Env = Map.empty
                Argv = ["R"; "--slave"; "-e"; "IRkernel::main()"; "--args"; "{connection_file}"]
            }
        | Custom config ->
            {
                Name = config.Name.ToLower()
                DisplayName = config.Name
                Language = config.Name.ToLower()
                InterruptMode = None
                Env = config.Environment
                Argv = config.Command :: config.Args
            }
    
    /// Create language information from kernel type
    let private createLanguageInformation (kernelType: SupportedKernel) : LanguageInformation =
        match kernelType with
        | Python config ->
            {
                Name = "python"
                Version = config.Version
                MimeType = Some "text/x-python"
                FileExtension = Some ".py"
                PygmentsLexer = Some "ipython3"
                CodeMirrorMode = Some "python"
            }
        | FSharp config ->
            {
                Name = "fsharp"
                Version = config.DotNetVersion
                MimeType = Some "text/x-fsharp"
                FileExtension = Some ".fs"
                PygmentsLexer = Some "fsharp"
                CodeMirrorMode = Some "fsharp"
            }
        | CSharp config ->
            {
                Name = "csharp"
                Version = config.DotNetVersion
                MimeType = Some "text/x-csharp"
                FileExtension = Some ".cs"
                PygmentsLexer = Some "csharp"
                CodeMirrorMode = Some "csharp"
            }
        | JavaScript config ->
            {
                Name = "javascript"
                Version = config.NodeVersion
                MimeType = Some "application/javascript"
                FileExtension = Some ".js"
                PygmentsLexer = Some "javascript"
                CodeMirrorMode = Some "javascript"
            }
        | SQL config ->
            {
                Name = "sql"
                Version = "1.0"
                MimeType = Some "text/x-sql"
                FileExtension = Some ".sql"
                PygmentsLexer = Some "sql"
                CodeMirrorMode = Some "sql"
            }
        | R config ->
            {
                Name = "R"
                Version = config.Version
                MimeType = Some "text/x-r"
                FileExtension = Some ".r"
                PygmentsLexer = Some "r"
                CodeMirrorMode = Some "r"
            }
        | Custom config ->
            {
                Name = config.Name.ToLower()
                Version = "1.0"
                MimeType = None
                FileExtension = None
                PygmentsLexer = None
                CodeMirrorMode = None
            }
    
    /// Generate cells based on strategy
    let private generateCells (analysis: MetascriptAnalysis) (strategy: NotebookGenerationStrategy) (requirements: NotebookRequirements) : NotebookCell list =
        let cells = ResizeArray<NotebookCell>()
        
        // Add title cell
        cells.Add(createTitleCell requirements.Title requirements.Description)
        
        // Add imports cell
        cells.Add(createImportsCell requirements.KernelType)
        
        // Add strategy-specific cells
        match strategy with
        | ExploratoryDataAnalysis ->
            cells.AddRange(generateEDAcells analysis)
        | MachineLearningPipeline ->
            cells.AddRange(generateMLCells analysis)
        | ResearchNotebook ->
            cells.AddRange(generateResearchCells analysis)
        | TutorialNotebook ->
            cells.AddRange(generateTutorialCells analysis)
        | DocumentationNotebook ->
            cells.AddRange(generateDocumentationCells analysis)
        | BusinessReport ->
            cells.AddRange(generateBusinessReportCells analysis)
        | AcademicPaper ->
            cells.AddRange(generateAcademicPaperCells analysis)
        
        // Add conclusion cell
        cells.Add(createConclusionCell analysis)
        
        cells |> List.ofSeq
    
    /// Create title cell
    let private createTitleCell (title: string) (description: string) : NotebookCell =
        let source = [
            $"# {title}"
            ""
            $"**Description:** {description}"
            ""
            $"**Generated:** {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC"
            ""
            "---"
        ]
        
        MarkdownCell {
            Source = source
            Metadata = Map [("tags", box ["title"; "header"])]
        }
    
    /// Create imports cell
    let private createImportsCell (kernelType: SupportedKernel) : NotebookCell =
        let source = 
            match kernelType with
            | Python _ -> [
                "# Import required libraries"
                "import pandas as pd"
                "import numpy as np"
                "import matplotlib.pyplot as plt"
                "import seaborn as sns"
                "from datetime import datetime"
                ""
                "# Configure plotting"
                "plt.style.use('default')"
                "sns.set_palette('husl')"
                ""
                "print('Libraries imported successfully')"
            ]
            | FSharp _ -> [
                "// Import required libraries"
                "#r \"nuget: FSharp.Data\""
                "#r \"nuget: Plotly.NET\""
                ""
                "open System"
                "open FSharp.Data"
                "open Plotly.NET"
                ""
                "printfn \"Libraries imported successfully\""
            ]
            | CSharp _ -> [
                "// Import required libraries"
                "#r \"nuget: CsvHelper\""
                "#r \"nuget: Plotly.NET.CSharp\""
                ""
                "using System;"
                "using System.Linq;"
                "using CsvHelper;"
                "using Plotly.NET.CSharp;"
                ""
                "Console.WriteLine(\"Libraries imported successfully\");"
            ]
            | JavaScript _ -> [
                "// Import required libraries"
                "const fs = require('fs');"
                "const path = require('path');"
                ""
                "console.log('Libraries imported successfully');"
            ]
            | SQL _ -> [
                "-- SQL Analysis Setup"
                "-- Database connection established"
                "SELECT 'Libraries imported successfully' as status;"
            ]
            | R _ -> [
                "# Import required libraries"
                "library(dplyr)"
                "library(ggplot2)"
                "library(readr)"
                ""
                "cat('Libraries imported successfully\\n')"
            ]
            | Custom _ -> [
                "# Custom kernel setup"
                "print('Custom kernel initialized')"
            ]
        
        CodeCell {
            Source = source
            Language = getLanguageName kernelType
            Outputs = []
            ExecutionCount = None
            Metadata = Map [("tags", box ["imports"; "setup"])]
        }
    
    /// Get language name from kernel type
    let private getLanguageName (kernelType: SupportedKernel) : string =
        match kernelType with
        | Python _ -> "python"
        | FSharp _ -> "fsharp"
        | CSharp _ -> "csharp"
        | JavaScript _ -> "javascript"
        | SQL _ -> "sql"
        | R _ -> "r"
        | Custom config -> config.Name.ToLower()
    
    /// Generate EDA cells
    let private generateEDAcells (analysis: MetascriptAnalysis) : NotebookCell list =
        [
            MarkdownCell {
                Source = [
                    "## Exploratory Data Analysis"
                    ""
                    "This section performs exploratory data analysis on the data sources identified in the metascript."
                ]
                Metadata = Map [("tags", box ["eda"; "analysis"])]
            }
            
            CodeCell {
                Source = [
                    "# Load and examine data"
                    "# TODO: Replace with actual data loading based on metascript data sources"
                    "print('Data loading implementation needed')"
                    ""
                    "# Example data loading:"
                    "# df = pd.read_csv('data.csv')"
                    "# print(f'Data shape: {df.shape}')"
                    "# print(df.head())"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["data-loading"])]
            }
            
            CodeCell {
                Source = [
                    "# Data summary and statistics"
                    "# df.describe()"
                    "# df.info()"
                    "print('Data summary implementation needed')"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["data-summary"])]
            }
        ]
    
    /// Generate ML cells
    let private generateMLCells (analysis: MetascriptAnalysis) : NotebookCell list =
        [
            MarkdownCell {
                Source = [
                    "## Machine Learning Pipeline"
                    ""
                    "This section implements a machine learning pipeline based on the metascript analysis."
                ]
                Metadata = Map [("tags", box ["ml"; "pipeline"])]
            }
            
            CodeCell {
                Source = [
                    "# Data preprocessing"
                    "from sklearn.model_selection import train_test_split"
                    "from sklearn.preprocessing import StandardScaler"
                    ""
                    "# TODO: Implement preprocessing based on metascript requirements"
                    "print('ML preprocessing implementation needed')"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["preprocessing"])]
            }
            
            CodeCell {
                Source = [
                    "# Model training"
                    "from sklearn.ensemble import RandomForestClassifier"
                    "from sklearn.metrics import classification_report"
                    ""
                    "# TODO: Implement model training based on metascript agents"
                    "print('Model training implementation needed')"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["training"])]
            }
        ]
    
    /// Generate research cells
    let private generateResearchCells (analysis: MetascriptAnalysis) : NotebookCell list =
        [
            MarkdownCell {
                Source = [
                    "## Research Analysis"
                    ""
                    "This notebook supports academic research based on the metascript analysis."
                    ""
                    $"**Research Objective:** {analysis.Narrative.Objective}"
                ]
                Metadata = Map [("tags", box ["research"; "academic"])]
            }
            
            CodeCell {
                Source = [
                    "# Research data analysis"
                    "# TODO: Implement research-specific analysis"
                    "print('Research analysis implementation needed')"
                    ""
                    "# Statistical analysis"
                    "# from scipy import stats"
                    "# Statistical tests and analysis here"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["research-analysis"])]
            }
        ]
    
    /// Generate tutorial cells
    let private generateTutorialCells (analysis: MetascriptAnalysis) : NotebookCell list =
        [
            MarkdownCell {
                Source = [
                    "## Tutorial: Understanding TARS Metascripts"
                    ""
                    "This tutorial explains the concepts and implementation of TARS metascripts."
                ]
                Metadata = Map [("tags", box ["tutorial"; "education"])]
            }
            
            CodeCell {
                Source = [
                    "# Tutorial examples"
                    "print('This is a tutorial notebook generated from TARS metascript')"
                    "print('Follow along to learn about automated analysis')"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["tutorial-example"])]
            }
        ]
    
    /// Generate documentation cells
    let private generateDocumentationCells (analysis: MetascriptAnalysis) : NotebookCell list =
        [
            MarkdownCell {
                Source = [
                    "## Documentation"
                    ""
                    "This notebook documents the metascript implementation and usage."
                    ""
                    "### Agents"
                    ""
                ] @ (analysis.Agents |> List.map (fun agent -> $"- **{agent.Name}** ({agent.Type}): {agent.Description |> Option.defaultValue \"No description\"}"))
                Metadata = Map [("tags", box ["documentation"])]
            }
        ]
    
    /// Generate business report cells
    let private generateBusinessReportCells (analysis: MetascriptAnalysis) : NotebookCell list =
        [
            MarkdownCell {
                Source = [
                    "## Business Report"
                    ""
                    "This report provides business insights based on the metascript analysis."
                ]
                Metadata = Map [("tags", box ["business"; "report"])]
            }
            
            CodeCell {
                Source = [
                    "# Business metrics and KPIs"
                    "# TODO: Implement business-specific analysis"
                    "print('Business analysis implementation needed')"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["business-metrics"])]
            }
        ]
    
    /// Generate academic paper cells
    let private generateAcademicPaperCells (analysis: MetascriptAnalysis) : NotebookCell list =
        [
            MarkdownCell {
                Source = [
                    "## Academic Paper Analysis"
                    ""
                    "This notebook supports academic paper writing and research validation."
                ]
                Metadata = Map [("tags", box ["academic"; "paper"])]
            }
            
            CodeCell {
                Source = [
                    "# Academic analysis and validation"
                    "# TODO: Implement academic-specific analysis"
                    "print('Academic analysis implementation needed')"
                ]
                Language = "python"
                Outputs = []
                ExecutionCount = None
                Metadata = Map [("tags", box ["academic-analysis"])]
            }
        ]
    
    /// Create conclusion cell
    let private createConclusionCell (analysis: MetascriptAnalysis) : NotebookCell =
        let source = [
            "## Conclusion"
            ""
            $"This notebook was automatically generated from the TARS metascript: `{Path.GetFileName(analysis.FilePath)}`"
            ""
            "### Summary"
            $"- **Agents:** {analysis.Complexity.AgentCount}"
            $"- **Actions:** {analysis.Complexity.ActionCount}"
            $"- **Variables:** {analysis.Complexity.VariableCount}"
            $"- **Complexity Score:** {analysis.Complexity.ComplexityScore:F1}/100"
            $"- **Estimated Execution Time:** {analysis.Complexity.EstimatedExecutionTime}"
            ""
            "### Next Steps"
            "1. Customize the generated code cells with your specific data and requirements"
            "2. Execute the notebook cells in sequence"
            "3. Analyze the results and iterate as needed"
            ""
            "---"
            $"*Generated by TARS Notebook Generator on {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC*"
        ]
        
        MarkdownCell {
            Source = source
            Metadata = Map [("tags", box ["conclusion"; "summary"])]
        }
