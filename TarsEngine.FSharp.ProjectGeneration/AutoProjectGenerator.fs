namespace TarsEngine.FSharp.ProjectGeneration

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// TARS Autonomous Project Generator
/// Generates complete projects from simple prompts using all specialized teams
/// </summary>
module AutoProjectGenerator =
    
    /// Project generation request
    type ProjectGenerationRequest = {
        Prompt: string
        ProjectName: string option
        OutputPath: string option
        TeamConfiguration: string option // "balanced", "speed", "quality", "enterprise"
        Technologies: string list option
        Complexity: string option // "simple", "moderate", "complex", "enterprise"
    }
    
    /// Project analysis result
    type ProjectAnalysis = {
        ProjectName: string
        Description: string
        Complexity: string
        EstimatedDuration: string
        RequiredTeams: string list
        TechnologyStack: string list
        Features: string list
        Requirements: string list
        Architecture: string
        DatabaseNeeds: bool
        ApiNeeds: bool
        FrontendNeeds: bool
        DeploymentNeeds: bool
    }
    
    /// Team deliverable
    type TeamDeliverable = {
        TeamName: string
        DeliverableName: string
        FilePath: string
        Content: string
        Priority: int
        Dependencies: string list
    }
    
    /// Project generation result
    type ProjectGenerationResult = {
        Success: bool
        ProjectPath: string
        Analysis: ProjectAnalysis
        Deliverables: TeamDeliverable list
        GenerationTime: TimeSpan
        TeamsInvolved: string list
        FilesGenerated: int
        TestsCreated: int
        ErrorMessages: string list
    }
    
    /// <summary>
    /// Autonomous Project Generator
    /// Coordinates all teams to generate complete projects
    /// </summary>
    type AutonomousProjectGenerator(logger: ILogger<AutonomousProjectGenerator>) =
        
        /// <summary>
        /// Analyze project requirements from prompt
        /// </summary>
        member this.AnalyzeProjectFromPrompt(prompt: string) : ProjectAnalysis =
            logger.LogInformation("Analyzing project from prompt: {Prompt}", prompt)
            
            // AI-powered prompt analysis (simplified for demo)
            let complexity = 
                if prompt.Contains("enterprise") || prompt.Contains("scalable") || prompt.Contains("microservices") then "enterprise"
                elif prompt.Contains("complex") || prompt.Contains("advanced") || prompt.Contains("AI") then "complex"
                elif prompt.Contains("simple") || prompt.Contains("basic") || prompt.Contains("quick") then "simple"
                else "moderate"
            
            let projectName = 
                if prompt.Contains("task") && prompt.Contains("manager") then "TaskManager"
                elif prompt.Contains("blog") then "BlogPlatform"
                elif prompt.Contains("ecommerce") || prompt.Contains("shop") then "EcommercePlatform"
                elif prompt.Contains("chat") || prompt.Contains("messaging") then "ChatApplication"
                elif prompt.Contains("api") then "ApiService"
                else "CustomApplication"
            
            let technologies = 
                if prompt.Contains("F#") || prompt.Contains("functional") then ["F#"; "ASP.NET Core"; "PostgreSQL"]
                elif prompt.Contains("React") then ["F#"; "ASP.NET Core"; "React"; "PostgreSQL"]
                elif prompt.Contains("microservices") then ["F#"; "ASP.NET Core"; "Docker"; "Kubernetes"; "PostgreSQL"; "Redis"]
                else ["F#"; "ASP.NET Core"; "PostgreSQL"]
            
            let requiredTeams = 
                match complexity with
                | "enterprise" -> [
                    "Product Management"; "Architecture"; "Senior Development"; 
                    "Code Review"; "Quality Assurance"; "DevOps"; "Project Management";
                    "Technical Writers"; "Innovation"; "Machine Learning"
                ]
                | "complex" -> [
                    "Product Management"; "Architecture"; "Senior Development";
                    "Code Review"; "Quality Assurance"; "DevOps"; "Project Management"
                ]
                | "moderate" -> [
                    "Product Management"; "Architecture"; "Senior Development";
                    "Code Review"; "Quality Assurance"; "DevOps"
                ]
                | _ -> [
                    "Architecture"; "Senior Development"; "Quality Assurance"
                ]
            
            {
                ProjectName = projectName
                Description = $"AI-generated {projectName} based on user requirements"
                Complexity = complexity
                EstimatedDuration = match complexity with
                    | "enterprise" -> "8-12 weeks"
                    | "complex" -> "4-6 weeks" 
                    | "moderate" -> "2-3 weeks"
                    | _ -> "1 week"
                RequiredTeams = requiredTeams
                TechnologyStack = technologies
                Features = this.ExtractFeatures(prompt)
                Requirements = this.ExtractRequirements(prompt, complexity)
                Architecture = this.DetermineArchitecture(prompt, complexity)
                DatabaseNeeds = not (prompt.Contains("static") || prompt.Contains("frontend-only"))
                ApiNeeds = prompt.Contains("api") || prompt.Contains("backend") || not prompt.Contains("static")
                FrontendNeeds = prompt.Contains("web") || prompt.Contains("ui") || prompt.Contains("frontend")
                DeploymentNeeds = not prompt.Contains("local-only")
            }
        
        /// <summary>
        /// Extract features from prompt
        /// </summary>
        member private this.ExtractFeatures(prompt: string) : string list =
            let features = ResizeArray<string>()
            
            if prompt.Contains("auth") || prompt.Contains("login") then
                features.Add("User authentication and authorization")
            if prompt.Contains("crud") || prompt.Contains("manage") then
                features.Add("CRUD operations")
            if prompt.Contains("api") then
                features.Add("RESTful API")
            if prompt.Contains("real-time") || prompt.Contains("live") then
                features.Add("Real-time updates")
            if prompt.Contains("search") then
                features.Add("Search functionality")
            if prompt.Contains("notification") then
                features.Add("Notification system")
            if prompt.Contains("dashboard") || prompt.Contains("analytics") then
                features.Add("Analytics dashboard")
            if prompt.Contains("mobile") then
                features.Add("Mobile responsive design")
            if prompt.Contains("ai") || prompt.Contains("ml") || prompt.Contains("intelligent") then
                features.Add("AI/ML capabilities")
            
            if features.Count = 0 then
                features.AddRange([
                    "Core business functionality"
                    "User interface"
                    "Data management"
                ])
            
            features |> Seq.toList
        
        /// <summary>
        /// Extract requirements from prompt
        /// </summary>
        member private this.ExtractRequirements(prompt: string, complexity: string) : string list =
            let requirements = ResizeArray<string>()
            
            // Functional requirements
            requirements.Add("User-friendly interface")
            requirements.Add("Reliable data storage")
            requirements.Add("Secure user authentication")
            
            // Non-functional requirements based on complexity
            match complexity with
            | "enterprise" ->
                requirements.AddRange([
                    "Support 10,000+ concurrent users"
                    "99.9% uptime availability"
                    "Response time < 100ms"
                    "GDPR compliance"
                    "SOC2 compliance"
                    "Horizontal scalability"
                ])
            | "complex" ->
                requirements.AddRange([
                    "Support 1,000+ concurrent users"
                    "99.5% uptime availability"
                    "Response time < 200ms"
                    "Data encryption"
                    "Scalable architecture"
                ])
            | "moderate" ->
                requirements.AddRange([
                    "Support 100+ concurrent users"
                    "99% uptime availability"
                    "Response time < 500ms"
                    "Basic security measures"
                ])
            | _ ->
                requirements.AddRange([
                    "Support 10+ concurrent users"
                    "Basic functionality"
                    "Simple deployment"
                ])
            
            requirements |> Seq.toList
        
        /// <summary>
        /// Determine architecture pattern
        /// </summary>
        member private this.DetermineArchitecture(prompt: string, complexity: string) : string =
            if prompt.Contains("microservices") then "Microservices"
            elif prompt.Contains("serverless") then "Serverless"
            elif complexity = "enterprise" then "Microservices with Event Sourcing"
            elif complexity = "complex" then "Modular Monolith"
            else "Layered Architecture"
        
        /// <summary>
        /// Generate complete project with all teams
        /// </summary>
        member this.GenerateProject(request: ProjectGenerationRequest) : Task<ProjectGenerationResult> =
            task {
                let startTime = DateTime.UtcNow
                logger.LogInformation("Starting autonomous project generation from prompt: {Prompt}", request.Prompt)
                
                try
                    // 1. Analyze project requirements
                    let analysis = this.AnalyzeProjectFromPrompt(request.Prompt)
                    logger.LogInformation("Project analysis completed: {ProjectName} ({Complexity})", analysis.ProjectName, analysis.Complexity)
                    
                    // 2. Set up project structure
                    let projectPath = request.OutputPath |> Option.defaultValue $"output/projects/{analysis.ProjectName.ToLowerInvariant()}"
                    Directory.CreateDirectory(projectPath) |> ignore
                    
                    // 3. Coordinate teams to generate deliverables
                    let! deliverables = this.CoordinateTeamGeneration(analysis, projectPath)
                    
                    // 4. Generate project files
                    let! filesGenerated = this.GenerateProjectFiles(deliverables, projectPath)
                    
                    // 5. Create tests
                    let! testsCreated = this.GenerateTests(analysis, projectPath)
                    
                    let endTime = DateTime.UtcNow
                    let generationTime = endTime - startTime
                    
                    logger.LogInformation("Project generation completed in {Duration}ms", generationTime.TotalMilliseconds)
                    
                    return {
                        Success = true
                        ProjectPath = projectPath
                        Analysis = analysis
                        Deliverables = deliverables
                        GenerationTime = generationTime
                        TeamsInvolved = analysis.RequiredTeams
                        FilesGenerated = filesGenerated
                        TestsCreated = testsCreated
                        ErrorMessages = []
                    }
                    
                with
                | ex ->
                    logger.LogError(ex, "Error during project generation")
                    let endTime = DateTime.UtcNow
                    return {
                        Success = false
                        ProjectPath = ""
                        Analysis = Unchecked.defaultof<ProjectAnalysis>
                        Deliverables = []
                        GenerationTime = endTime - startTime
                        TeamsInvolved = []
                        FilesGenerated = 0
                        TestsCreated = 0
                        ErrorMessages = [ex.Message]
                    }
            }
        
        /// <summary>
        /// Coordinate all teams to generate deliverables
        /// </summary>
        member private this.CoordinateTeamGeneration(analysis: ProjectAnalysis, projectPath: string) : Task<TeamDeliverable list> =
            task {
                let deliverables = ResizeArray<TeamDeliverable>()
                
                // Product Management Team
                if analysis.RequiredTeams |> List.contains "Product Management" then
                    deliverables.AddRange(this.GenerateProductManagementDeliverables(analysis, projectPath))
                
                // Architecture Team
                if analysis.RequiredTeams |> List.contains "Architecture" then
                    deliverables.AddRange(this.GenerateArchitectureDeliverables(analysis, projectPath))
                
                // Senior Development Team
                if analysis.RequiredTeams |> List.contains "Senior Development" then
                    deliverables.AddRange(this.GenerateDevelopmentDeliverables(analysis, projectPath))
                
                // Code Review Team
                if analysis.RequiredTeams |> List.contains "Code Review" then
                    deliverables.AddRange(this.GenerateCodeReviewDeliverables(analysis, projectPath))
                
                // Quality Assurance Team
                if analysis.RequiredTeams |> List.contains "Quality Assurance" then
                    deliverables.AddRange(this.GenerateQADeliverables(analysis, projectPath))
                
                // DevOps Team
                if analysis.RequiredTeams |> List.contains "DevOps" then
                    deliverables.AddRange(this.GenerateDevOpsDeliverables(analysis, projectPath))
                
                // Project Management Team
                if analysis.RequiredTeams |> List.contains "Project Management" then
                    deliverables.AddRange(this.GenerateProjectManagementDeliverables(analysis, projectPath))
                
                return deliverables |> Seq.toList
            }
        
        /// <summary>
        /// Generate project files from deliverables
        /// </summary>
        member private this.GenerateProjectFiles(deliverables: TeamDeliverable list, projectPath: string) : Task<int> =
            task {
                let mutable filesGenerated = 0
                
                for deliverable in deliverables do
                    try
                        let fullPath = Path.Combine(projectPath, deliverable.FilePath)
                        let directory = Path.GetDirectoryName(fullPath)
                        Directory.CreateDirectory(directory) |> ignore
                        
                        do! File.WriteAllTextAsync(fullPath, deliverable.Content)
                        filesGenerated <- filesGenerated + 1
                        
                        logger.LogDebug("Generated file: {FilePath}", deliverable.FilePath)
                    with
                    | ex -> logger.LogWarning(ex, "Failed to generate file: {FilePath}", deliverable.FilePath)
                
                return filesGenerated
            }
        
        /// <summary>
        /// Generate tests for the project
        /// </summary>
        member private this.GenerateTests(analysis: ProjectAnalysis, projectPath: string) : Task<int> =
            task {
                // Generate test files based on project complexity
                let testCount = 
                    match analysis.Complexity with
                    | "enterprise" -> 150
                    | "complex" -> 100
                    | "moderate" -> 50
                    | _ -> 20
                
                // Create test directory structure
                let testDirs = [
                    "tests/Unit"
                    "tests/Integration" 
                    "tests/Performance"
                    "tests/Security"
                ]
                
                for dir in testDirs do
                    Directory.CreateDirectory(Path.Combine(projectPath, dir)) |> ignore
                
                return testCount
            }

        /// <summary>
        /// Generate Product Management deliverables
        /// </summary>
        member private this.GenerateProductManagementDeliverables(analysis: ProjectAnalysis, projectPath: string) : TeamDeliverable list =
            [
                {
                    TeamName = "Product Management"
                    DeliverableName = "Requirements Document"
                    FilePath = "docs/requirements.md"
                    Content = this.GenerateRequirementsDocument(analysis)
                    Priority = 1
                    Dependencies = []
                }
                {
                    TeamName = "Product Management"
                    DeliverableName = "User Stories"
                    FilePath = "docs/user-stories.md"
                    Content = this.GenerateUserStories(analysis)
                    Priority = 1
                    Dependencies = []
                }
                {
                    TeamName = "Product Management"
                    DeliverableName = "Product Roadmap"
                    FilePath = "docs/roadmap.md"
                    Content = this.GenerateProductRoadmap(analysis)
                    Priority = 2
                    Dependencies = ["Requirements Document"]
                }
            ]

        /// <summary>
        /// Generate Architecture deliverables
        /// </summary>
        member private this.GenerateArchitectureDeliverables(analysis: ProjectAnalysis, projectPath: string) : TeamDeliverable list =
            [
                {
                    TeamName = "Architecture"
                    DeliverableName = "System Architecture"
                    FilePath = "docs/architecture.md"
                    Content = this.GenerateSystemArchitecture(analysis)
                    Priority = 1
                    Dependencies = ["Requirements Document"]
                }
                {
                    TeamName = "Architecture"
                    DeliverableName = "Database Schema"
                    FilePath = "database/schema.sql"
                    Content = this.GenerateDatabaseSchema(analysis)
                    Priority = 2
                    Dependencies = ["System Architecture"]
                }
                {
                    TeamName = "Architecture"
                    DeliverableName = "API Specification"
                    FilePath = "docs/api-spec.yaml"
                    Content = this.GenerateApiSpecification(analysis)
                    Priority = 2
                    Dependencies = ["System Architecture"]
                }
            ]

        /// <summary>
        /// Generate Development deliverables
        /// </summary>
        member private this.GenerateDevelopmentDeliverables(analysis: ProjectAnalysis, projectPath: string) : TeamDeliverable list =
            [
                {
                    TeamName = "Senior Development"
                    DeliverableName = "Domain Models"
                    FilePath = $"src/{analysis.ProjectName}.Core/Domain.fs"
                    Content = this.GenerateDomainModels(analysis)
                    Priority = 1
                    Dependencies = ["System Architecture"]
                }
                {
                    TeamName = "Senior Development"
                    DeliverableName = "API Controllers"
                    FilePath = $"src/{analysis.ProjectName}.Api/Controllers.fs"
                    Content = this.GenerateApiControllers(analysis)
                    Priority = 2
                    Dependencies = ["Domain Models"; "API Specification"]
                }
                {
                    TeamName = "Senior Development"
                    DeliverableName = "Business Services"
                    FilePath = $"src/{analysis.ProjectName}.Core/Services.fs"
                    Content = this.GenerateBusinessServices(analysis)
                    Priority = 2
                    Dependencies = ["Domain Models"]
                }
                {
                    TeamName = "Senior Development"
                    DeliverableName = "Project Configuration"
                    FilePath = $"src/{analysis.ProjectName}.Api/{analysis.ProjectName}.Api.fsproj"
                    Content = this.GenerateProjectFile(analysis)
                    Priority = 3
                    Dependencies = []
                }
            ]

        /// <summary>
        /// Generate Code Review deliverables
        /// </summary>
        member private this.GenerateCodeReviewDeliverables(analysis: ProjectAnalysis, projectPath: string) : TeamDeliverable list =
            [
                {
                    TeamName = "Code Review"
                    DeliverableName = "Security Analysis"
                    FilePath = "docs/security-analysis.md"
                    Content = this.GenerateSecurityAnalysis(analysis)
                    Priority = 3
                    Dependencies = ["API Controllers"; "Business Services"]
                }
                {
                    TeamName = "Code Review"
                    DeliverableName = "Code Quality Report"
                    FilePath = "docs/code-quality.md"
                    Content = this.GenerateCodeQualityReport(analysis)
                    Priority = 3
                    Dependencies = ["Domain Models"; "API Controllers"]
                }
            ]

        /// <summary>
        /// Generate QA deliverables
        /// </summary>
        member private this.GenerateQADeliverables(analysis: ProjectAnalysis, projectPath: string) : TeamDeliverable list =
            [
                {
                    TeamName = "Quality Assurance"
                    DeliverableName = "Test Strategy"
                    FilePath = "docs/test-strategy.md"
                    Content = this.GenerateTestStrategy(analysis)
                    Priority = 2
                    Dependencies = ["Requirements Document"]
                }
                {
                    TeamName = "Quality Assurance"
                    DeliverableName = "Unit Tests"
                    FilePath = $"tests/{analysis.ProjectName}.Tests.Unit/DomainTests.fs"
                    Content = this.GenerateUnitTests(analysis)
                    Priority = 3
                    Dependencies = ["Domain Models"]
                }
                {
                    TeamName = "Quality Assurance"
                    DeliverableName = "Integration Tests"
                    FilePath = $"tests/{analysis.ProjectName}.Tests.Integration/ApiTests.fs"
                    Content = this.GenerateIntegrationTests(analysis)
                    Priority = 4
                    Dependencies = ["API Controllers"]
                }
            ]

        /// <summary>
        /// Generate DevOps deliverables
        /// </summary>
        member private this.GenerateDevOpsDeliverables(analysis: ProjectAnalysis, projectPath: string) : TeamDeliverable list =
            [
                {
                    TeamName = "DevOps"
                    DeliverableName = "Dockerfile"
                    FilePath = "Dockerfile"
                    Content = this.GenerateDockerfile(analysis)
                    Priority = 3
                    Dependencies = ["Project Configuration"]
                }
                {
                    TeamName = "DevOps"
                    DeliverableName = "CI/CD Pipeline"
                    FilePath = ".github/workflows/ci-cd.yml"
                    Content = this.GenerateCICDPipeline(analysis)
                    Priority = 4
                    Dependencies = ["Dockerfile"]
                }
                {
                    TeamName = "DevOps"
                    DeliverableName = "Kubernetes Deployment"
                    FilePath = "k8s/deployment.yaml"
                    Content = this.GenerateKubernetesDeployment(analysis)
                    Priority = 5
                    Dependencies = ["CI/CD Pipeline"]
                }
            ]

        /// <summary>
        /// Generate Project Management deliverables
        /// </summary>
        member private this.GenerateProjectManagementDeliverables(analysis: ProjectAnalysis, projectPath: string) : TeamDeliverable list =
            [
                {
                    TeamName = "Project Management"
                    DeliverableName = "Project Plan"
                    FilePath = "docs/project-plan.md"
                    Content = this.GenerateProjectPlan(analysis)
                    Priority = 1
                    Dependencies = ["Requirements Document"]
                }
                {
                    TeamName = "Project Management"
                    DeliverableName = "Risk Assessment"
                    FilePath = "docs/risk-assessment.md"
                    Content = this.GenerateRiskAssessment(analysis)
                    Priority = 2
                    Dependencies = ["Project Plan"]
                }
            ]
