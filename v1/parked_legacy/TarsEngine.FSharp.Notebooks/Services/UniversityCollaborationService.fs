namespace TarsEngine.FSharp.Notebooks.Services

open System
open System.Collections.Generic
open System.Net.Http
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Notebooks.Types
open TarsEngine.FSharp.Notebooks.Discovery

/// <summary>
/// Service for university and academic collaboration features
/// </summary>

/// University information
type University = {
    Name: string
    Country: string
    Website: string option
    ResearchAreas: string list
    Departments: string list
}

/// Academic collaboration request
type CollaborationRequest = {
    RequestId: string
    Title: string
    Description: string
    ResearchArea: string
    RequiredSkills: string list
    Duration: TimeSpan option
    ContactEmail: string
    University: University option
    CreatedDate: DateTime
    Status: CollaborationStatus
}

/// Collaboration status
and CollaborationStatus = 
    | Open
    | InProgress
    | Completed
    | Cancelled

/// Research project
type ResearchProject = {
    ProjectId: string
    Title: string
    Description: string
    PrincipalInvestigator: string
    University: University
    ResearchArea: string
    StartDate: DateTime
    EndDate: DateTime option
    Notebooks: string list
    Publications: string list
    Status: ProjectStatus
}

/// Project status
and ProjectStatus = 
    | Planning
    | Active
    | OnHold
    | Completed
    | Archived

/// Academic notebook metadata
type AcademicNotebookMetadata = {
    CourseCode: string option
    CourseName: string option
    Instructor: string option
    University: University option
    Semester: string option
    AcademicYear: string option
    LearningObjectives: string list
    Prerequisites: string list
    DifficultyLevel: DifficultyLevel
    EstimatedCompletionTime: TimeSpan option
}

/// Difficulty levels
and DifficultyLevel = 
    | Beginner
    | Intermediate
    | Advanced
    | Expert

/// University collaboration service
type UniversityCollaborationService(httpClient: HttpClient, logger: ILogger<UniversityCollaborationService>) =
    
    let collaborationRequests = Dictionary<string, CollaborationRequest>()
    let researchProjects = Dictionary<string, ResearchProject>()
    let universities = Dictionary<string, University>()
    
    /// Initialize with sample universities
    do
        let sampleUniversities = [
            {
                Name = "MIT"
                Country = "USA"
                Website = Some "https://web.mit.edu"
                ResearchAreas = ["Computer Science"; "AI/ML"; "Data Science"; "Engineering"]
                Departments = ["EECS"; "Mathematics"; "Physics"; "Chemistry"]
            }
            {
                Name = "Stanford University"
                Country = "USA"
                Website = Some "https://stanford.edu"
                ResearchAreas = ["Computer Science"; "AI/ML"; "Biomedical Engineering"; "Economics"]
                Departments = ["Computer Science"; "Engineering"; "Medicine"; "Business"]
            }
            {
                Name = "University of Cambridge"
                Country = "UK"
                Website = Some "https://cam.ac.uk"
                ResearchAreas = ["Mathematics"; "Physics"; "Computer Science"; "Natural Sciences"]
                Departments = ["Mathematics"; "Computer Science"; "Physics"; "Engineering"]
            }
        ]
        
        for university in sampleUniversities do
            universities.[university.Name] <- university
    
    /// Create collaboration request
    member _.CreateCollaborationRequestAsync(title: string, description: string, researchArea: string, requiredSkills: string list, contactEmail: string) : Async<CollaborationRequest> = async {
        try
            let request = {
                RequestId = Guid.NewGuid().ToString()
                Title = title
                Description = description
                ResearchArea = researchArea
                RequiredSkills = requiredSkills
                Duration = None
                ContactEmail = contactEmail
                University = None
                CreatedDate = DateTime.UtcNow
                Status = Open
            }
            
            collaborationRequests.[request.RequestId] <- request
            
            logger.LogInformation("Created collaboration request: {Title} ({RequestId})", title, request.RequestId)
            return request
            
        with
        | ex ->
            logger.LogError(ex, "Failed to create collaboration request")
            return failwith $"Failed to create collaboration request: {ex.Message}"
    }
    
    /// Search collaboration requests
    member _.SearchCollaborationRequestsAsync(researchArea: string option, skills: string list) : Async<CollaborationRequest list> = async {
        try
            let requests = 
                collaborationRequests.Values
                |> Seq.filter (fun req -> req.Status = Open)
                |> Seq.filter (fun req ->
                    match researchArea with
                    | Some area -> req.ResearchArea.Contains(area, StringComparison.OrdinalIgnoreCase)
                    | None -> true)
                |> Seq.filter (fun req ->
                    if skills.IsEmpty then true
                    else
                        skills |> List.exists (fun skill ->
                            req.RequiredSkills |> List.exists (fun reqSkill ->
                                reqSkill.Contains(skill, StringComparison.OrdinalIgnoreCase))))
                |> List.ofSeq
            
            logger.LogInformation("Found {Count} collaboration requests matching criteria", requests.Length)
            return requests
            
        with
        | ex ->
            logger.LogError(ex, "Failed to search collaboration requests")
            return []
    }
    
    /// Create research project
    member _.CreateResearchProjectAsync(title: string, description: string, pi: string, university: University, researchArea: string) : Async<ResearchProject> = async {
        try
            let project = {
                ProjectId = Guid.NewGuid().ToString()
                Title = title
                Description = description
                PrincipalInvestigator = pi
                University = university
                ResearchArea = researchArea
                StartDate = DateTime.UtcNow
                EndDate = None
                Notebooks = []
                Publications = []
                Status = Planning
            }
            
            researchProjects.[project.ProjectId] <- project
            
            logger.LogInformation("Created research project: {Title} ({ProjectId})", title, project.ProjectId)
            return project
            
        with
        | ex ->
            logger.LogError(ex, "Failed to create research project")
            return failwith $"Failed to create research project: {ex.Message}"
    }
    
    /// Search research projects
    member _.SearchResearchProjectsAsync(researchArea: string option, university: string option) : Async<ResearchProject list> = async {
        try
            let projects = 
                researchProjects.Values
                |> Seq.filter (fun proj ->
                    match researchArea with
                    | Some area -> proj.ResearchArea.Contains(area, StringComparison.OrdinalIgnoreCase)
                    | None -> true)
                |> Seq.filter (fun proj ->
                    match university with
                    | Some uni -> proj.University.Name.Contains(uni, StringComparison.OrdinalIgnoreCase)
                    | None -> true)
                |> List.ofSeq
            
            logger.LogInformation("Found {Count} research projects matching criteria", projects.Length)
            return projects
            
        with
        | ex ->
            logger.LogError(ex, "Failed to search research projects")
            return []
    }
    
    /// Add academic metadata to notebook
    member _.AddAcademicMetadataAsync(notebook: JupyterNotebook, academicMetadata: AcademicNotebookMetadata) : Async<JupyterNotebook> = async {
        try
            let customMetadata = 
                notebook.Metadata.Custom
                |> Map.add "academic" (academicMetadata :> obj)
                |> Map.add "course_code" (academicMetadata.CourseCode |> Option.defaultValue "" :> obj)
                |> Map.add "course_name" (academicMetadata.CourseName |> Option.defaultValue "" :> obj)
                |> Map.add "instructor" (academicMetadata.Instructor |> Option.defaultValue "" :> obj)
                |> Map.add "difficulty_level" (academicMetadata.DifficultyLevel.ToString() :> obj)
            
            let updatedMetadata = { notebook.Metadata with Custom = customMetadata }
            let updatedNotebook = { notebook with Metadata = updatedMetadata }
            
            logger.LogInformation("Added academic metadata to notebook")
            return updatedNotebook
            
        with
        | ex ->
            logger.LogError(ex, "Failed to add academic metadata")
            return failwith $"Failed to add academic metadata: {ex.Message}"
    }
    
    /// Generate course notebook template
    member _.GenerateCourseNotebookAsync(courseCode: string, courseName: string, instructor: string, university: University, learningObjectives: string list) : Async<JupyterNotebook> = async {
        try
            logger.LogInformation("Generating course notebook template for: {CourseCode} - {CourseName}", courseCode, courseName)
            
            let cells = [
                // Title cell
                MarkdownCell {
                    Source = [
                        $"# {courseName}"
                        $"**Course Code:** {courseCode}"
                        $"**Instructor:** {instructor}"
                        $"**University:** {university.Name}"
                        ""
                        "## Learning Objectives"
                    ] @ (learningObjectives |> List.map (fun obj -> $"- {obj}"))
                    Metadata = Map.empty
                }
                
                // Setup cell
                CodeCell {
                    Source = [
                        "# Course setup and imports"
                        "import numpy as np"
                        "import pandas as pd"
                        "import matplotlib.pyplot as plt"
                        "import seaborn as sns"
                        ""
                        "# Configure plotting"
                        "plt.style.use('seaborn-v0_8')"
                        "plt.rcParams['figure.figsize'] = (10, 6)"
                    ]
                    Outputs = None
                    ExecutionCount = None
                    Metadata = Map.empty
                }
                
                // Introduction cell
                MarkdownCell {
                    Source = [
                        "## Introduction"
                        ""
                        "This notebook covers the key concepts and practical exercises for this course."
                        ""
                        "### Prerequisites"
                        "- Basic programming knowledge"
                        "- Familiarity with Python"
                        ""
                        "### What you will learn"
                        "By the end of this notebook, you will be able to:"
                    ] @ (learningObjectives |> List.map (fun obj -> $"- {obj}"))
                    Metadata = Map.empty
                }
                
                // Exercise template cell
                MarkdownCell {
                    Source = [
                        "## Exercise 1: Getting Started"
                        ""
                        "**Instructions:** Complete the following tasks..."
                        ""
                        "**Expected Output:** ..."
                    ]
                    Metadata = Map.empty
                }
                
                CodeCell {
                    Source = [
                        "# Your code here"
                        ""
                    ]
                    Outputs = None
                    ExecutionCount = None
                    Metadata = Map.empty
                }
            ]
            
            let academicMetadata = {
                CourseCode = Some courseCode
                CourseName = Some courseName
                Instructor = Some instructor
                University = Some university
                Semester = None
                AcademicYear = None
                LearningObjectives = learningObjectives
                Prerequisites = ["Basic programming knowledge"; "Python familiarity"]
                DifficultyLevel = Intermediate
                EstimatedCompletionTime = Some (TimeSpan.FromHours(2.0))
            }
            
            let metadata = {
                Title = Some courseName
                Authors = [instructor]
                Description = Some $"Course notebook for {courseCode} - {courseName}"
                Tags = ["education"; "course"; courseCode.ToLower()]
                KernelSpec = Some {
                    Name = "python3"
                    DisplayName = "Python 3"
                    Language = Some "python"
                }
                LanguageInfo = Some {
                    Name = "python"
                    Version = "3.9"
                    MimeType = "text/x-python"
                    FileExtension = ".py"
                    PygmentsLexer = Some "ipython3"
                    CodeMirrorMode = Some "python"
                    NBConvertExporter = Some "python"
                }
                CreatedDate = Some DateTime.UtcNow
                ModifiedDate = Some DateTime.UtcNow
                Version = Some "1.0"
                Custom = Map.ofList [("academic", academicMetadata :> obj)]
            }
            
            let notebook = {
                NbFormat = 4
                NbFormatMinor = 5
                Metadata = metadata
                Cells = cells
            }
            
            logger.LogInformation("Course notebook template generated with {CellCount} cells", cells.Length)
            return notebook
            
        with
        | ex ->
            logger.LogError(ex, "Failed to generate course notebook")
            return failwith $"Failed to generate course notebook: {ex.Message}"
    }
    
    /// Get universities
    member _.GetUniversitiesAsync() : Async<University list> = async {
        return universities.Values |> List.ofSeq
    }
    
    /// Find university by name
    member _.FindUniversityAsync(name: string) : Async<University option> = async {
        let found = 
            universities.Values
            |> Seq.tryFind (fun uni -> uni.Name.Contains(name, StringComparison.OrdinalIgnoreCase))
        
        return found
    }
    
    /// Get collaboration statistics
    member _.GetCollaborationStatisticsAsync() : Async<CollaborationStatistics> = async {
        let totalRequests = collaborationRequests.Count
        let openRequests = collaborationRequests.Values |> Seq.filter (fun r -> r.Status = Open) |> Seq.length
        let inProgressRequests = collaborationRequests.Values |> Seq.filter (fun r -> r.Status = InProgress) |> Seq.length
        let completedRequests = collaborationRequests.Values |> Seq.filter (fun r -> r.Status = Completed) |> Seq.length
        
        let totalProjects = researchProjects.Count
        let activeProjects = researchProjects.Values |> Seq.filter (fun p -> p.Status = Active) |> Seq.length
        let completedProjects = researchProjects.Values |> Seq.filter (fun p -> p.Status = Completed) |> Seq.length
        
        return {
            TotalCollaborationRequests = totalRequests
            OpenRequests = openRequests
            InProgressRequests = inProgressRequests
            CompletedRequests = completedRequests
            TotalResearchProjects = totalProjects
            ActiveProjects = activeProjects
            CompletedProjects = completedProjects
            TotalUniversities = universities.Count
        }

/// Collaboration statistics
and CollaborationStatistics = {
    TotalCollaborationRequests: int
    OpenRequests: int
    InProgressRequests: int
    CompletedRequests: int
    TotalResearchProjects: int
    ActiveProjects: int
    CompletedProjects: int
    TotalUniversities: int
}

/// Collaboration utilities
module CollaborationUtils =
    
    /// Create academic metadata
    let createAcademicMetadata courseCode courseName instructor university = {
        CourseCode = courseCode
        CourseName = courseName
        Instructor = instructor
        University = university
        Semester = None
        AcademicYear = None
        LearningObjectives = []
        Prerequisites = []
        DifficultyLevel = Intermediate
        EstimatedCompletionTime = None
    }
    
    /// Format collaboration request
    let formatCollaborationRequest (request: CollaborationRequest) : string =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine($"🤝 Collaboration Request: {request.Title}") |> ignore
        sb.AppendLine($"Research Area: {request.ResearchArea}") |> ignore
        sb.AppendLine($"Status: {request.Status}") |> ignore
        sb.AppendLine($"Contact: {request.ContactEmail}") |> ignore
        sb.AppendLine($"Created: {request.CreatedDate:yyyy-MM-dd}") |> ignore
        
        if not request.RequiredSkills.IsEmpty then
            sb.AppendLine($"Required Skills: {String.Join(", ", request.RequiredSkills)}") |> ignore
        
        sb.AppendLine($"Description: {request.Description}") |> ignore
        
        sb.ToString()
    
    /// Format research project
    let formatResearchProject (project: ResearchProject) : string =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine($"🔬 Research Project: {project.Title}") |> ignore
        sb.AppendLine($"PI: {project.PrincipalInvestigator}") |> ignore
        sb.AppendLine($"University: {project.University.Name}") |> ignore
        sb.AppendLine($"Research Area: {project.ResearchArea}") |> ignore
        sb.AppendLine($"Status: {project.Status}") |> ignore
        sb.AppendLine($"Start Date: {project.StartDate:yyyy-MM-dd}") |> ignore
        
        match project.EndDate with
        | Some endDate -> sb.AppendLine($"End Date: {endDate:yyyy-MM-dd}") |> ignore
        | None -> sb.AppendLine("End Date: Ongoing") |> ignore
        
        sb.AppendLine($"Notebooks: {project.Notebooks.Length}") |> ignore
        sb.AppendLine($"Publications: {project.Publications.Length}") |> ignore
        sb.AppendLine($"Description: {project.Description}") |> ignore
        
        sb.ToString()
    
    /// Get difficulty level emoji
    let getDifficultyEmoji (level: DifficultyLevel) : string =
        match level with
        | Beginner -> "🟢"
        | Intermediate -> "🟡"
        | Advanced -> "🟠"
        | Expert -> "🔴"
