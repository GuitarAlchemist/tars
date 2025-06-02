module TarsFableUI.App

open System
open Fable.React
open Fable.React.Props
open Elmish
open Elmish.React
open Elmish.Browser.Navigation
open Elmish.Browser.UrlParser
open TarsFableUI.Types
open TarsFableUI.AgentTypes

// Routing
type Page =
    | Dashboard
    | AgentMonitor
    | LiveDemo
    | NotFound

let pageParser: Parser<Page -> Page, Page> =
    oneOf [
        map Dashboard (s "dashboard")
        map AgentMonitor (s "agents")
        map LiveDemo (s "demo")
        map Dashboard top
    ]

// Model
type Model = {
    CurrentPage: Page
    SystemStatus: TarsSystemStatus
    Agents: TarsAgent list
    AgentTeamStatus: AgentTeamStatus
    CurrentWorkflow: UIGenerationWorkflow option
    DemoPrompt: string
    IsGenerating: bool
    GeneratedComponents: string list
    LastUpdate: DateTime
}

// Messages
type Msg =
    | UrlChanged of Page
    | RefreshSystemStatus
    | SystemStatusReceived of TarsSystemStatus
    | AgentsReceived of TarsAgent list
    | AgentTeamStatusReceived of AgentTeamStatus
    | UpdateDemoPrompt of string
    | GenerateComponent
    | ComponentGenerated of string
    | WorkflowUpdated of UIGenerationWorkflow
    | Tick

// Initial state
let init (page: Page option) =
    let initialPage = page |> Option.defaultValue Dashboard
    
    let initialModel = {
        CurrentPage = initialPage
        SystemStatus = {
            IsOnline = true
            Version = "1.0.0"
            Uptime = TimeSpan.FromHours(2.5)
            CpuUsage = 45.2
            MemoryUsage = 67.8
            CudaAvailable = true
            AgentCount = 6
            LastUpdate = DateTime.UtcNow
        }
        Agents = [
            {
                Id = "ui-architect"
                Name = "UIArchitectAgent"
                Persona = "UI Designer"
                Status = Active
                Capabilities = ["UI Design"; "Component Architecture"; "Layout Planning"]
                CurrentTask = Some "Analyzing dashboard layout"
                Performance = {
                    TasksCompleted = 15
                    AverageExecutionTime = TimeSpan.FromSeconds(2.3)
                    SuccessRate = 0.95
                    LastActivity = DateTime.UtcNow.AddMinutes(-2.0)
                }
            }
            {
                Id = "component-generator"
                Name = "ComponentGeneratorAgent"
                Persona = "Code Generator"
                Status = Busy
                Capabilities = ["F# Code Generation"; "React Components"; "Type Safety"]
                CurrentTask = Some "Generating StatusCard component"
                Performance = {
                    TasksCompleted = 23
                    AverageExecutionTime = TimeSpan.FromSeconds(1.8)
                    SuccessRate = 0.98
                    LastActivity = DateTime.UtcNow.AddMinutes(-0.5)
                }
            }
            {
                Id = "style-agent"
                Name = "StyleAgent"
                Persona = "Style Designer"
                Status = Active
                Capabilities = ["CSS Generation"; "Tailwind Classes"; "Responsive Design"]
                CurrentTask = Some "Optimizing mobile styles"
                Performance = {
                    TasksCompleted = 31
                    AverageExecutionTime = TimeSpan.FromSeconds(1.2)
                    SuccessRate = 0.92
                    LastActivity = DateTime.UtcNow.AddMinutes(-1.0)
                }
            }
            {
                Id = "state-manager"
                Name = "StateManagerAgent"
                Persona = "State Architect"
                Status = Idle
                Capabilities = ["Elmish Architecture"; "State Management"; "Message Flow"]
                CurrentTask = None
                Performance = {
                    TasksCompleted = 12
                    AverageExecutionTime = TimeSpan.FromSeconds(3.1)
                    SuccessRate = 0.89
                    LastActivity = DateTime.UtcNow.AddMinutes(-5.0)
                }
            }
            {
                Id = "integration-agent"
                Name = "IntegrationAgent"
                Persona = "Build Engineer"
                Status = Active
                Capabilities = ["Fable Compilation"; "Webpack"; "Hot Reload"]
                CurrentTask = Some "Compiling F# to JavaScript"
                Performance = {
                    TasksCompleted = 8
                    AverageExecutionTime = TimeSpan.FromSeconds(4.5)
                    SuccessRate = 0.94
                    LastActivity = DateTime.UtcNow.AddMinutes(-0.2)
                }
            }
            {
                Id = "quality-agent"
                Name = "QualityAgent"
                Persona = "QA Engineer"
                Status = Idle
                Capabilities = ["Testing"; "Accessibility"; "Performance Analysis"]
                CurrentTask = None
                Performance = {
                    TasksCompleted = 19
                    AverageExecutionTime = TimeSpan.FromSeconds(2.8)
                    SuccessRate = 0.97
                    LastActivity = DateTime.UtcNow.AddMinutes(-3.0)
                }
            }
        ]
        AgentTeamStatus = {
            UIArchitect = { Status = Active; CurrentTask = Some "Layout analysis"; Progress = 0.75; EstimatedCompletion = Some (DateTime.UtcNow.AddMinutes(1.0)); LastOutput = Some DateTime.UtcNow }
            ComponentGenerator = { Status = Busy; CurrentTask = Some "Component generation"; Progress = 0.60; EstimatedCompletion = Some (DateTime.UtcNow.AddMinutes(2.0)); LastOutput = Some DateTime.UtcNow }
            StyleAgent = { Status = Active; CurrentTask = Some "Style optimization"; Progress = 0.80; EstimatedCompletion = Some (DateTime.UtcNow.AddMinutes(0.5)); LastOutput = Some DateTime.UtcNow }
            StateManager = { Status = Idle; CurrentTask = None; Progress = 0.0; EstimatedCompletion = None; LastOutput = Some (DateTime.UtcNow.AddMinutes(-5.0)) }
            IntegrationAgent = { Status = Active; CurrentTask = Some "Compilation"; Progress = 0.90; EstimatedCompletion = Some (DateTime.UtcNow.AddMinutes(0.3)); LastOutput = Some DateTime.UtcNow }
            QualityAgent = { Status = Idle; CurrentTask = None; Progress = 0.0; EstimatedCompletion = None; LastOutput = Some (DateTime.UtcNow.AddMinutes(-3.0)) }
            LastCoordination = DateTime.UtcNow.AddMinutes(-0.1)
            ActiveWorkflow = Some "Dashboard Component Generation"
        }
        CurrentWorkflow = None
        DemoPrompt = ""
        IsGenerating = false
        GeneratedComponents = ["StatusCard"; "AgentCard"; "PromptInput"]
        LastUpdate = DateTime.UtcNow
    }
    
    initialModel, Cmd.none

// Update
let update (msg: Msg) (model: Model) =
    match msg with
    | UrlChanged page ->
        { model with CurrentPage = page }, Cmd.none
    
    | RefreshSystemStatus ->
        // Simulate system status update
        let updatedStatus = { 
            model.SystemStatus with 
                CpuUsage = Random().NextDouble() * 100.0
                MemoryUsage = Random().NextDouble() * 100.0
                LastUpdate = DateTime.UtcNow
        }
        { model with SystemStatus = updatedStatus }, Cmd.none
    
    | SystemStatusReceived status ->
        { model with SystemStatus = status }, Cmd.none
    
    | AgentsReceived agents ->
        { model with Agents = agents }, Cmd.none
    
    | AgentTeamStatusReceived teamStatus ->
        { model with AgentTeamStatus = teamStatus }, Cmd.none
    
    | UpdateDemoPrompt prompt ->
        { model with DemoPrompt = prompt }, Cmd.none
    
    | GenerateComponent ->
        { model with IsGenerating = true }, Cmd.none
    
    | ComponentGenerated componentName ->
        { model with 
            IsGenerating = false
            GeneratedComponents = componentName :: model.GeneratedComponents }, Cmd.none
    
    | WorkflowUpdated workflow ->
        { model with CurrentWorkflow = Some workflow }, Cmd.none
    
    | Tick ->
        let updatedModel = { model with LastUpdate = DateTime.UtcNow }
        updatedModel, Cmd.none

// View Components
let statusCard (systemStatus: TarsSystemStatus) (dispatch: Msg -> unit) =
    let statusColor = if systemStatus.IsOnline then "text-green-400" else "text-red-400"
    let statusIcon = if systemStatus.IsOnline then "fas fa-check-circle" else "fas fa-exclamation-triangle"

    div [ ClassName "bg-tars-gray rounded-lg p-6 tars-glow" ] [
        div [ ClassName "flex items-center justify-between mb-4" ] [
            h3 [ ClassName "text-lg font-semibold text-white" ] [ str "System Status" ]
            button [
                ClassName "text-tars-cyan hover:text-white transition-colors"
                OnClick (fun _ -> dispatch RefreshSystemStatus)
            ] [
                i [ ClassName "fas fa-sync-alt" ] []
            ]
        ]
        div [ ClassName "space-y-3" ] [
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "Status" ]
                div [ ClassName ("flex items-center " + statusColor) ] [
                    i [ ClassName statusIcon ] []
                    span [ ClassName "ml-2" ] [ str (if systemStatus.IsOnline then "Online" else "Offline") ]
                ]
            ]
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "Version" ]
                span [ ClassName "text-white" ] [ str systemStatus.Version ]
            ]
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "Agents" ]
                span [ ClassName "text-tars-cyan font-semibold" ] [ str (string systemStatus.AgentCount) ]
            ]
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "CPU Usage" ]
                span [ ClassName "text-white" ] [ str (sprintf "%.1f%%" systemStatus.CpuUsage) ]
            ]
            if systemStatus.CudaAvailable then
                div [ ClassName "flex items-center justify-between" ] [
                    span [ ClassName "text-gray-300" ] [ str "CUDA" ]
                    span [ ClassName "text-green-400 flex items-center" ] [
                        i [ ClassName "fas fa-bolt mr-1" ] []
                        str "Available"
                    ]
                ]
        ]
    ]

let agentCard (agent: TarsAgent) =
    let statusColor =
        match agent.Status with
        | Active -> "text-green-400"
        | Busy -> "text-yellow-400"
        | Idle -> "text-gray-400"
        | Error _ -> "text-red-400"

    let statusIcon =
        match agent.Status with
        | Active -> "fas fa-play-circle"
        | Busy -> "fas fa-cog fa-spin"
        | Idle -> "fas fa-pause-circle"
        | Error _ -> "fas fa-exclamation-circle"

    div [ ClassName "bg-tars-gray rounded-lg p-4 hover:bg-opacity-80 transition-all border border-transparent hover:border-tars-cyan" ] [
        div [ ClassName "flex items-center justify-between mb-3" ] [
            h4 [ ClassName "text-white font-semibold" ] [ str agent.Name ]
            div [ ClassName ("flex items-center " + statusColor) ] [
                i [ ClassName statusIcon ] []
            ]
        ]
        div [ ClassName "space-y-2" ] [
            div [ ClassName "text-sm" ] [
                span [ ClassName "text-gray-400" ] [ str "Persona: " ]
                span [ ClassName "text-tars-cyan" ] [ str agent.Persona ]
            ]
            if agent.CurrentTask.IsSome then
                div [ ClassName "text-sm" ] [
                    span [ ClassName "text-gray-400" ] [ str "Task: " ]
                    span [ ClassName "text-white text-xs" ] [ str agent.CurrentTask.Value ]
                ]
            div [ ClassName "flex justify-between text-xs text-gray-400" ] [
                span [] [ str (sprintf "Tasks: %d" agent.Performance.TasksCompleted) ]
                span [] [ str (sprintf "Success: %.1f%%" (agent.Performance.SuccessRate * 100.0)) ]
            ]
        ]
    ]

let promptInput (value: string) (isGenerating: bool) (dispatch: Msg -> unit) =
    div [ ClassName "relative" ] [
        textarea [
            ClassName "w-full p-4 bg-tars-gray rounded-lg text-white placeholder-gray-400 border border-gray-600 focus:border-tars-cyan focus:ring-1 focus:ring-tars-cyan resize-none"
            Placeholder "Describe the UI component you want TARS agents to generate..."
            Value value
            OnChange (fun e -> dispatch (UpdateDemoPrompt e.target?value))
            OnKeyDown (fun e ->
                if e.key = "Enter" && e.ctrlKey then
                    e.preventDefault()
                    dispatch GenerateComponent)
            Rows 4
            Disabled isGenerating
        ] []
        button [
            ClassName ("absolute bottom-3 right-3 px-4 py-2 bg-tars-cyan text-white rounded-md hover:bg-opacity-80 transition-colors " +
                      if isGenerating then "opacity-50 cursor-not-allowed" else "")
            OnClick (fun _ -> if not isGenerating then dispatch GenerateComponent)
            Disabled isGenerating
        ] [
            if isGenerating then
                span [ ClassName "flex items-center" ] [
                    i [ ClassName "fas fa-spinner fa-spin mr-2" ] []
                    str "Generating..."
                ]
            else
                span [ ClassName "flex items-center" ] [
                    i [ ClassName "fas fa-magic mr-2" ] []
                    str "Generate (Ctrl+Enter)"
                ]
        ]
    ]

// Page Views
let dashboardView (model: Model) (dispatch: Msg -> unit) =
    div [ ClassName "space-y-6" ] [
        div [ ClassName "flex items-center justify-between" ] [
            h1 [ ClassName "text-3xl font-bold text-white" ] [ str "TARS Dashboard" ]
            div [ ClassName "flex items-center space-x-2 text-sm text-gray-400" ] [
                i [ ClassName "fas fa-clock" ] []
                span [] [ str ("Last updated: " + model.LastUpdate.ToString("HH:mm:ss")) ]
            ]
        ]

        div [ ClassName "grid grid-cols-1 lg:grid-cols-3 gap-6" ] [
            div [ ClassName "lg:col-span-1" ] [
                statusCard model.SystemStatus dispatch
            ]
            div [ ClassName "lg:col-span-2" ] [
                div [ ClassName "bg-tars-gray rounded-lg p-6" ] [
                    h3 [ ClassName "text-lg font-semibold text-white mb-4" ] [ str "Agent Team Status" ]
                    div [ ClassName "grid grid-cols-1 md:grid-cols-2 gap-4" ] [
                        for agent in model.Agents |> List.take 4 do
                            agentCard agent
                    ]
                ]
            ]
        ]

        div [ ClassName "bg-tars-gray rounded-lg p-6" ] [
            h3 [ ClassName "text-lg font-semibold text-white mb-4" ] [ str "Recent Activity" ]
            div [ ClassName "space-y-3" ] [
                div [ ClassName "flex items-center space-x-3 text-sm" ] [
                    i [ ClassName "fas fa-check-circle text-green-400" ] []
                    span [ ClassName "text-white" ] [ str "ComponentGeneratorAgent completed StatusCard generation" ]
                    span [ ClassName "text-gray-400" ] [ str "2 minutes ago" ]
                ]
                div [ ClassName "flex items-center space-x-3 text-sm" ] [
                    i [ ClassName "fas fa-cog fa-spin text-yellow-400" ] []
                    span [ ClassName "text-white" ] [ str "StyleAgent optimizing responsive breakpoints" ]
                    span [ ClassName "text-gray-400" ] [ str "30 seconds ago" ]
                ]
                div [ ClassName "flex items-center space-x-3 text-sm" ] [
                    i [ ClassName "fas fa-bolt text-tars-cyan" ] []
                    span [ ClassName "text-white" ] [ str "IntegrationAgent compiled F# to JavaScript" ]
                    span [ ClassName "text-gray-400" ] [ str "1 minute ago" ]
                ]
            ]
        ]
    ]

let agentMonitorView (model: Model) (dispatch: Msg -> unit) =
    div [ ClassName "space-y-6" ] [
        div [ ClassName "flex items-center justify-between" ] [
            h1 [ ClassName "text-3xl font-bold text-white" ] [ str "Agent Monitor" ]
            div [ ClassName "flex items-center space-x-4" ] [
                span [ ClassName "text-sm text-gray-400" ] [ str (sprintf "%d agents active" (model.Agents |> List.filter (fun a -> a.Status = Active || a.Status = Busy) |> List.length)) ]
                button [
                    ClassName "px-4 py-2 bg-tars-cyan text-white rounded-md hover:bg-opacity-80 transition-colors"
                    OnClick (fun _ -> dispatch RefreshSystemStatus)
                ] [
                    i [ ClassName "fas fa-sync-alt mr-2" ] []
                    str "Refresh"
                ]
            ]
        ]

        div [ ClassName "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" ] [
            for agent in model.Agents do
                agentCard agent
        ]

        div [ ClassName "bg-tars-gray rounded-lg p-6" ] [
            h3 [ ClassName "text-lg font-semibold text-white mb-4" ] [ str "Team Coordination" ]
            div [ ClassName "space-y-4" ] [
                div [ ClassName "flex items-center justify-between p-3 bg-tars-dark rounded-lg" ] [
                    span [ ClassName "text-white" ] [ str "Active Workflow" ]
                    span [ ClassName "text-tars-cyan" ] [ str (model.AgentTeamStatus.ActiveWorkflow |> Option.defaultValue "None") ]
                ]
                div [ ClassName "flex items-center justify-between p-3 bg-tars-dark rounded-lg" ] [
                    span [ ClassName "text-white" ] [ str "Last Coordination" ]
                    span [ ClassName "text-gray-400" ] [ str (model.AgentTeamStatus.LastCoordination.ToString("HH:mm:ss")) ]
                ]
            ]
        ]
    ]

let liveDemoView (model: Model) (dispatch: Msg -> unit) =
    div [ ClassName "max-w-4xl mx-auto space-y-6" ] [
        div [ ClassName "text-center" ] [
            h1 [ ClassName "text-3xl font-bold text-white mb-2" ] [ str "Live UI Generation Demo" ]
            p [ ClassName "text-gray-400" ] [ str "Watch TARS agent teams generate UI components in real-time" ]
        ]

        div [ ClassName "bg-tars-gray rounded-lg p-6" ] [
            h3 [ ClassName "text-lg font-semibold text-white mb-4" ] [ str "Component Request" ]
            promptInput model.DemoPrompt model.IsGenerating dispatch
        ]

        if model.IsGenerating then
            div [ ClassName "bg-tars-gray rounded-lg p-6" ] [
                h3 [ ClassName "text-lg font-semibold text-white mb-4" ] [ str "Agent Team Working..." ]
                div [ ClassName "space-y-3" ] [
                    div [ ClassName "flex items-center justify-between p-3 bg-tars-dark rounded-lg" ] [
                        span [ ClassName "text-white" ] [ str "UIArchitectAgent" ]
                        div [ ClassName "flex items-center space-x-2" ] [
                            div [ ClassName "w-32 bg-gray-600 rounded-full h-2" ] [
                                div [ ClassName "bg-tars-cyan h-2 rounded-full transition-all duration-300"; Style [Width "75%"] ] []
                            ]
                            span [ ClassName "text-tars-cyan text-sm" ] [ str "75%" ]
                        ]
                    ]
                    div [ ClassName "flex items-center justify-between p-3 bg-tars-dark rounded-lg" ] [
                        span [ ClassName "text-white" ] [ str "ComponentGeneratorAgent" ]
                        div [ ClassName "flex items-center space-x-2" ] [
                            div [ ClassName "w-32 bg-gray-600 rounded-full h-2" ] [
                                div [ ClassName "bg-yellow-400 h-2 rounded-full transition-all duration-300"; Style [Width "60%"] ] []
                            ]
                            span [ ClassName "text-yellow-400 text-sm" ] [ str "60%" ]
                        ]
                    ]
                    div [ ClassName "flex items-center justify-between p-3 bg-tars-dark rounded-lg" ] [
                        span [ ClassName "text-white" ] [ str "StyleAgent" ]
                        div [ ClassName "flex items-center space-x-2" ] [
                            div [ ClassName "w-32 bg-gray-600 rounded-full h-2" ] [
                                div [ ClassName "bg-green-400 h-2 rounded-full transition-all duration-300"; Style [Width "80%"] ] []
                            ]
                            span [ ClassName "text-green-400 text-sm" ] [ str "80%" ]
                        ]
                    ]
                ]
            ]

        if not (List.isEmpty model.GeneratedComponents) then
            div [ ClassName "bg-tars-gray rounded-lg p-6" ] [
                h3 [ ClassName "text-lg font-semibold text-white mb-4" ] [ str "Generated Components" ]
                div [ ClassName "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" ] [
                    for componentName in model.GeneratedComponents do
                        div [ ClassName "p-4 bg-tars-dark rounded-lg border border-tars-cyan" ] [
                            div [ ClassName "flex items-center justify-between mb-2" ] [
                                h4 [ ClassName "text-white font-semibold" ] [ str componentName ]
                                i [ ClassName "fas fa-check-circle text-green-400" ] []
                            ]
                            p [ ClassName "text-gray-400 text-sm" ] [ str "F# React component generated by TARS agents" ]
                        ]
                ]
            ]
    ]
