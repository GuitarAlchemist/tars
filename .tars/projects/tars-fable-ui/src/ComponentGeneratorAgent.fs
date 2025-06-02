module TarsFableUI.ComponentGeneratorAgent

open System
open TarsFableUI.Types
open TarsFableUI.AgentTypes

// ComponentGeneratorAgent - Generates F# React components
type ComponentGeneratorAgent() =
    let mutable status = Idle
    let mutable currentTask: string option = None
    let mutable generatedComponents: Map<string, string> = Map.empty
    
    // F# React component templates
    let componentTemplates = Map [
        ("StatusCard", """
module Components.StatusCard

open Fable.React
open Fable.React.Props
open TarsFableUI.Types

type Props = {
    SystemStatus: TarsSystemStatus
    OnRefresh: unit -> unit
}

let statusCard (props: Props) =
    let status = props.SystemStatus
    let statusColor = if status.IsOnline then "text-green-400" else "text-red-400"
    let statusIcon = if status.IsOnline then "fas fa-check-circle" else "fas fa-exclamation-triangle"
    
    div [ ClassName "bg-tars-gray rounded-lg p-6 tars-glow" ] [
        div [ ClassName "flex items-center justify-between mb-4" ] [
            h3 [ ClassName "text-lg font-semibold text-white" ] [ str "System Status" ]
            button [ 
                ClassName "text-tars-cyan hover:text-white transition-colors"
                OnClick (fun _ -> props.OnRefresh())
            ] [
                i [ ClassName "fas fa-sync-alt" ] []
            ]
        ]
        div [ ClassName "space-y-3" ] [
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "Status" ]
                div [ ClassName ("flex items-center " + statusColor) ] [
                    i [ ClassName statusIcon ] []
                    span [ ClassName "ml-2" ] [ str (if status.IsOnline then "Online" else "Offline") ]
                ]
            ]
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "Version" ]
                span [ ClassName "text-white" ] [ str status.Version ]
            ]
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "Agents" ]
                span [ ClassName "text-tars-cyan font-semibold" ] [ str (string status.AgentCount) ]
            ]
            div [ ClassName "flex items-center justify-between" ] [
                span [ ClassName "text-gray-300" ] [ str "CPU Usage" ]
                span [ ClassName "text-white" ] [ str (sprintf "%.1f%%" status.CpuUsage) ]
            ]
            if status.CudaAvailable then
                div [ ClassName "flex items-center justify-between" ] [
                    span [ ClassName "text-gray-300" ] [ str "CUDA" ]
                    span [ ClassName "text-green-400 flex items-center" ] [
                        i [ ClassName "fas fa-bolt mr-1" ] []
                        str "Available"
                    ]
                ]
        ]
    ]
""")
        
        ("AgentCard", """
module Components.AgentCard

open Fable.React
open Fable.React.Props
open TarsFableUI.Types

type Props = {
    Agent: TarsAgent
    OnSelect: TarsAgent -> unit
}

let agentCard (props: Props) =
    let agent = props.Agent
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
    
    div [ 
        ClassName "bg-tars-gray rounded-lg p-4 hover:bg-opacity-80 transition-all cursor-pointer border border-transparent hover:border-tars-cyan"
        OnClick (fun _ -> props.OnSelect(agent))
    ] [
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
                    span [ ClassName "text-white" ] [ str agent.CurrentTask.Value ]
                ]
            div [ ClassName "flex justify-between text-xs text-gray-400" ] [
                span [] [ str (sprintf "Tasks: %d" agent.Performance.TasksCompleted) ]
                span [] [ str (sprintf "Success: %.1f%%" (agent.Performance.SuccessRate * 100.0)) ]
            ]
        ]
    ]
""")
        
        ("PromptInput", """
module Components.PromptInput

open Fable.React
open Fable.React.Props

type Props = {
    Value: string
    OnChange: string -> unit
    OnSubmit: unit -> unit
    Placeholder: string
    IsGenerating: bool
}

let promptInput (props: Props) =
    div [ ClassName "relative" ] [
        textarea [
            ClassName "w-full p-4 bg-tars-gray rounded-lg text-white placeholder-gray-400 border border-gray-600 focus:border-tars-cyan focus:ring-1 focus:ring-tars-cyan resize-none"
            Placeholder props.Placeholder
            Value props.Value
            OnChange (fun e -> props.OnChange(e.target?value))
            OnKeyDown (fun e -> 
                if e.key = "Enter" && e.ctrlKey then
                    e.preventDefault()
                    props.OnSubmit())
            Rows 4
            Disabled props.IsGenerating
        ] []
        button [
            ClassName ("absolute bottom-3 right-3 px-4 py-2 bg-tars-cyan text-white rounded-md hover:bg-opacity-80 transition-colors " + 
                      if props.IsGenerating then "opacity-50 cursor-not-allowed" else "")
            OnClick (fun _ -> if not props.IsGenerating then props.OnSubmit())
            Disabled props.IsGenerating
        ] [
            if props.IsGenerating then
                span [ ClassName "flex items-center" ] [
                    i [ ClassName "fas fa-spinner fa-spin mr-2" ] []
                    str "Generating..."
                ]
            else
                span [ ClassName "flex items-center" ] [
                    i [ ClassName "fas fa-magic mr-2" ] []
                    str "Generate"
                ]
        ]
    ]
""")
    ]
    
    member this.GetStatus() = status
    member this.GetCurrentTask() = currentTask
    member this.GetGeneratedComponents() = generatedComponents
    
    member this.GenerateComponent(request: ComponentGenerationRequest) =
        async {
            status <- Active
            currentTask <- Some ("Generating component: " + request.ComponentSpec.Name)
            
            // Simulate component generation
            do! Async.Sleep(800)
            
            let componentName = request.ComponentSpec.Name
            let template = 
                componentTemplates 
                |> Map.tryFind componentName
                |> Option.defaultValue (this.GenerateCustomComponent(request))
            
            // Store generated component
            generatedComponents <- generatedComponents |> Map.add componentName template
            
            let response = {
                FSharpCode = template
                ComponentName = componentName
                Dependencies = ["Fable.React"; "Fable.React.Props"; "TarsFableUI.Types"]
                TestSuggestions = [
                    sprintf "Test %s renders correctly" componentName
                    sprintf "Test %s handles props properly" componentName
                    sprintf "Test %s event handlers work" componentName
                ]
            }
            
            status <- Idle
            currentTask <- None
            
            return response
        }
    
    member private this.GenerateCustomComponent(request: ComponentGenerationRequest) =
        let spec = request.ComponentSpec
        let styleClasses = String.concat " " spec.Styles
        
        sprintf """
module Components.%s

open Fable.React
open Fable.React.Props

type Props = {
    // Generated props based on component spec
    Data: obj option
    OnAction: unit -> unit
}

let %s (props: Props) =
    div [ ClassName "%s" ] [
        h3 [ ClassName "text-lg font-semibold text-white mb-4" ] [ str "%s" ]
        div [ ClassName "text-gray-300" ] [ str "Generated by ComponentGeneratorAgent" ]
        // Add more content based on component specification
    ]
""" spec.Name (spec.Name.ToLower()) styleClasses spec.Name
