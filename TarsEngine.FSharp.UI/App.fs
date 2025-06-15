namespace TarsEngine.FSharp.UI

open System
open Elmish
open Feliz
open Feliz.Bulma
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.UI.Types
open TarsEngine.FSharp.UI.ElmishIntegration
open TarsEngine.FSharp.Core.RevolutionaryTypes

/// Main TARS UI Application using Elmish with Revolutionary Integration
module App =
    
    /// Update function - handles all state transitions
    let update (msg: Msg) (model: Model) : Model * Cmd<Msg> =
        match msg with
        | LoadAgents ->
            model, Cmd.ofMsg (AgentsLoaded SampleData.sampleAgents)
            
        | AgentsLoaded agents ->
            { model with Agents = agents; LastUpdate = DateTime.Now }, Cmd.none
            
        | AgentStatusChanged (agentId, newStatus) ->
            let updatedAgents = 
                model.Agents 
                |> List.map (fun agent -> 
                    if agent.Id = agentId then 
                        { agent with Status = newStatus; LastActivity = DateTime.Now }
                    else agent)
            { model with Agents = updatedAgents; LastUpdate = DateTime.Now }, Cmd.none
            
        | SelectAgent agent ->
            { model with SelectedAgent = Some agent }, Cmd.none
            
        | DeselectAgent ->
            { model with SelectedAgent = None }, Cmd.none
            
        | LoadDepartments ->
            model, Cmd.ofMsg (DepartmentsLoaded SampleData.sampleDepartments)
            
        | DepartmentsLoaded departments ->
            { model with Departments = departments }, Cmd.none
            
        | LoadTeams ->
            { model with Teams = [] }, Cmd.none
            
        | TeamsLoaded teams ->
            { model with Teams = teams }, Cmd.none
            
        | LoadNetworkNodes ->
            model, Cmd.ofMsg (NetworkNodesLoaded SampleData.sampleNetworkNodes)
            
        | NetworkNodesLoaded nodes ->
            { model with NetworkNodes = nodes }, Cmd.none
            
        | SelectNode node ->
            { model with SelectedNode = Some node }, Cmd.none
            
        | DeselectNode ->
            { model with SelectedNode = None }, Cmd.none
            
        | StartChat agentId ->
            { model with ActiveChat = Some agentId; ChatMessages = [] }, Cmd.none
            
        | EndChat ->
            { model with ActiveChat = None; ChatMessages = [] }, Cmd.none
            
        | SendMessage content ->
            match model.ActiveChat with
            | Some agentId ->
                let userMessage = {
                    Id = Guid.NewGuid().ToString()
                    AgentId = agentId
                    AgentName = "User"
                    Content = content
                    Timestamp = DateTime.Now
                    MessageType = "user"
                }
                let updatedMessages = model.ChatMessages @ [userMessage]
                
                // Simulate agent response
                let agentResponse = {
                    Id = Guid.NewGuid().ToString()
                    AgentId = agentId
                    AgentName = model.Agents |> List.tryFind (fun a -> a.Id = agentId) |> Option.map (fun a -> a.Name) |> Option.defaultValue "Agent"
                    Content = sprintf "I understand your request: '%s'. I'm processing this now." content
                    Timestamp = DateTime.Now.AddSeconds(1.0)
                    MessageType = "agent"
                }
                
                { model with ChatMessages = updatedMessages @ [agentResponse] }, Cmd.none
            | None ->
                model, Cmd.none
                
        | MessageReceived message ->
            { model with ChatMessages = model.ChatMessages @ [message] }, Cmd.none
            
        | UpdateGenerationPrompt prompt ->
            { model with GenerationPrompt = prompt }, Cmd.none
            
        | StartUIGeneration ->
            if not model.GenerationInProgress then
                { model with GenerationInProgress = true }, 
                Cmd.ofSub (fun dispatch ->
                    // Simulate UI generation process
                    async {
                        do! Async.Sleep 3000
                        let generatedComponent = {
                            Name = "TarsNetworkVisualization"
                            Type = "NetworkVisualization"
                            Code = "// Generated Elmish component code here"
                            Styling = "/* Generated CSS styles here */"
                            Dependencies = ["Feliz"; "Fable.React"; "Three.js"]
                            GeneratedBy = ["UI Development Team"; "Design Team"; "UX Team"]
                            Timestamp = DateTime.Now
                            Status = "Generated"
                        }
                        dispatch (UIGenerationCompleted generatedComponent)
                    } |> Async.StartImmediate
                )
            else
                model, Cmd.none
                
        | UIGenerationCompleted component ->
            { model with 
                GenerationInProgress = false
                GeneratedComponents = component :: model.GeneratedComponents }, Cmd.none
                
        | UIGenerationFailed error ->
            { model with 
                GenerationInProgress = false
                Errors = error :: model.Errors }, Cmd.none
                
        | UpdateSearchQuery query ->
            { model with SearchQuery = query }, Cmd.none
            
        | SetStatusFilter filter ->
            { model with StatusFilter = filter }, Cmd.none
            
        | SetDepartmentFilter filter ->
            { model with DepartmentFilter = filter }, Cmd.none
            
        | ClearFilters ->
            { model with SearchQuery = ""; StatusFilter = None; DepartmentFilter = None }, Cmd.none
            
        | WebSocketConnected ->
            { model with ConnectionStatus = "Connected" }, Cmd.none
            
        | WebSocketDisconnected ->
            { model with ConnectionStatus = "Disconnected" }, Cmd.none
            
        | AddError error ->
            { model with Errors = error :: model.Errors }, Cmd.none
            
        | ClearError error ->
            { model with Errors = model.Errors |> List.filter ((<>) error) }, Cmd.none
            
        | ClearAllErrors ->
            { model with Errors = [] }, Cmd.none
            
        | Tick time ->
            { model with LastUpdate = time }, Cmd.none
            
        | _ ->
            model, Cmd.none
    
    /// Helper functions for rendering
    module Helpers =
        
        let statusColor = function
            | Active -> "is-success"
            | Busy -> "is-warning" 
            | Error _ -> "is-danger"
            | Idle -> "is-info"
            | Offline -> "is-dark"
            
        let statusIcon = function
            | Active -> "fas fa-check-circle"
            | Busy -> "fas fa-spinner fa-spin"
            | Error _ -> "fas fa-exclamation-triangle"
            | Idle -> "fas fa-pause-circle"
            | Offline -> "fas fa-times-circle"
            
        let formatDateTime (dt: DateTime) =
            dt.ToString("HH:mm:ss")
    
    /// Agent card component
    let agentCard (agent: AgentInfo) (dispatch: Msg -> unit) =
        Bulma.card [
            card.isClickable
            prop.onClick (fun _ -> dispatch (SelectAgent agent))
            prop.children [
                Bulma.cardContent [
                    Bulma.media [
                        Bulma.mediaLeft [
                            Bulma.icon [
                                icon.isLarge
                                color.hasTextPrimary
                                prop.children [
                                    Html.i [
                                        prop.className "fas fa-robot fa-2x"
                                    ]
                                ]
                            ]
                        ]
                        Bulma.mediaContent [
                            Bulma.title.p [
                                title.is5
                                prop.text agent.Name
                            ]
                            Bulma.subtitle.p [
                                subtitle.is6
                                prop.text agent.Type
                            ]
                        ]
                        Bulma.mediaRight [
                            Bulma.tag [
                                prop.className (Helpers.statusColor agent.Status)
                                prop.children [
                                    Html.i [
                                        prop.className (Helpers.statusIcon agent.Status)
                                    ]
                                    Html.span [
                                        prop.style [ style.marginLeft 5 ]
                                        prop.text (string agent.Status)
                                    ]
                                ]
                            ]
                        ]
                    ]
                    
                    Html.div [
                        prop.children [
                            Html.p [
                                prop.children [
                                    Html.strong "Department: "
                                    Html.text agent.Department
                                ]
                            ]
                            match agent.Team with
                            | Some team ->
                                Html.p [
                                    prop.children [
                                        Html.strong "Team: "
                                        Html.text team
                                    ]
                                ]
                            | None -> Html.none
                            
                            match agent.CurrentTask with
                            | Some task ->
                                Html.p [
                                    prop.children [
                                        Html.strong "Current Task: "
                                        Html.text task
                                    ]
                                ]
                            | None -> Html.none
                            
                            Html.p [
                                prop.children [
                                    Html.strong "Performance: "
                                    Html.text (sprintf "%.1f%%" (agent.Performance * 100.0))
                                ]
                            ]
                            
                            Html.p [
                                prop.children [
                                    Html.strong "Last Activity: "
                                    Html.text (Helpers.formatDateTime agent.LastActivity)
                                ]
                            ]
                        ]
                    ]
                ]
                
                Bulma.cardFooter [
                    Bulma.cardFooterItem.a [
                        prop.onClick (fun _ -> dispatch (StartChat agent.Id))
                        prop.children [
                            Html.i [ prop.className "fas fa-comments" ]
                            Html.span [
                                prop.style [ style.marginLeft 5 ]
                                prop.text "Chat"
                            ]
                        ]
                    ]
                    Bulma.cardFooterItem.a [
                        prop.children [
                            Html.i [ prop.className "fas fa-chart-line" ]
                            Html.span [
                                prop.style [ style.marginLeft 5 ]
                                prop.text "Metrics"
                            ]
                        ]
                    ]
                ]
            ]
        ]
    
    /// Main view function
    let view (model: Model) (dispatch: Msg -> unit) =
        Html.div [
            prop.className "tars-ui-app"
            prop.children [
                // Header
                Bulma.navbar [
                    navbar.isFixedTop
                    color.isDark
                    prop.children [
                        Bulma.navbarBrand [
                            Bulma.navbarItem.div [
                                Html.h1 [
                                    prop.className "title is-4 has-text-primary"
                                    prop.children [
                                        Html.i [ prop.className "fas fa-robot" ]
                                        Html.span [
                                            prop.style [ style.marginLeft 10 ]
                                            prop.text "TARS UI Agent System"
                                        ]
                                    ]
                                ]
                            ]
                        ]
                        Bulma.navbarMenu [
                            Bulma.navbarEnd [
                                Bulma.navbarItem.div [
                                    Bulma.tag [
                                        if model.ConnectionStatus = "Connected" then
                                            color.isSuccess
                                        else
                                            color.isDanger
                                        prop.text model.ConnectionStatus
                                    ]
                                ]
                                Bulma.navbarItem.div [
                                    Html.text (sprintf "Last Update: %s" (Helpers.formatDateTime model.LastUpdate))
                                ]
                            ]
                        ]
                    ]
                ]
                
                // Main content
                Html.div [
                    prop.style [ style.marginTop 60; style.padding 20 ]
                    prop.children [
                        Bulma.columns [
                            // Left panel - Controls and filters
                            Bulma.column [
                                column.is3
                                prop.children [
                                    Bulma.box [
                                        Bulma.title.h4 "🎯 UI Generation"
                                        
                                        Bulma.field.div [
                                            Bulma.control.div [
                                                Bulma.textarea [
                                                    prop.placeholder "Describe the UI you want to generate..."
                                                    prop.value model.GenerationPrompt
                                                    prop.onChange (UpdateGenerationPrompt >> dispatch)
                                                    prop.rows 4
                                                ]
                                            ]
                                        ]
                                        
                                        Bulma.field.div [
                                            Bulma.control.div [
                                                Bulma.button.button [
                                                    color.isPrimary
                                                    prop.disabled model.GenerationInProgress
                                                    prop.onClick (fun _ -> dispatch StartUIGeneration)
                                                    prop.children [
                                                        if model.GenerationInProgress then
                                                            Html.i [ prop.className "fas fa-spinner fa-spin" ]
                                                        else
                                                            Html.i [ prop.className "fas fa-magic" ]
                                                        Html.span [
                                                            prop.style [ style.marginLeft 5 ]
                                                            prop.text (if model.GenerationInProgress then "Generating..." else "Generate UI")
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                    
                                    Bulma.box [
                                        Bulma.title.h4 "🔍 Filters"
                                        
                                        Bulma.field.div [
                                            Bulma.control.div [
                                                Bulma.input.text [
                                                    prop.placeholder "Search agents..."
                                                    prop.value model.SearchQuery
                                                    prop.onChange (UpdateSearchQuery >> dispatch)
                                                ]
                                            ]
                                        ]
                                        
                                        Bulma.field.div [
                                            Bulma.control.div [
                                                Bulma.button.button [
                                                    button.isSmall
                                                    color.isLight
                                                    prop.onClick (fun _ -> dispatch ClearFilters)
                                                    prop.text "Clear Filters"
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                            
                            // Center panel - Agent grid
                            Bulma.column [
                                column.is6
                                prop.children [
                                    Bulma.title.h3 "🤖 Active Agents"
                                    
                                    Html.div [
                                        prop.className "agent-grid"
                                        prop.style [ 
                                            style.display.grid
                                            style.gridTemplateColumns "repeat(auto-fill, minmax(300px, 1fr))"
                                            style.gap 20
                                        ]
                                        prop.children [
                                            for agent in model.Agents do
                                                agentCard agent dispatch
                                        ]
                                    ]
                                ]
                            ]
                            
                            // Right panel - Selected agent details and chat
                            Bulma.column [
                                column.is3
                                prop.children [
                                    match model.SelectedAgent with
                                    | Some agent ->
                                        Bulma.box [
                                            Bulma.title.h4 "Agent Details"
                                            Html.div [
                                                prop.children [
                                                    Html.h5 [
                                                        prop.className "title is-5"
                                                        prop.text agent.Name
                                                    ]
                                                    Html.p [ prop.text ("Type: " + agent.Type) ]
                                                    Html.p [ prop.text ("Department: " + agent.Department) ]
                                                    Html.br []
                                                    Html.strong "Capabilities:"
                                                    Html.ul [
                                                        for capability in agent.Capabilities do
                                                            Html.li [ prop.text capability ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    | None ->
                                        Bulma.box [
                                            Html.p [
                                                prop.className "has-text-grey"
                                                prop.text "Select an agent to view details"
                                            ]
                                        ]
                                    
                                    // Chat interface
                                    match model.ActiveChat with
                                    | Some agentId ->
                                        Bulma.box [
                                            Bulma.title.h4 "💬 Agent Chat"
                                            
                                            Html.div [
                                                prop.style [ 
                                                    style.height 200
                                                    style.overflowY.auto
                                                    style.border (1, borderStyle.solid, "hsl(0, 0%, 86%)")
                                                    style.padding 10
                                                    style.marginBottom 10
                                                ]
                                                prop.children [
                                                    for message in model.ChatMessages do
                                                        Html.div [
                                                            prop.style [ style.marginBottom 10 ]
                                                            prop.children [
                                                                Html.strong [
                                                                    prop.text (message.AgentName + ": ")
                                                                ]
                                                                Html.span [ prop.text message.Content ]
                                                                Html.br []
                                                                Html.small [
                                                                    prop.className "has-text-grey"
                                                                    prop.text (Helpers.formatDateTime message.Timestamp)
                                                                ]
                                                            ]
                                                        ]
                                                ]
                                            ]
                                            
                                            Bulma.field.div [
                                                field.hasAddons
                                                prop.children [
                                                    Bulma.control.div [
                                                        control.isExpanded
                                                        prop.children [
                                                            Bulma.input.text [
                                                                prop.placeholder "Type your message..."
                                                                prop.onKeyPress (fun e ->
                                                                    if e.key = "Enter" then
                                                                        let target = e.target :?> Browser.Types.HTMLInputElement
                                                                        dispatch (SendMessage target.value)
                                                                        target.value <- ""
                                                                )
                                                            ]
                                                        ]
                                                    ]
                                                    Bulma.control.div [
                                                        Bulma.button.button [
                                                            color.isPrimary
                                                            prop.onClick (fun _ -> dispatch EndChat)
                                                            prop.text "End"
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    | None -> Html.none
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
