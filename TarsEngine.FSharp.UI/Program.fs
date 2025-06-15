module TarsEngine.FSharp.UI.Program

open System
open Elmish
open Elmish.React
open Fable.Core.JsInterop
open TarsEngine.FSharp.UI.Types
open TarsEngine.FSharp.UI.App

// Import CSS
importSideEffects "./style.css"

/// Subscription for real-time updates
let subscription (model: Model) =
    let sub dispatch =
        // Timer for periodic updates
        let timer = Browser.Dom.window.setInterval((fun () ->
            dispatch (Tick DateTime.Now)
        ), 5000) // Update every 5 seconds
        
        // WebSocket connection for real-time agent updates
        let wsUrl = "ws://localhost:8080/tars-updates"
        try
            let ws = Browser.WebSocket.WebSocket.Create(wsUrl)
            
            ws.onopen <- fun _ ->
                Browser.Dom.console.log("WebSocket connected to TARS")
                dispatch WebSocketConnected
            
            ws.onclose <- fun _ ->
                Browser.Dom.console.log("WebSocket disconnected from TARS")
                dispatch WebSocketDisconnected
            
            ws.onmessage <- fun event ->
                let data = event.data :?> string
                dispatch (WebSocketMessage data)
            
            ws.onerror <- fun error ->
                Browser.Dom.console.error("WebSocket error:", error)
                dispatch (AddError "WebSocket connection failed")
        with
        | ex ->
            Browser.Dom.console.warn("WebSocket not available, using polling mode")
            dispatch (AddError "Real-time updates unavailable")
        
        // Cleanup function
        { new System.IDisposable with
            member _.Dispose() = 
                Browser.Dom.window.clearInterval(timer)
        }
    
    Cmd.ofSub sub

/// Enhanced update function with command handling
let updateWithCommands (msg: Msg) (model: Model) : Model * Cmd<Msg> =
    let newModel, cmd = update msg model
    
    // Handle additional side effects based on commands
    let additionalCmds = 
        match msg with
        | WebSocketMessage data ->
            try
                // Parse WebSocket message and update agent status
                let parsed = Thoth.Json.Decode.fromString (Thoth.Json.Decode.field "type" Thoth.Json.Decode.string) data
                match parsed with
                | Ok "agent_status_update" ->
                    let agentUpdate = Thoth.Json.Decode.fromString 
                        (Thoth.Json.Decode.object (fun get ->
                            let agentId = get.Required.Field "agentId" Thoth.Json.Decode.string
                            let status = get.Required.Field "status" Thoth.Json.Decode.string
                            let newStatus = 
                                match status with
                                | "active" -> Active
                                | "busy" -> Busy
                                | "idle" -> Idle
                                | "offline" -> Offline
                                | error -> Error error
                            (agentId, newStatus)
                        )) data
                    match agentUpdate with
                    | Ok (agentId, status) -> [Cmd.ofMsg (AgentStatusChanged (agentId, status))]
                    | Error _ -> []
                | _ -> []
            with
            | _ -> []
            
        | StartUIGeneration when not model.GenerationInProgress ->
            // Simulate agent coordination for UI generation
            [
                Cmd.ofSub (fun dispatch ->
                    async {
                        // Phase 1: Requirements analysis
                        do! Async.Sleep 1000
                        dispatch (AddError "UI Dev Team: Analyzing requirements...")
                        
                        // Phase 2: Design specification
                        do! Async.Sleep 1000
                        dispatch (AddError "Design Team: Creating visual specifications...")
                        
                        // Phase 3: Component generation
                        do! Async.Sleep 1000
                        dispatch (AddError "UX Team: Ensuring accessibility compliance...")
                        
                        // Phase 4: Code generation
                        let generatedComponent = {
                            Name = "TarsNetworkVisualization"
                            Type = "NetworkVisualization"
                            Code = sprintf """
// Generated Elmish component for: %s
module TarsNetworkVisualization

open Feliz
open Feliz.Bulma

let networkVisualization (nodes: NetworkNode list) (dispatch: Msg -> unit) =
    Html.div [
        prop.className "network-visualization"
        prop.style [
            style.width (length.percent 100)
            style.height 600
            style.position.relative
            style.backgroundColor "#0a0a0a"
            style.borderRadius 8
            style.overflow.hidden
        ]
        prop.children [
            // Network nodes
            for node in nodes do
                Html.div [
                    prop.className "network-node"
                    prop.style [
                        style.position.absolute
                        style.left (fst node.Position)
                        style.top (snd node.Position)
                        style.width 60
                        style.height 60
                        style.borderRadius (length.percent 50)
                        style.backgroundColor (
                            match node.Type with
                            | "core" -> "#00ff88"
                            | "department" -> "#0088ff"
                            | "team" -> "#ff8800"
                            | "agent" -> "#ff4444"
                            | _ -> "#666666"
                        )
                        style.display.flex
                        style.alignItems.center
                        style.justifyContent.center
                        style.cursor.pointer
                        style.transition "all 0.3s ease"
                    ]
                    prop.onClick (fun _ -> dispatch (SelectNode node))
                    prop.children [
                        Html.span [
                            prop.style [
                                style.fontSize 12
                                style.fontWeight.bold
                                style.color "#ffffff"
                                style.textAlign.center
                            ]
                            prop.text node.Name
                        ]
                    ]
                ]
        ]
    ]
""" model.GenerationPrompt
                            Styling = """
.network-visualization {
    background: radial-gradient(circle at center, rgba(0, 255, 136, 0.1) 0%, transparent 70%);
}

.network-node:hover {
    transform: scale(1.2);
    box-shadow: 0 0 20px currentColor;
    z-index: 10;
}

.network-node {
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
"""
                            Dependencies = ["Feliz"; "Feliz.Bulma"; "Fable.React"]
                            GeneratedBy = ["UI Development Team"; "Design Team"; "UX Team"]
                            Timestamp = DateTime.Now
                            Status = "Generated"
                        }
                        
                        dispatch (UIGenerationCompleted generatedComponent)
                        dispatch (ClearAllErrors)
                    } |> Async.StartImmediate
                )
            ]
            
        | _ -> []
    
    newModel, Cmd.batch (cmd :: additionalCmds)

/// Initialize the application
let init () = 
    let model, cmds = Types.init ()
    model, Cmd.batch [
        yield! cmds |> List.map (function
            | LoadAgentsCmd -> Cmd.ofMsg LoadAgents
            | LoadDepartmentsCmd -> Cmd.ofMsg LoadDepartments
            | LoadTeamsCmd -> Cmd.ofMsg LoadTeams
            | LoadNetworkNodesCmd -> Cmd.ofMsg LoadNetworkNodes
            | ConnectWebSocketCmd -> Cmd.none // Handled by subscription
            | _ -> Cmd.none
        )
    ]

/// Configure the Elmish program
let program =
    Program.mkProgram init updateWithCommands view
    |> Program.withSubscription subscription
    |> Program.withReactSynchronous "tars-ui-root"
    |> Program.withConsoleTrace

/// Start the application
program |> Program.run
