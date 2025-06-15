namespace TarsEngine.FSharp.UI.Pages

open System
open Feliz
open Feliz.MaterialUI
open TarsEngine.FSharp.UI.Types

/// Dashboard page showing system overview
module Dashboard =
    
    [<ReactComponent>]
    let Dashboard (model: Model) (dispatch: Msg -> unit) =
        let (refreshing, setRefreshing) = React.useState(false)
        
        let handleRefresh () =
            setRefreshing(true)
            dispatch Refresh
            async {
                do! Async.Sleep(1000)
                setRefreshing(false)
            } |> Async.StartImmediate
        
        let getSystemStatusColor () =
            if model.WebSocketConnected && model.SemanticKernelReady then "success"
            elif model.WebSocketConnected || model.SemanticKernelReady then "warning"
            else "error"
        
        let getSystemStatusText () =
            match model.WebSocketConnected, model.SemanticKernelReady with
            | true, true -> "All Systems Operational"
            | true, false -> "WebSocket Connected, Semantic Kernel Initializing"
            | false, true -> "Semantic Kernel Ready, WebSocket Disconnected"
            | false, false -> "Systems Initializing"
        
        Mui.container [
            container.maxWidth.lg
            container.children [
                // Header
                Mui.box [
                    box.sx [
                        sx.display.flex
                        sx.justifyContent.spaceBetween
                        sx.alignItems.center
                        sx.marginBottom 3
                    ]
                    box.children [
                        Mui.typography [
                            typography.variant.h4
                            typography.component' "h1"
                            typography.children [ "TARS Dashboard" ]
                        ]
                        Mui.button [
                            button.variant.outlined
                            button.startIcon (Mui.icon [ Icons.refresh ])
                            button.onClick (fun _ -> handleRefresh())
                            button.disabled refreshing
                            button.children [
                                if refreshing then "Refreshing..." else "Refresh"
                            ]
                        ]
                    ]
                ]
                
                // System Status Card
                Mui.card [
                    card.sx [ sx.marginBottom 3 ]
                    card.children [
                        Mui.cardContent [
                            cardContent.children [
                                Mui.box [
                                    box.sx [
                                        sx.display.flex
                                        sx.alignItems.center
                                        sx.marginBottom 2
                                    ]
                                    box.children [
                                        Mui.avatar [
                                            avatar.sx [
                                                sx.width 48
                                                sx.height 48
                                                sx.marginRight 2
                                                sx.backgroundColor (getSystemStatusColor() + ".main")
                                            ]
                                            avatar.children [
                                                Mui.icon [ Icons.computer ]
                                            ]
                                        ]
                                        Mui.box [
                                            box.children [
                                                Mui.typography [
                                                    typography.variant.h6
                                                    typography.children [ "System Status" ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.body2
                                                    typography.color.textSecondary
                                                    typography.children [ getSystemStatusText() ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                                
                                Mui.grid [
                                    grid.container true
                                    grid.spacing 2
                                    grid.children [
                                        Mui.grid [
                                            grid.item true
                                            grid.xs 6
                                            grid.children [
                                                Mui.box [
                                                    box.sx [
                                                        sx.display.flex
                                                        sx.alignItems.center
                                                    ]
                                                    box.children [
                                                        Mui.icon [
                                                            icon.sx [
                                                                sx.marginRight 1
                                                                sx.color (if model.WebSocketConnected then "success.main" else "error.main")
                                                            ]
                                                            icon.children [
                                                                if model.WebSocketConnected then Icons.wifi else Icons.wifiOff
                                                            ]
                                                        ]
                                                        Mui.typography [
                                                            typography.variant.body2
                                                            typography.children [
                                                                "WebSocket: " + (if model.WebSocketConnected then "Connected" else "Disconnected")
                                                            ]
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                        Mui.grid [
                                            grid.item true
                                            grid.xs 6
                                            grid.children [
                                                Mui.box [
                                                    box.sx [
                                                        sx.display.flex
                                                        sx.alignItems.center
                                                    ]
                                                    box.children [
                                                        Mui.icon [
                                                            icon.sx [
                                                                sx.marginRight 1
                                                                sx.color (if model.SemanticKernelReady then "success.main" else "error.main")
                                                            ]
                                                            icon.children [
                                                                if model.SemanticKernelReady then Icons.smartToy else Icons.smartToyOutlined
                                                            ]
                                                        ]
                                                        Mui.typography [
                                                            typography.variant.body2
                                                            typography.children [
                                                                "Semantic Kernel: " + (if model.SemanticKernelReady then "Ready" else "Initializing")
                                                            ]
                                                        ]
                                                    ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
                
                // Quick Stats Grid
                Mui.grid [
                    grid.container true
                    grid.spacing 3
                    grid.sx [ sx.marginBottom 3 ]
                    grid.children [
                        // Agents Card
                        Mui.grid [
                            grid.item true
                            grid.xs 12
                            grid.sm 6
                            grid.md 3
                            grid.children [
                                Mui.card [
                                    card.sx [
                                        sx.height "100%"
                                        sx.cursor.pointer
                                    ]
                                    card.onClick (fun _ -> dispatch (NavigateTo Agents))
                                    card.children [
                                        Mui.cardContent [
                                            cardContent.sx [ sx.textAlign.center ]
                                            cardContent.children [
                                                Mui.avatar [
                                                    avatar.sx [
                                                        sx.width 56
                                                        sx.height 56
                                                        sx.margin "0 auto 16px"
                                                        sx.backgroundColor "primary.main"
                                                    ]
                                                    avatar.children [
                                                        Mui.icon [ Icons.group ]
                                                    ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.h4
                                                    typography.children [ string model.Agents.Length ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.body2
                                                    typography.color.textSecondary
                                                    typography.children [ "Active Agents" ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                        
                        // Metascripts Card
                        Mui.grid [
                            grid.item true
                            grid.xs 12
                            grid.sm 6
                            grid.md 3
                            grid.children [
                                Mui.card [
                                    card.sx [
                                        sx.height "100%"
                                        sx.cursor.pointer
                                    ]
                                    card.onClick (fun _ -> dispatch (NavigateTo Metascripts))
                                    card.children [
                                        Mui.cardContent [
                                            cardContent.sx [ sx.textAlign.center ]
                                            cardContent.children [
                                                Mui.avatar [
                                                    avatar.sx [
                                                        sx.width 56
                                                        sx.height 56
                                                        sx.margin "0 auto 16px"
                                                        sx.backgroundColor "secondary.main"
                                                    ]
                                                    avatar.children [
                                                        Mui.icon [ Icons.code ]
                                                    ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.h4
                                                    typography.children [ string model.Metascripts.Length ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.body2
                                                    typography.color.textSecondary
                                                    typography.children [ "Metascripts" ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                        
                        // Nodes Card
                        Mui.grid [
                            grid.item true
                            grid.xs 12
                            grid.sm 6
                            grid.md 3
                            grid.children [
                                Mui.card [
                                    card.sx [
                                        sx.height "100%"
                                        sx.cursor.pointer
                                    ]
                                    card.onClick (fun _ -> dispatch (NavigateTo Nodes))
                                    card.children [
                                        Mui.cardContent [
                                            cardContent.sx [ sx.textAlign.center ]
                                            cardContent.children [
                                                Mui.avatar [
                                                    avatar.sx [
                                                        sx.width 56
                                                        sx.height 56
                                                        sx.margin "0 auto 16px"
                                                        sx.backgroundColor "info.main"
                                                    ]
                                                    avatar.children [
                                                        Mui.icon [ Icons.computer ]
                                                    ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.h4
                                                    typography.children [ string model.Nodes.Length ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.body2
                                                    typography.color.textSecondary
                                                    typography.children [ "Connected Nodes" ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                        
                        // Chat Messages Card
                        Mui.grid [
                            grid.item true
                            grid.xs 12
                            grid.sm 6
                            grid.md 3
                            grid.children [
                                Mui.card [
                                    card.sx [
                                        sx.height "100%"
                                        sx.cursor.pointer
                                    ]
                                    card.onClick (fun _ -> dispatch (NavigateTo Chat))
                                    card.children [
                                        Mui.cardContent [
                                            cardContent.sx [ sx.textAlign.center ]
                                            cardContent.children [
                                                Mui.avatar [
                                                    avatar.sx [
                                                        sx.width 56
                                                        sx.height 56
                                                        sx.margin "0 auto 16px"
                                                        sx.backgroundColor "success.main"
                                                    ]
                                                    avatar.children [
                                                        Mui.icon [ Icons.chat ]
                                                    ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.h4
                                                    typography.children [ string model.ChatHistory.Length ]
                                                ]
                                                Mui.typography [
                                                    typography.variant.body2
                                                    typography.color.textSecondary
                                                    typography.children [ "Chat Messages" ]
                                                ]
                                            ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
                
                // Recent Activity
                Mui.card [
                    card.children [
                        Mui.cardHeader [
                            cardHeader.title "Recent Activity"
                            cardHeader.action (
                                Mui.iconButton [
                                    iconButton.onClick (fun _ -> dispatch (NavigateTo Chat))
                                    iconButton.children [
                                        Mui.icon [ Icons.openInNew ]
                                    ]
                                ]
                            )
                        ]
                        Mui.cardContent [
                            cardContent.children [
                                if List.isEmpty model.ChatHistory then
                                    Mui.box [
                                        box.sx [
                                            sx.textAlign.center
                                            sx.padding 4
                                        ]
                                        box.children [
                                            Mui.typography [
                                                typography.variant.body2
                                                typography.color.textSecondary
                                                typography.children [ "No recent activity. Start a conversation with TARS!" ]
                                            ]
                                            Mui.button [
                                                button.variant.contained
                                                button.sx [ sx.marginTop 2 ]
                                                button.onClick (fun _ -> dispatch (NavigateTo Chat))
                                                button.children [ "Start Chat" ]
                                            ]
                                        ]
                                    ]
                                else
                                    Mui.list [
                                        list.children [
                                            for message in model.ChatHistory |> List.rev |> List.take (min 5 model.ChatHistory.Length) do
                                                Mui.listItem [
                                                    listItem.children [
                                                        Mui.listItemAvatar [
                                                            listItemAvatar.children [
                                                                Mui.avatar [
                                                                    avatar.sx [
                                                                        sx.backgroundColor (if message.IsFromUser then "primary.main" else "secondary.main")
                                                                    ]
                                                                    avatar.children [
                                                                        Mui.icon [
                                                                            if message.IsFromUser then Icons.person else Icons.smartToy
                                                                        ]
                                                                    ]
                                                                ]
                                                            ]
                                                        ]
                                                        Mui.listItemText [
                                                            listItemText.primary (
                                                                if message.Content.Length > 100 then
                                                                    message.Content.Substring(0, 100) + "..."
                                                                else
                                                                    message.Content
                                                            )
                                                            listItemText.secondary (
                                                                $"{if message.IsFromUser then "You" else "TARS"} • {message.Timestamp.ToString("HH:mm:ss")}"
                                                            )
                                                        ]
                                                    ]
                                                ]
                                        ]
                                    ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
