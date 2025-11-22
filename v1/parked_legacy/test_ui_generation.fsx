// Test AI-Driven UI Generation
#r "TarsEngine.FSharp.Cli/bin/Debug/net9.0/TarsEngine.FSharp.Cli.dll"

open System
open System.IO
open TarsEngine.FSharp.Cli.UI.TarsElmishGenerator

// Load the sample UI DSL
let dslContent = """ui {
  view_id: "TarsAgentDashboard"
  title: "TARS Agent Activity Dashboard"
  feedback_enabled: true
  real_time_updates: true

  header "TARS Agent Monitoring System"

  metrics_panel bind(cognitiveMetrics)

  thought_flow bind(thoughtPatterns)

  table bind(agentRows)

  button "Refresh Data" on refreshClicked

  line_chart bind(agentPerformance)

  threejs bind(agent3DVisualization)

  chat_panel bind(agentCommunication)

  projects_panel bind(activeProjects)

  diagnostics_panel bind(systemDiagnostics)
}"""

printfn "🎨 TARS AI-Driven UI Generation Test"
printfn "======================================"
printfn ""
printfn "📝 Input DSL:"
printfn "%s" dslContent
printfn ""

// Parse and generate Elmish code
try
    let generatedCode = processUiDsl dslContent
    
    printfn "✅ Generated Elmish Code:"
    printfn "=========================="
    printfn "%s" generatedCode
    
    // Save the generated code
    File.WriteAllText("Generated_TarsAgentDashboard.fs", generatedCode)
    printfn ""
    printfn "💾 Generated code saved to: Generated_TarsAgentDashboard.fs"
    printfn "📏 Generated code size: %d characters" generatedCode.Length
    
with
| ex ->
    printfn "❌ Error: %s" ex.Message
