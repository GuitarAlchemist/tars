module TarsEngine.Web.Main

open Fable.Core
open Fable.Core.JsInterop
open Browser.Dom
open TarsEngine.FSharp.Core.GameTheoryElmishModels
open TarsEngine.FSharp.Core.GameTheoryElmishServices
open TarsEngine.FSharp.Core.GameTheory3DIntegrationService

// Web application entry point
let init() =
    console.log("🚀 TARS Web Application Starting...")
    
    // Initialize 3D service
    let serviceFactory = Complete3DServiceFactory()
    let service3D = serviceFactory.CreateComplete3DService()
    let config = serviceFactory.CreateInterstellarConfig("tars-3d-container")
    
    // Generate initialization JavaScript
    let initScript = service3D.Initialize3DSystem(config)
    console.log("✅ 3D System initialized")
    
    // Create sample agents
    let sampleAgents = [
        { AgentId = "Strategic_Alpha"; GameTheoryModel = QuantalResponseEquilibrium(1.2); PerformanceScore = 0.85; ConfidenceLevel = 0.9; IsActive = true }
        { AgentId = "Cognitive_Beta"; GameTheoryModel = CognitiveHierarchy(4); PerformanceScore = 0.78; ConfidenceLevel = 0.85; IsActive = true }
        { AgentId = "Learning_Gamma"; GameTheoryModel = NoRegretLearning(0.95); PerformanceScore = 0.92; ConfidenceLevel = 0.95; IsActive = true }
        { AgentId = "Evolution_Delta"; GameTheoryModel = EvolutionaryGameTheory(0.05); PerformanceScore = 0.71; ConfidenceLevel = 0.8; IsActive = true }
        { AgentId = "Correlated_Epsilon"; GameTheoryModel = CorrelatedEquilibrium([|"signal1"; "signal2"|]); PerformanceScore = 0.88; ConfidenceLevel = 0.87; IsActive = true }
    ]
    
    // Add agents to scene
    let agentScript = service3D.AddAgentsToScene(sampleAgents)
    console.log("✅ Agents added to scene")
    
    // Create coordination state
    let coordination = {
        AverageCoordination = 0.83
        EquilibriumType = "Nash Equilibrium"
        ConvergenceRate = 0.85
        StabilityScore = 0.78
        LastUpdate = System.DateTime.UtcNow
    }
    
    // Create connections
    let connectionScript = service3D.CreateCoordinationConnections(coordination, sampleAgents)
    console.log("✅ Coordination connections created")
    
    // Execute all scripts
    let fullScript = initScript + "\n" + agentScript + "\n" + connectionScript
    Browser.Dom.window?eval(fullScript)
    
    console.log("🎉 TARS Web Application Ready!")

// Start the application when DOM is loaded
let start() =
    if Browser.Dom.document.readyState = "loading" then
        Browser.Dom.document.addEventListener("DOMContentLoaded", fun _ -> init())
    else
        init()

start()
