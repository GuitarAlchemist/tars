namespace TarsEngine.FSharp.Core.Visualization

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Tracing.AgenticTraceCapture

/// 3D Visualization Engine for TARS with Interstellar-style interfaces
/// Provides real-time visualization of grammar evolution, auto-improvement, and FLUX operations
module ThreeDVisualizationEngine =

    // ============================================================================
    // 3D VISUALIZATION TYPES
    // ============================================================================

    /// 3D coordinate system
    type Vector3D = {
        X: float
        Y: float
        Z: float
    }

    /// 3D rotation in Euler angles
    type Rotation3D = {
        Pitch: float // X-axis rotation
        Yaw: float   // Y-axis rotation
        Roll: float  // Z-axis rotation
    }

    /// 3D transformation matrix
    type Transform3D = {
        Position: Vector3D
        Rotation: Rotation3D
        Scale: Vector3D
    }

    /// TARS robot visual components (Interstellar-style)
    type TarsRobotComponent =
        | MainBody of dimensions: Vector3D * material: string
        | RotatingSegment of radius: float * height: float * rotationSpeed: float
        | DataPanel of width: float * height: float * content: string
        | ProcessingCore of radius: float * pulseRate: float * activity: float
        | CommunicationArray of antennaCount: int * signalStrength: float

    /// Visualization scene types
    type VisualizationScene =
        | GrammarEvolutionScene of currentTier: int * domains: string list * performance: float
        | AutoImprovementScene of engines: string list * progress: Map<string, float>
        | FluxExecutionScene of languageMode: string * complexity: float * tierLevel: int
        | SystemOverviewScene of components: string list * health: float
        | InteractiveControlScene of availableCommands: string list

    /// 3D visualization result
    type VisualizationResult = {
        Success: bool
        SceneData: string
        RenderTime: TimeSpan
        FrameRate: float
        InteractionPoints: Vector3D list
        TarsRobotState: Map<string, obj>
        WebGLCode: string option
        D3JSCode: string option
    }

    // ============================================================================
    // 3D VISUALIZATION ENGINE
    // ============================================================================

    /// 3D Visualization Engine for TARS
    type ThreeDVisualizationEngine() =
        let mutable currentScene = SystemOverviewScene (["Grammar"; "AutoImprovement"; "FLUX"], 1.0)
        let mutable tarsRobotTransform = {
            Position = { X = 0.0; Y = 0.0; Z = 0.0 }
            Rotation = { Pitch = 0.0; Yaw = 0.0; Roll = 0.0 }
            Scale = { X = 1.0; Y = 1.0; Z = 1.0 }
        }
        let mutable animationTime = 0.0

        /// Generate TARS robot 3D model (Interstellar-style)
        member this.GenerateTarsRobot(activity: float) : string =
            let bodyHeight = 2.0 + (activity * 0.5) // Robot grows with activity
            let segmentRotation = animationTime * 45.0 // Rotating segments
            let coreActivity = activity * 100.0

            sprintf "// TARS Robot 3D Model - Interstellar Style\n// Activity Level: %.1f%% | Animation Time: %.2f\n\nconst tarsRobot = {\n  mainBody: {\n    geometry: new THREE.BoxGeometry(0.8, %.2f, 0.4),\n    material: new THREE.MeshPhongMaterial({ color: 0x2c3e50 }),\n    position: [0, 0, 0]\n  },\n  processingCore: {\n    geometry: new THREE.SphereGeometry(0.15),\n    material: new THREE.MeshPhongMaterial({ color: 0x3498db, emissiveIntensity: %.2f }),\n    position: [0, 0, 0.25]\n  }\n};" coreActivity animationTime bodyHeight (activity * 0.5)

        /// Generate WebGL scene for grammar evolution visualization
        member this.GenerateGrammarEvolutionScene(currentTier: int, domains: string list, performance: float) : string =
            let tierHeight = float currentTier * 0.3

            sprintf "// Grammar Evolution 3D Scene\n// Tier: %d | Domains: %s | Performance: %.1f%%\n\nconst grammarScene = new THREE.Scene();\nconst camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);\nconst renderer = new THREE.WebGLRenderer({ antialias: true });\n\n// Tier visualization\nconst tierGeometry = new THREE.CylinderGeometry(0.5, 0.8, %.2f);\nconst tierMaterial = new THREE.MeshPhongMaterial({ color: 0x2ecc71, opacity: %.2f });\nconst tierTower = new THREE.Mesh(tierGeometry, tierMaterial);\ntierTower.position.set(0, %.2f / 2, 0);\ngrammarScene.add(tierTower);\n\n// Lighting\nconst ambientLight = new THREE.AmbientLight(0x404040, 0.6);\nconst directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);\ndirectionalLight.position.set(10, 10, 5);\ngrammarScene.add(ambientLight);\ngrammarScene.add(directionalLight);\n\ncamera.position.set(5, 3, 5);\ncamera.lookAt(0, 0, 0);"
                currentTier
                (String.concat ", " domains)
                (performance * 100.0)
                tierHeight
                (performance * 0.8 + 0.2)
                tierHeight

        /// Generate auto-improvement visualization scene
        member this.GenerateAutoImprovementScene(engines: string list, progress: Map<string, float>) : string =
            let avgProgress = if progress.IsEmpty then 0.0 else progress |> Map.values |> Seq.average

            sprintf "// Auto-Improvement 3D Scene\nconst autoImprovementScene = new THREE.Scene();\n\n// Engine visualizations for %d engines\n// Average progress: %.1f%%\n\n// Central TARS robot\n%s\n\n// Lighting\nconst ambientLight = new THREE.AmbientLight(0x404040, 0.6);\nconst directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);\ndirectionalLight.position.set(10, 10, 5);\nautoImprovementScene.add(ambientLight);\nautoImprovementScene.add(directionalLight);"
                engines.Length
                (avgProgress * 100.0)
                (this.GenerateTarsRobot(0.8))

        /// Render 3D visualization scene
        member this.RenderScene(scene: VisualizationScene) : Task<VisualizationResult> = task {
            let startTime = DateTime.UtcNow
            animationTime <- animationTime + 0.016 // 60 FPS increment

            try
                let (sceneData, webglCode) =
                    match scene with
                    | GrammarEvolutionScene (tier, domains, performance) ->
                        let sceneCode = this.GenerateGrammarEvolutionScene(tier, domains, performance)
                        (sprintf "Grammar Evolution: Tier %d, %d domains, %.1f%% performance" tier domains.Length (performance * 100.0), Some sceneCode)

                    | AutoImprovementScene (engines, progress) ->
                        let sceneCode = this.GenerateAutoImprovementScene(engines, progress)
                        let avgProgress = if progress.IsEmpty then 0.0 else progress |> Map.values |> Seq.average
                        (sprintf "Auto-Improvement: %d engines, %.1f%% average progress" engines.Length (avgProgress * 100.0), Some sceneCode)

                    | FluxExecutionScene (languageMode, complexity, tierLevel) ->
                        let sceneCode = sprintf "// FLUX Execution Scene - %s\nconst fluxScene = new THREE.Scene();\nconst fluxComplexity = %.2f;\nconst fluxTier = %d;" languageMode complexity tierLevel
                        (sprintf "FLUX Execution: %s, complexity %.2f, tier %d" languageMode complexity tierLevel, Some sceneCode)

                    | SystemOverviewScene (components, health) ->
                        let sceneCode = sprintf "// System Overview Scene\nconst systemScene = new THREE.Scene();\nconst systemHealth = %.2f;\n\n// Central TARS robot\n%s" health (this.GenerateTarsRobot(health))
                        (sprintf "System Overview: %d components, %.1f%% health" components.Length (health * 100.0), Some sceneCode)

                    | InteractiveControlScene (commands) ->
                        let sceneCode = sprintf "// Interactive Control Scene\nconst controlScene = new THREE.Scene();\n\n// Interactive TARS robot\n%s" (this.GenerateTarsRobot(1.0))
                        (sprintf "Interactive Control: %d commands available" commands.Length, Some sceneCode)

                // Generate D3.js complementary visualization
                let d3Code = sprintf "// D3.js Data Visualization Complement\nconst d3Container = d3.select('#tars-data-viz');\n// Animation time: %.2f" animationTime

                // Calculate interaction points (clickable areas)
                let interactionPoints = [
                    { X = 0.0; Y = 0.0; Z = 0.0 }  // TARS robot center
                    { X = 2.0; Y = 1.0; Z = 0.0 }  // Control panel
                    { X = -2.0; Y = 1.0; Z = 0.0 } // Status panel
                ]

                let result = {
                    Success = true
                    SceneData = sceneData
                    RenderTime = DateTime.UtcNow - startTime
                    FrameRate = 60.0
                    InteractionPoints = interactionPoints
                    TarsRobotState = Map.ofList [
                        ("position", tarsRobotTransform.Position :> obj)
                        ("rotation", tarsRobotTransform.Rotation :> obj)
                        ("animation_time", animationTime :> obj)
                        ("activity_level", 0.8 :> obj)
                    ]
                    WebGLCode = webglCode
                    D3JSCode = Some d3Code
                }

                GlobalTraceCapture.LogAgentEvent(
                    "3d_visualization_engine",
                    "SceneRendered",
                    sprintf "Rendered 3D scene: %s" sceneData,
                    Map.ofList [("scene_type", sprintf "%A" scene :> obj); ("render_time", result.RenderTime.TotalMilliseconds :> obj)],
                    Map.ofList [("frame_rate", result.FrameRate); ("animation_time", animationTime)],
                    result.FrameRate / 60.0,
                    14,
                    []
                )

                return result

            with
            | ex ->
                let errorResult = {
                    Success = false
                    SceneData = sprintf "3D visualization failed: %s" ex.Message
                    RenderTime = DateTime.UtcNow - startTime
                    FrameRate = 0.0
                    InteractionPoints = []
                    TarsRobotState = Map.empty
                    WebGLCode = None
                    D3JSCode = None
                }

                GlobalTraceCapture.LogAgentEvent(
                    "3d_visualization_engine",
                    "SceneRenderError",
                    sprintf "3D visualization failed: %s" ex.Message,
                    Map.ofList [("error", ex.Message :> obj)],
                    Map.empty,
                    0.0,
                    14,
                    []
                )

                return errorResult
        }

        /// Update TARS robot animation
        member this.UpdateTarsRobotAnimation(activity: float) : unit =
            // Update robot transform based on activity
            tarsRobotTransform <- {
                tarsRobotTransform with
                    Rotation = {
                        tarsRobotTransform.Rotation with
                            Yaw = tarsRobotTransform.Rotation.Yaw + (activity * 2.0)
                    }
                    Scale = {
                        X = 1.0 + (activity * 0.1)
                        Y = 1.0 + (activity * 0.1) 
                        Z = 1.0 + (activity * 0.1)
                    }
            }

        /// Set current visualization scene
        member this.SetScene(scene: VisualizationScene) : unit =
            currentScene <- scene

        /// Get current visualization statistics
        member this.GetVisualizationStats() : Map<string, obj> =
            Map.ofList [
                ("current_scene", sprintf "%A" currentScene :> obj)
                ("animation_time", animationTime :> obj)
                ("tars_robot_position", tarsRobotTransform.Position :> obj)
                ("tars_robot_rotation", tarsRobotTransform.Rotation :> obj)
                ("supported_scenes", ["Grammar Evolution"; "Auto-Improvement"; "FLUX Execution"; "System Overview"; "Interactive Control"] :> obj)
                ("rendering_engine", "Three.js WebGL + D3.js" :> obj)
                ("frame_rate", 60.0 :> obj)
            ]

    /// 3D Visualization service for TARS
    type ThreeDVisualizationService() =
        let visualizationEngine = ThreeDVisualizationEngine()

        /// Render TARS 3D visualization
        member this.RenderVisualization(scene: VisualizationScene) : Task<VisualizationResult> =
            visualizationEngine.RenderScene(scene)

        /// Update TARS robot animation
        member this.UpdateRobotAnimation(activity: float) : unit =
            visualizationEngine.UpdateTarsRobotAnimation(activity)

        /// Get visualization status
        member this.GetVisualizationStatus() : Map<string, obj> =
            visualizationEngine.GetVisualizationStats()
