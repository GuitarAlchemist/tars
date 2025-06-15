namespace TarsEngine.FSharp.Core

open System
open System.Collections.Generic
open TarsEngine.FSharp.Core.GameTheoryElmishModels
open TarsEngine.FSharp.Core.ModernGameTheory

/// Three.js Integration for Game Theory 3D Visualization
module GameTheoryThreeJsIntegration =

    /// 3D Scene Configuration
    type SceneConfig = {
        ContainerId: string
        Width: int
        Height: int
        BackgroundColor: int
        FogColor: int
        FogNear: float
        FogFar: float
        CameraFov: float
        CameraNear: float
        CameraFar: float
        EnableShadows: bool
        EnablePostProcessing: bool
        InterstellarMode: bool
    }

    /// Agent 3D Representation
    type Agent3D = {
        Id: string
        Position: float * float * float
        Velocity: float * float * float
        Color: int
        Size: float
        GameTheoryModel: GameTheoryModel
        PerformanceScore: float
        IsActive: bool
        TrajectoryPoints: (float * float * float) list
        ConnectionStrengths: Map<string, float>
    }

    /// Connection between agents
    type AgentConnection = {
        FromAgent: string
        ToAgent: string
        Strength: float
        Color: int
        Animated: bool
        FlowDirection: float
    }

    /// 3D Scene State
    type Scene3DState = {
        Agents: Map<string, Agent3D>
        Connections: AgentConnection list
        CameraPosition: float * float * float
        CameraTarget: float * float * float
        LightPositions: (float * float * float) list
        ParticleEffects: bool
        CoordinationField: bool
        InterstellarEffects: bool
        AnimationSpeed: float
        LastUpdate: DateTime
    }

    /// WebGPU Shader Definitions
    module Shaders =
        
        /// Vertex shader for agent visualization
        let agentVertexShader = """
            attribute vec3 position;
            attribute vec3 normal;
            attribute vec2 uv;
            
            uniform mat4 modelViewMatrix;
            uniform mat4 projectionMatrix;
            uniform mat3 normalMatrix;
            uniform float time;
            uniform float performanceScore;
            uniform vec3 agentColor;
            
            varying vec3 vNormal;
            varying vec2 vUv;
            varying vec3 vColor;
            varying float vPerformance;
            
            void main() {
                vNormal = normalize(normalMatrix * normal);
                vUv = uv;
                vColor = agentColor;
                vPerformance = performanceScore;
                
                // Add subtle pulsing based on performance
                vec3 pos = position * (1.0 + 0.1 * sin(time * 2.0) * performanceScore);
                
                gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
            }
        """
        
        /// Fragment shader for agent visualization
        let agentFragmentShader = """
            precision highp float;
            
            uniform float time;
            uniform bool interstellarMode;
            
            varying vec3 vNormal;
            varying vec2 vUv;
            varying vec3 vColor;
            varying float vPerformance;
            
            void main() {
                vec3 light = normalize(vec3(1.0, 1.0, 1.0));
                float diff = max(dot(vNormal, light), 0.0);
                
                vec3 color = vColor;
                
                if (interstellarMode) {
                    // Interstellar-style glow effect
                    float glow = 1.0 + 0.5 * sin(time * 3.0 + vPerformance * 10.0);
                    color *= glow;
                    
                    // Add rim lighting
                    float rim = 1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0));
                    color += vec3(0.3, 0.6, 1.0) * pow(rim, 2.0);
                }
                
                gl_FragColor = vec4(color * (0.3 + 0.7 * diff), 1.0);
            }
        """
        
        /// Vertex shader for coordination connections
        let connectionVertexShader = """
            attribute vec3 position;
            attribute float strength;
            
            uniform mat4 modelViewMatrix;
            uniform mat4 projectionMatrix;
            uniform float time;
            uniform float animationSpeed;
            
            varying float vStrength;
            varying float vFlow;
            
            void main() {
                vStrength = strength;
                vFlow = sin(time * animationSpeed + position.x * 0.1);
                
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        """
        
        /// Fragment shader for coordination connections
        let connectionFragmentShader = """
            precision highp float;
            
            uniform bool interstellarMode;
            uniform vec3 connectionColor;
            
            varying float vStrength;
            varying float vFlow;
            
            void main() {
                vec3 color = connectionColor;
                float alpha = vStrength;
                
                if (interstellarMode) {
                    // Flowing energy effect
                    alpha *= (0.5 + 0.5 * vFlow);
                    color += vec3(0.2, 0.4, 0.8) * vFlow;
                }
                
                gl_FragColor = vec4(color, alpha);
            }
        """

    /// Three.js Scene Manager
    type ThreeJsSceneManager() =
        
        let mutable sceneState = {
            Agents = Map.empty
            Connections = []
            CameraPosition = (0.0, 0.0, 10.0)
            CameraTarget = (0.0, 0.0, 0.0)
            LightPositions = [(5.0, 5.0, 5.0); (-5.0, 5.0, 5.0)]
            ParticleEffects = false
            CoordinationField = false
            InterstellarEffects = false
            AnimationSpeed = 1.0
            LastUpdate = DateTime.UtcNow
        }
        
        /// Initialize Three.js scene
        member this.InitializeScene(config: SceneConfig) : string =
            let (camX, camY, camZ) = sceneState.CameraPosition
            sprintf """
                // TARS Game Theory 3D Scene Initialization
                const container = document.getElementById('%s');
                if (!container) {
                    console.error('Container not found: %s');
                    return;
                }

                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x%06x);
                scene.fog = new THREE.Fog(0x%06x, %f, %f);

                // Camera setup
                const camera = new THREE.PerspectiveCamera(%f, container.clientWidth / container.clientHeight, %f, %f);
                camera.position.set(%f, %f, %f);

                // Renderer setup with WebGPU support
                const renderer = new THREE.WebGPURenderer({ antialias: true });
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.shadowMap.enabled = %s;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                container.appendChild(renderer.domElement);

                // Lighting setup
                const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
                scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 5, 5);
                directionalLight.castShadow = %s;
                scene.add(directionalLight);

                // Controls
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;

                // Store references globally for TARS integration
                window.tarsGameTheoryScene = {
                    scene: scene,
                    camera: camera,
                    renderer: renderer,
                    controls: controls,
                    agents: new Map(),
                    connections: [],
                    interstellarMode: %s
                };

                console.log('🌌 TARS Game Theory 3D Scene initialized');
            """
                config.ContainerId config.ContainerId
                config.BackgroundColor config.FogColor config.FogNear config.FogFar
                config.CameraFov config.CameraNear config.CameraFar
                camX camY camZ
                (if config.EnableShadows then "true" else "false")
                (if config.EnableShadows then "true" else "false")
                (if config.InterstellarMode then "true" else "false")
        
        /// Create agent 3D representation
        member this.CreateAgent3D(agent: AgentUIState) : Agent3D =
            let color = 
                match agent.GameTheoryModel with
                | QuantalResponseEquilibrium _ -> 0x4a9eff  // Blue
                | CognitiveHierarchy _ -> 0x00ff88        // Green
                | NoRegretLearning _ -> 0xffaa00          // Orange
                | CorrelatedEquilibrium _ -> 0xff6b6b     // Red
                | EvolutionaryGameTheory _ -> 0x9b59b6    // Purple
                | _ -> 0xffffff                           // White
            
            {
                Id = agent.AgentId
                Position = (0.0, 0.0, 0.0)
                Velocity = (0.0, 0.0, 0.0)
                Color = color
                Size = 0.5 + agent.PerformanceScore * 0.5
                GameTheoryModel = agent.GameTheoryModel
                PerformanceScore = agent.PerformanceScore
                IsActive = agent.IsActive
                TrajectoryPoints = []
                ConnectionStrengths = Map.empty
            }
        
        /// Generate JavaScript for adding agent to scene
        member this.AddAgentToScene(agent: Agent3D) : string =
            let (posX, posY, posZ) = agent.Position
            sprintf """
                // Add agent %s to 3D scene
                if (window.tarsGameTheoryScene) {
                    const scene = window.tarsGameTheoryScene.scene;

                    // Create agent geometry
                    const geometry = new THREE.SphereGeometry(%f, 32, 32);

                    // Create custom material with shaders
                    const material = new THREE.ShaderMaterial({
                        vertexShader: `%s`,
                        fragmentShader: `%s`,
                        uniforms: {
                            time: { value: 0.0 },
                            performanceScore: { value: %f },
                            agentColor: { value: new THREE.Color(0x%06x) },
                            interstellarMode: { value: %s }
                        },
                        transparent: true
                    });

                    const agentMesh = new THREE.Mesh(geometry, material);
                    agentMesh.position.set(%f, %f, %f);
                    agentMesh.castShadow = true;
                    agentMesh.receiveShadow = true;
                    agentMesh.userData = {
                        id: '%s',
                        gameTheoryModel: '%A',
                        performance: %f
                    };

                    scene.add(agentMesh);
                    window.tarsGameTheoryScene.agents.set('%s', agentMesh);

                    console.log('🎯 Added agent %s to 3D scene');
                }
            """
                agent.Id agent.Size
                Shaders.agentVertexShader Shaders.agentFragmentShader
                agent.PerformanceScore agent.Color
                (if sceneState.InterstellarEffects then "true" else "false")
                posX posY posZ
                agent.Id agent.GameTheoryModel agent.PerformanceScore
                agent.Id agent.Id
        
        /// Generate JavaScript for creating connections
        member this.CreateConnections(connections: AgentConnection list) : string =
            let connectionJs = 
                connections
                |> List.map (fun conn ->
                    sprintf """
                        // Create connection from %s to %s
                        if (window.tarsGameTheoryScene) {
                            const scene = window.tarsGameTheoryScene.scene;
                            const fromAgent = window.tarsGameTheoryScene.agents.get('%s');
                            const toAgent = window.tarsGameTheoryScene.agents.get('%s');
                            
                            if (fromAgent && toAgent) {
                                const points = [fromAgent.position, toAgent.position];
                                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                                
                                const material = new THREE.ShaderMaterial({
                                    vertexShader: `%s`,
                                    fragmentShader: `%s`,
                                    uniforms: {
                                        time: { value: 0.0 },
                                        animationSpeed: { value: %f },
                                        connectionColor: { value: new THREE.Color(0x%06x) },
                                        interstellarMode: { value: %s }
                                    },
                                    transparent: true,
                                    opacity: %f
                                });
                                
                                const line = new THREE.Line(geometry, material);
                                scene.add(line);
                                window.tarsGameTheoryScene.connections.push(line);
                            }
                        }
                    """ 
                        conn.FromAgent conn.ToAgent conn.FromAgent conn.ToAgent
                        Shaders.connectionVertexShader Shaders.connectionFragmentShader
                        sceneState.AnimationSpeed conn.Color
                        (if sceneState.InterstellarEffects then "true" else "false")
                        conn.Strength)
                |> String.concat "\n"
            
            connectionJs
        
        /// Generate animation loop JavaScript
        member this.CreateAnimationLoop() : string =
            """
                // TARS Game Theory 3D Animation Loop
                function animateGameTheoryScene() {
                    if (window.tarsGameTheoryScene) {
                        const { scene, camera, renderer, controls } = window.tarsGameTheoryScene;
                        
                        // Update time for shaders
                        const time = Date.now() * 0.001;
                        
                        // Update agent materials
                        window.tarsGameTheoryScene.agents.forEach((agent) => {
                            if (agent.material.uniforms) {
                                agent.material.uniforms.time.value = time;
                            }
                        });
                        
                        // Update connection materials
                        window.tarsGameTheoryScene.connections.forEach((connection) => {
                            if (connection.material.uniforms) {
                                connection.material.uniforms.time.value = time;
                            }
                        });
                        
                        // Update controls
                        controls.update();
                        
                        // Render scene
                        renderer.render(scene, camera);
                    }
                    
                    requestAnimationFrame(animateGameTheoryScene);
                }
                
                // Start animation loop
                animateGameTheoryScene();
                console.log('🎬 TARS Game Theory 3D animation started');
            """
        
        /// Toggle Interstellar mode
        member this.ToggleInterstellarMode(enabled: bool) : string =
            sceneState <- { sceneState with InterstellarEffects = enabled }
            sprintf """
                // Toggle Interstellar Mode
                if (window.tarsGameTheoryScene) {
                    window.tarsGameTheoryScene.interstellarMode = %s;
                    
                    // Update all agent materials
                    window.tarsGameTheoryScene.agents.forEach((agent) => {
                        if (agent.material.uniforms) {
                            agent.material.uniforms.interstellarMode.value = %s;
                        }
                    });
                    
                    // Update all connection materials
                    window.tarsGameTheoryScene.connections.forEach((connection) => {
                        if (connection.material.uniforms) {
                            connection.material.uniforms.interstellarMode.value = %s;
                        }
                    });
                    
                    console.log('🚀 Interstellar mode: %s');
                }
            """ 
                (if enabled then "true" else "false")
                (if enabled then "true" else "false")
                (if enabled then "true" else "false")
                (if enabled then "ENABLED" else "DISABLED")
        
        /// Update agent positions
        member this.UpdateAgentPositions(positions: Map<string, float * float * float>) : string =
            let updateJs = 
                positions
                |> Map.toSeq
                |> Seq.map (fun (agentId, (x, y, z)) ->
                    sprintf """
                        // Update position for agent %s
                        const agent = window.tarsGameTheoryScene.agents.get('%s');
                        if (agent) {
                            agent.position.set(%f, %f, %f);
                        }
                    """ agentId agentId x y z)
                |> String.concat "\n"
            
            updateJs
        
        /// Get current scene state
        member this.GetSceneState() = sceneState
        
        /// Update scene state
        member this.UpdateSceneState(newState: Scene3DState) =
            sceneState <- newState
