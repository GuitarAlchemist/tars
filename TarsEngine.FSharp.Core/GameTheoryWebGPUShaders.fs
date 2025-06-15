namespace TarsEngine.FSharp.Core

open System
open TarsEngine.FSharp.Core.GameTheoryElmishModels

/// WebGPU Compute Shaders for Advanced Game Theory Visualization
module GameTheoryWebGPUShaders =

    /// WebGPU Compute Shader for Agent Coordination Field
    let coordinationFieldComputeShader = """
        @group(0) @binding(0) var<storage, read> agentPositions: array<vec3<f32>>;
        @group(0) @binding(1) var<storage, read> agentPerformances: array<f32>;
        @group(0) @binding(2) var<storage, read_write> coordinationField: array<vec4<f32>>;
        @group(0) @binding(3) var<uniform> params: CoordinationParams;
        
        struct CoordinationParams {
            numAgents: u32,
            fieldResolution: u32,
            fieldSize: f32,
            time: f32,
            coordinationStrength: f32,
            interstellarMode: u32,
        }
        
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let fieldIndex = global_id.x + global_id.y * params.fieldResolution;
            if (fieldIndex >= params.fieldResolution * params.fieldResolution) {
                return;
            }
            
            // Calculate field position
            let fieldPos = vec3<f32>(
                (f32(global_id.x) / f32(params.fieldResolution) - 0.5) * params.fieldSize,
                (f32(global_id.y) / f32(params.fieldResolution) - 0.5) * params.fieldSize,
                0.0
            );
            
            var coordination = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            
            // Calculate coordination influence from all agents
            for (var i = 0u; i < params.numAgents; i++) {
                let agentPos = agentPositions[i];
                let distance = length(fieldPos - agentPos);
                let performance = agentPerformances[i];
                
                // Coordination field strength based on distance and performance
                let influence = performance * exp(-distance * 2.0);
                
                // Color based on game theory model (encoded in performance)
                let hue = performance * 6.28318; // 2π for full color wheel
                let color = vec3<f32>(
                    0.5 + 0.5 * cos(hue),
                    0.5 + 0.5 * cos(hue + 2.094), // 2π/3
                    0.5 + 0.5 * cos(hue + 4.188)  // 4π/3
                );
                
                coordination += vec4<f32>(color * influence, influence);
            }
            
            // Interstellar mode effects
            if (params.interstellarMode == 1u) {
                let wave = sin(params.time * 2.0 + fieldPos.x * 0.1 + fieldPos.y * 0.1);
                coordination.w *= (1.0 + 0.3 * wave);
                
                // Add flowing energy patterns
                let flow = sin(params.time * 3.0 + fieldPos.x * 0.2) * cos(params.time * 2.5 + fieldPos.y * 0.15);
                coordination.xyz += vec3<f32>(0.1, 0.2, 0.4) * flow * coordination.w;
            }
            
            coordinationField[fieldIndex] = coordination;
        }
    """

    /// WebGPU Compute Shader for Agent Trajectory Calculation
    let trajectoryComputeShader = """
        @group(0) @binding(0) var<storage, read> agentStates: array<AgentState>;
        @group(0) @binding(1) var<storage, read> gameTheoryForces: array<vec3<f32>>;
        @group(0) @binding(2) var<storage, read_write> trajectories: array<vec3<f32>>;
        @group(0) @binding(3) var<uniform> params: TrajectoryParams;
        
        struct AgentState {
            position: vec3<f32>,
            velocity: vec3<f32>,
            performance: f32,
            modelType: u32,
        }
        
        struct TrajectoryParams {
            numAgents: u32,
            deltaTime: f32,
            attractionStrength: f32,
            repulsionStrength: f32,
            dampingFactor: f32,
            interstellarMode: u32,
        }
        
        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let agentIndex = global_id.x;
            if (agentIndex >= params.numAgents) {
                return;
            }
            
            let agent = agentStates[agentIndex];
            var force = gameTheoryForces[agentIndex];
            
            // Calculate forces from other agents
            for (var i = 0u; i < params.numAgents; i++) {
                if (i == agentIndex) {
                    continue;
                }
                
                let other = agentStates[i];
                let direction = agent.position - other.position;
                let distance = length(direction);
                
                if (distance > 0.0) {
                    let normalizedDir = direction / distance;
                    
                    // Attraction based on coordination potential
                    let coordinationPotential = agent.performance * other.performance;
                    let attraction = normalizedDir * coordinationPotential * params.attractionStrength / (distance * distance);
                    
                    // Repulsion to prevent clustering
                    let repulsion = normalizedDir * params.repulsionStrength / (distance * distance * distance);
                    
                    force += attraction - repulsion;
                }
            }
            
            // Interstellar mode: Add gravitational wave effects
            if (params.interstellarMode == 1u) {
                let wavePhase = length(agent.position) * 0.1 + params.deltaTime * 10.0;
                let waveAmplitude = 0.5 * agent.performance;
                let waveForce = vec3<f32>(
                    sin(wavePhase) * waveAmplitude,
                    cos(wavePhase * 1.3) * waveAmplitude,
                    sin(wavePhase * 0.7) * waveAmplitude * 0.5
                );
                force += waveForce;
            }
            
            // Update velocity with damping
            let newVelocity = agent.velocity * params.dampingFactor + force * params.deltaTime;
            
            // Update position
            let newPosition = agent.position + newVelocity * params.deltaTime;
            
            // Store new trajectory point
            trajectories[agentIndex] = newPosition;
        }
    """

    /// WebGPU Compute Shader for Equilibrium Analysis
    let equilibriumAnalysisShader = """
        @group(0) @binding(0) var<storage, read> agentDecisions: array<Decision>;
        @group(0) @binding(1) var<storage, read> payoffMatrix: array<f32>;
        @group(0) @binding(2) var<storage, read_write> equilibriumMetrics: array<f32>;
        @group(0) @binding(3) var<uniform> params: EquilibriumParams;
        
        struct Decision {
            agentId: u32,
            action: u32,
            expectedReward: f32,
            actualReward: f32,
            regret: f32,
            timestamp: f32,
        }
        
        struct EquilibriumParams {
            numAgents: u32,
            numActions: u32,
            convergenceThreshold: f32,
            analysisWindow: u32,
            gameTheoryModel: u32,
        }
        
        @compute @workgroup_size(32, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let agentIndex = global_id.x;
            if (agentIndex >= params.numAgents) {
                return;
            }
            
            var totalRegret = 0.0;
            var convergenceScore = 0.0;
            var stabilityScore = 0.0;
            
            // Analyze recent decisions for this agent
            let startIndex = agentIndex * params.analysisWindow;
            for (var i = 0u; i < params.analysisWindow; i++) {
                let decisionIndex = startIndex + i;
                if (decisionIndex < arrayLength(&agentDecisions)) {
                    let decision = agentDecisions[decisionIndex];
                    totalRegret += abs(decision.regret);
                    
                    // Calculate best response deviation
                    var bestReward = decision.actualReward;
                    for (var action = 0u; action < params.numActions; action++) {
                        let payoffIndex = agentIndex * params.numActions + action;
                        if (payoffIndex < arrayLength(&payoffMatrix)) {
                            bestReward = max(bestReward, payoffMatrix[payoffIndex]);
                        }
                    }
                    
                    let deviation = abs(bestReward - decision.actualReward);
                    convergenceScore += 1.0 / (1.0 + deviation);
                }
            }
            
            // Calculate stability based on regret variance
            let avgRegret = totalRegret / f32(params.analysisWindow);
            var regretVariance = 0.0;
            
            for (var i = 0u; i < params.analysisWindow; i++) {
                let decisionIndex = startIndex + i;
                if (decisionIndex < arrayLength(&agentDecisions)) {
                    let decision = agentDecisions[decisionIndex];
                    let diff = abs(decision.regret) - avgRegret;
                    regretVariance += diff * diff;
                }
            }
            
            stabilityScore = 1.0 / (1.0 + regretVariance / f32(params.analysisWindow));
            
            // Store metrics
            let metricsIndex = agentIndex * 4u;
            equilibriumMetrics[metricsIndex] = avgRegret;
            equilibriumMetrics[metricsIndex + 1u] = convergenceScore / f32(params.analysisWindow);
            equilibriumMetrics[metricsIndex + 2u] = stabilityScore;
            equilibriumMetrics[metricsIndex + 3u] = select(0.0, 1.0, avgRegret < params.convergenceThreshold);
        }
    """

    /// WebGPU Particle System for Coordination Visualization
    let coordinationParticleShader = """
        @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
        @group(0) @binding(1) var<storage, read> agentPositions: array<vec3<f32>>;
        @group(0) @binding(2) var<storage, read> coordinationStrengths: array<f32>;
        @group(0) @binding(3) var<uniform> params: ParticleParams;
        
        struct Particle {
            position: vec3<f32>,
            velocity: vec3<f32>,
            life: f32,
            size: f32,
            color: vec4<f32>,
        }
        
        struct ParticleParams {
            numParticles: u32,
            numAgents: u32,
            deltaTime: f32,
            emissionRate: f32,
            particleLife: f32,
            interstellarMode: u32,
        }
        
        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let particleIndex = global_id.x;
            if (particleIndex >= params.numParticles) {
                return;
            }
            
            var particle = particles[particleIndex];
            
            // Update particle life
            particle.life -= params.deltaTime;
            
            // Respawn particle if dead
            if (particle.life <= 0.0) {
                // Find strongest coordination connection
                var maxStrength = 0.0;
                var sourceAgent = 0u;
                var targetAgent = 1u;
                
                for (var i = 0u; i < params.numAgents; i++) {
                    for (var j = i + 1u; j < params.numAgents; j++) {
                        let strengthIndex = i * params.numAgents + j;
                        if (strengthIndex < arrayLength(&coordinationStrengths)) {
                            let strength = coordinationStrengths[strengthIndex];
                            if (strength > maxStrength) {
                                maxStrength = strength;
                                sourceAgent = i;
                                targetAgent = j;
                            }
                        }
                    }
                }
                
                // Spawn particle at source agent
                if (sourceAgent < params.numAgents && targetAgent < params.numAgents) {
                    particle.position = agentPositions[sourceAgent];
                    let direction = normalize(agentPositions[targetAgent] - agentPositions[sourceAgent]);
                    particle.velocity = direction * maxStrength * 2.0;
                    particle.life = params.particleLife;
                    particle.size = 0.1 + maxStrength * 0.2;
                    
                    // Color based on coordination strength
                    if (params.interstellarMode == 1u) {
                        particle.color = vec4<f32>(
                            0.3 + maxStrength * 0.7,
                            0.6 + maxStrength * 0.4,
                            1.0,
                            maxStrength
                        );
                    } else {
                        particle.color = vec4<f32>(
                            maxStrength,
                            1.0 - maxStrength,
                            0.5,
                            maxStrength
                        );
                    }
                }
            } else {
                // Update existing particle
                particle.position += particle.velocity * params.deltaTime;
                
                // Fade out over time
                let lifeFactor = particle.life / params.particleLife;
                particle.color.w = lifeFactor;
                particle.size *= 0.995; // Slight shrinking
                
                // Interstellar mode: Add swirling motion
                if (params.interstellarMode == 1u) {
                    let swirl = vec3<f32>(
                        sin(particle.life * 5.0) * 0.1,
                        cos(particle.life * 4.0) * 0.1,
                        sin(particle.life * 3.0) * 0.05
                    );
                    particle.velocity += swirl;
                }
            }
            
            particles[particleIndex] = particle;
        }
    """

    /// WebGPU Integration Manager
    type WebGPUManager() =
        
        /// Generate WebGPU initialization JavaScript
        member this.GenerateWebGPUInit() : string =
            """
                // TARS Game Theory WebGPU Initialization
                async function initTarsWebGPU() {
                    if (!navigator.gpu) {
                        console.warn('WebGPU not supported, falling back to WebGL');
                        return null;
                    }
                    
                    try {
                        const adapter = await navigator.gpu.requestAdapter();
                        const device = await adapter.requestDevice();
                        
                        // Store WebGPU context globally
                        window.tarsWebGPU = {
                            device: device,
                            adapter: adapter,
                            computePipelines: new Map(),
                            buffers: new Map(),
                            bindGroups: new Map()
                        };
                        
                        console.log('🚀 TARS WebGPU initialized successfully');
                        return device;
                    } catch (error) {
                        console.error('Failed to initialize WebGPU:', error);
                        return null;
                    }
                }
                
                // Initialize WebGPU on load
                initTarsWebGPU();
            """
        
        /// Generate compute pipeline creation JavaScript
        member this.GenerateComputePipeline(name: string, shaderCode: string) : string =
            sprintf """
                // Create %s compute pipeline
                if (window.tarsWebGPU && window.tarsWebGPU.device) {
                    const device = window.tarsWebGPU.device;
                    
                    const shaderModule = device.createShaderModule({
                        code: `%s`
                    });
                    
                    const computePipeline = device.createComputePipeline({
                        layout: 'auto',
                        compute: {
                            module: shaderModule,
                            entryPoint: 'main'
                        }
                    });
                    
                    window.tarsWebGPU.computePipelines.set('%s', computePipeline);
                    console.log('⚡ Created %s compute pipeline');
                }
            """ name shaderCode name name
        
        /// Generate buffer creation JavaScript
        member this.GenerateBufferCreation(name: string, size: int, usage: string) : string =
            sprintf """
                // Create %s buffer
                if (window.tarsWebGPU && window.tarsWebGPU.device) {
                    const device = window.tarsWebGPU.device;
                    
                    const buffer = device.createBuffer({
                        size: %d,
                        usage: GPUBufferUsage.%s
                    });
                    
                    window.tarsWebGPU.buffers.set('%s', buffer);
                    console.log('📦 Created %s buffer (%d bytes)');
                }
            """ name size usage name name size
        
        /// Generate compute dispatch JavaScript
        member this.GenerateComputeDispatch(pipelineName: string, workgroupsX: int, workgroupsY: int, workgroupsZ: int) : string =
            sprintf """
                // Dispatch %s compute shader
                if (window.tarsWebGPU && window.tarsWebGPU.device) {
                    const device = window.tarsWebGPU.device;
                    const pipeline = window.tarsWebGPU.computePipelines.get('%s');
                    
                    if (pipeline) {
                        const commandEncoder = device.createCommandEncoder();
                        const passEncoder = commandEncoder.beginComputePass();
                        
                        passEncoder.setPipeline(pipeline);
                        // Bind groups would be set here based on the specific pipeline
                        passEncoder.dispatchWorkgroups(%d, %d, %d);
                        passEncoder.end();
                        
                        device.queue.submit([commandEncoder.finish()]);
                    }
                }
            """ pipelineName pipelineName workgroupsX workgroupsY workgroupsZ
