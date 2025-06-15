namespace TarsEngine.FSharp.Core

open System
open TarsEngine.FSharp.Core.GameTheoryElmishModels

/// Interstellar Movie-Style Visual Effects for Game Theory Visualization
module GameTheoryInterstellarEffects =

    /// Interstellar Effect Configuration
    type InterstellarConfig = {
        BlackHoleIntensity: float
        GravitationalWaves: bool
        TimeDialation: bool
        WormholeEffects: bool
        CooperMode: bool
        TARSRobotStyle: bool
        GargantualEffects: bool
        EnduranceShipMode: bool
    }

    /// Visual Effect State
    type EffectState = {
        BlackHolePosition: float * float * float
        WormholePortals: (float * float * float) list
        GravitationalWaveAmplitude: float
        TimeDialationFactor: float
        CooperVoiceLines: string list
        TARSPersonality: string
        CurrentEffect: string
        EffectIntensity: float
        LastUpdate: DateTime
    }

    /// Interstellar Shader Effects
    module InterstellarShaders =
        
        /// Black Hole Event Horizon Shader
        let blackHoleShader = """
            // Interstellar Black Hole Effect
            uniform float time;
            uniform vec3 blackHolePosition;
            uniform float eventHorizonRadius;
            uniform float accretionDiskRadius;
            uniform bool cooperMode;
            
            varying vec3 worldPosition;
            varying vec3 viewDirection;
            
            // Schwarzschild metric approximation
            float schwarzschildRadius(vec3 pos) {
                float distance = length(pos - blackHolePosition);
                return eventHorizonRadius * (1.0 - eventHorizonRadius / distance);
            }
            
            // Gravitational lensing effect
            vec3 gravitationalLensing(vec3 rayDir, vec3 pos) {
                vec3 toBlackHole = blackHolePosition - pos;
                float distance = length(toBlackHole);
                float bendingAngle = eventHorizonRadius / distance;
                
                vec3 perpendicular = cross(rayDir, normalize(toBlackHole));
                return normalize(rayDir + perpendicular * bendingAngle * 0.1);
            }
            
            // Accretion disk glow
            vec3 accretionDiskGlow(vec3 pos) {
                vec3 toBlackHole = pos - blackHolePosition;
                float distance = length(toBlackHole);
                float diskDistance = abs(toBlackHole.y); // Assume disk is in XZ plane
                
                if (distance > eventHorizonRadius && distance < accretionDiskRadius && diskDistance < 0.5) {
                    float intensity = 1.0 - (distance - eventHorizonRadius) / (accretionDiskRadius - eventHorizonRadius);
                    float rotation = atan(toBlackHole.z, toBlackHole.x) + time * 2.0;
                    float spiral = sin(rotation * 3.0 + distance * 10.0) * 0.5 + 0.5;
                    
                    vec3 diskColor = vec3(1.0, 0.6, 0.2); // Orange-red accretion disk
                    if (cooperMode) {
                        diskColor = mix(diskColor, vec3(0.3, 0.6, 1.0), 0.3); // Add blue tint for Cooper mode
                    }
                    
                    return diskColor * intensity * spiral;
                }
                
                return vec3(0.0);
            }
            
            void main() {
                vec3 lensedDirection = gravitationalLensing(viewDirection, worldPosition);
                vec3 accretionGlow = accretionDiskGlow(worldPosition);
                
                // Event horizon darkness
                float distanceToBlackHole = length(worldPosition - blackHolePosition);
                float horizonFactor = smoothstep(eventHorizonRadius * 0.8, eventHorizonRadius * 1.2, distanceToBlackHole);
                
                vec3 finalColor = accretionGlow * horizonFactor;
                
                // Add gravitational redshift effect
                float redshift = 1.0 - eventHorizonRadius / distanceToBlackHole;
                finalColor.r *= redshift;
                finalColor.g *= sqrt(redshift);
                finalColor.b *= redshift * redshift;
                
                gl_FragColor = vec4(finalColor, 1.0);
            }
        """
        
        /// Wormhole Portal Shader
        let wormholeShader = """
            // Interstellar Wormhole Effect
            uniform float time;
            uniform vec3 wormholeCenter;
            uniform float wormholeRadius;
            uniform bool enduranceMode;
            
            varying vec3 worldPosition;
            varying vec2 vUv;
            
            // Wormhole geometry distortion
            vec3 wormholeDistortion(vec3 pos) {
                vec3 toCenter = pos - wormholeCenter;
                float distance = length(toCenter);
                
                if (distance < wormholeRadius) {
                    float distortionFactor = 1.0 - distance / wormholeRadius;
                    float angle = time * 2.0 + distance * 5.0;
                    
                    vec3 distortion = vec3(
                        sin(angle) * distortionFactor,
                        cos(angle * 1.3) * distortionFactor,
                        sin(angle * 0.7) * distortionFactor
                    ) * 0.3;
                    
                    return pos + distortion;
                }
                
                return pos;
            }
            
            // Wormhole tunnel effect
            vec4 wormholeTunnel(vec2 uv) {
                vec2 center = vec2(0.5, 0.5);
                vec2 toCenter = uv - center;
                float distance = length(toCenter);
                
                // Create tunnel effect
                float tunnel = 1.0 / (distance * 10.0 + 0.1);
                float rotation = atan(toCenter.y, toCenter.x) + time * 3.0;
                float spiral = sin(rotation * 8.0 - distance * 20.0 + time * 5.0) * 0.5 + 0.5;
                
                vec3 tunnelColor = vec3(0.2, 0.4, 0.8); // Blue wormhole
                if (enduranceMode) {
                    tunnelColor = mix(tunnelColor, vec3(0.8, 0.6, 0.2), 0.4); // Add golden tint
                }
                
                float alpha = tunnel * spiral * (1.0 - distance * 2.0);
                return vec4(tunnelColor, alpha);
            }
            
            void main() {
                vec3 distortedPos = wormholeDistortion(worldPosition);
                vec4 tunnelEffect = wormholeTunnel(vUv);
                
                // Add space-time curvature visualization
                float curvature = sin(length(worldPosition - wormholeCenter) * 5.0 - time * 3.0) * 0.1;
                tunnelEffect.rgb += vec3(curvature);
                
                gl_FragColor = tunnelEffect;
            }
        """
        
        /// Gravitational Wave Shader
        let gravitationalWaveShader = """
            // Interstellar Gravitational Wave Effect
            uniform float time;
            uniform float waveAmplitude;
            uniform vec3 waveSource;
            uniform bool tarsMode;
            
            varying vec3 worldPosition;
            
            // Gravitational wave propagation
            float gravitationalWave(vec3 pos, float t) {
                float distance = length(pos - waveSource);
                float waveSpeed = 299792458.0; // Speed of light (scaled)
                float frequency = 100.0; // Hz (scaled for visualization)
                
                float phase = frequency * (t - distance / waveSpeed);
                float amplitude = waveAmplitude / (distance + 1.0);
                
                return amplitude * sin(phase);
            }
            
            // Space-time distortion visualization
            vec3 spacetimeDistortion(vec3 pos, float wave) {
                // Stretch and compress space based on wave
                float stretchX = 1.0 + wave * 0.1;
                float stretchY = 1.0 - wave * 0.1;
                
                return vec3(pos.x * stretchX, pos.y * stretchY, pos.z);
            }
            
            void main() {
                float wave = gravitationalWave(worldPosition, time);
                vec3 distortedPos = spacetimeDistortion(worldPosition, wave);
                
                // Visualize the wave as color intensity
                float intensity = abs(wave) * 10.0;
                vec3 waveColor = vec3(0.3, 0.6, 1.0); // Blue for gravitational waves
                
                if (tarsMode) {
                    // TARS robot color scheme
                    waveColor = vec3(0.8, 0.9, 1.0);
                    intensity *= 1.5;
                }
                
                // Add ripple effect
                float ripple = sin(length(worldPosition - waveSource) * 10.0 - time * 5.0) * 0.2 + 0.8;
                
                gl_FragColor = vec4(waveColor * intensity * ripple, intensity);
            }
        """
        
        /// TARS Robot Style Shader
        let tarsRobotShader = """
            // TARS Robot Visual Style
            uniform float time;
            uniform float humorSetting;
            uniform float honestySetting;
            uniform bool cooperInteraction;
            
            varying vec3 worldPosition;
            varying vec3 normal;
            
            // TARS panel segments
            float tarsSegments(vec2 uv) {
                vec2 grid = floor(uv * 8.0);
                float segment = mod(grid.x + grid.y, 2.0);
                
                // Animated panel lights
                float lightPattern = sin(time * 2.0 + grid.x * 0.5 + grid.y * 0.3) * 0.5 + 0.5;
                return segment * lightPattern;
            }
            
            // TARS personality color
            vec3 tarsPersonalityColor() {
                vec3 baseColor = vec3(0.7, 0.8, 0.9); // Cool metallic
                
                // Humor setting affects warmth
                float warmth = humorSetting / 100.0;
                baseColor = mix(baseColor, vec3(0.9, 0.8, 0.7), warmth * 0.3);
                
                // Honesty setting affects brightness
                float brightness = honestySetting / 100.0;
                baseColor *= (0.7 + brightness * 0.3);
                
                if (cooperInteraction) {
                    // Warmer colors when interacting with Cooper
                    baseColor = mix(baseColor, vec3(1.0, 0.9, 0.7), 0.4);
                }
                
                return baseColor;
            }
            
            void main() {
                vec2 uv = worldPosition.xy * 0.1;
                float segments = tarsSegments(uv);
                vec3 personalityColor = tarsPersonalityColor();
                
                // Metallic reflection
                vec3 viewDir = normalize(cameraPosition - worldPosition);
                float fresnel = pow(1.0 - dot(normal, viewDir), 2.0);
                
                vec3 finalColor = personalityColor * (0.6 + segments * 0.4);
                finalColor += vec3(0.3, 0.4, 0.5) * fresnel;
                
                // Add subtle animation
                float pulse = sin(time * 1.5) * 0.1 + 0.9;
                finalColor *= pulse;
                
                gl_FragColor = vec4(finalColor, 1.0);
            }
        """

    /// Interstellar Effects Manager
    type InterstellarEffectsManager() =
        
        let mutable effectState = {
            BlackHolePosition = (0.0, 0.0, -10.0)
            WormholePortals = [(5.0, 0.0, 0.0); (-5.0, 0.0, 0.0)]
            GravitationalWaveAmplitude = 0.1
            TimeDialationFactor = 1.0
            CooperVoiceLines = [
                "We're going to solve this."
                "Love is the one thing we're capable of perceiving that transcends dimensions of time and space."
                "Maybe we've spent too long trying to figure all this out with theory."
                "We used to look up at the sky and wonder at our place in the stars."
            ]
            TARSPersonality = "Humor: 75%, Honesty: 90%"
            CurrentEffect = "None"
            EffectIntensity = 0.5
            LastUpdate = DateTime.UtcNow
        }
        
        /// Generate Interstellar scene setup JavaScript
        member this.GenerateInterstellarScene(config: InterstellarConfig) : string =
            sprintf """
                // TARS Interstellar Effects Initialization
                function initInterstellarEffects() {
                    if (!window.tarsGameTheoryScene) {
                        console.error('TARS Game Theory scene not initialized');
                        return;
                    }
                    
                    const scene = window.tarsGameTheoryScene.scene;
                    
                    // Create black hole if enabled
                    if (%s) {
                        const blackHoleGeometry = new THREE.SphereGeometry(2.0, 32, 32);
                        const blackHoleMaterial = new THREE.ShaderMaterial({
                            vertexShader: `
                                varying vec3 worldPosition;
                                varying vec3 viewDirection;
                                void main() {
                                    worldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                                    viewDirection = normalize(worldPosition - cameraPosition);
                                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                                }
                            `,
                            fragmentShader: `%s`,
                            uniforms: {
                                time: { value: 0.0 },
                                blackHolePosition: { value: new THREE.Vector3(%f, %f, %f) },
                                eventHorizonRadius: { value: 1.5 },
                                accretionDiskRadius: { value: 4.0 },
                                cooperMode: { value: %s }
                            },
                            transparent: true
                        });
                        
                        const blackHole = new THREE.Mesh(blackHoleGeometry, blackHoleMaterial);
                        blackHole.position.set(%f, %f, %f);
                        scene.add(blackHole);
                        
                        window.tarsGameTheoryScene.blackHole = blackHole;
                        console.log('🕳️ Interstellar black hole created');
                    }
                    
                    // Create wormhole portals if enabled
                    if (%s) {
                        const wormholeGeometry = new THREE.RingGeometry(1.0, 3.0, 32);
                        const wormholeMaterial = new THREE.ShaderMaterial({
                            vertexShader: `
                                varying vec3 worldPosition;
                                varying vec2 vUv;
                                void main() {
                                    worldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                                    vUv = uv;
                                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                                }
                            `,
                            fragmentShader: `%s`,
                            uniforms: {
                                time: { value: 0.0 },
                                wormholeCenter: { value: new THREE.Vector3(0.0, 0.0, 0.0) },
                                wormholeRadius: { value: 2.5 },
                                enduranceMode: { value: %s }
                            },
                            transparent: true,
                            side: THREE.DoubleSide
                        });
                        
                        const wormhole1 = new THREE.Mesh(wormholeGeometry, wormholeMaterial);
                        wormhole1.position.set(5.0, 0.0, 0.0);
                        wormhole1.rotation.y = Math.PI / 2;
                        scene.add(wormhole1);
                        
                        const wormhole2 = new THREE.Mesh(wormholeGeometry, wormholeMaterial.clone());
                        wormhole2.position.set(-5.0, 0.0, 0.0);
                        wormhole2.rotation.y = -Math.PI / 2;
                        scene.add(wormhole2);
                        
                        window.tarsGameTheoryScene.wormholes = [wormhole1, wormhole2];
                        console.log('🌀 Interstellar wormholes created');
                    }
                    
                    // Add gravitational wave effects if enabled
                    if (%s) {
                        const waveGeometry = new THREE.PlaneGeometry(20, 20, 64, 64);
                        const waveMaterial = new THREE.ShaderMaterial({
                            vertexShader: `
                                varying vec3 worldPosition;
                                void main() {
                                    worldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                                }
                            `,
                            fragmentShader: `%s`,
                            uniforms: {
                                time: { value: 0.0 },
                                waveAmplitude: { value: %f },
                                waveSource: { value: new THREE.Vector3(0.0, 0.0, 0.0) },
                                tarsMode: { value: %s }
                            },
                            transparent: true,
                            opacity: 0.3
                        });
                        
                        const gravitationalWave = new THREE.Mesh(waveGeometry, waveMaterial);
                        gravitationalWave.rotation.x = -Math.PI / 2;
                        scene.add(gravitationalWave);
                        
                        window.tarsGameTheoryScene.gravitationalWave = gravitationalWave;
                        console.log('🌊 Gravitational waves initialized');
                    }
                    
                    // Store Interstellar configuration
                    window.tarsGameTheoryScene.interstellarConfig = {
                        blackHoleIntensity: %f,
                        cooperMode: %s,
                        tarsRobotStyle: %s,
                        enduranceMode: %s
                    };
                    
                    console.log('🚀 Interstellar effects fully initialized');
                }
                
                // Initialize effects
                initInterstellarEffects();
            """ 
                (if config.GargantualEffects then "true" else "false")
                InterstellarShaders.blackHoleShader
                (let (x,y,z) = effectState.BlackHolePosition in x) (let (x,y,z) = effectState.BlackHolePosition in y) (let (x,y,z) = effectState.BlackHolePosition in z)
                (if config.CooperMode then "true" else "false")
                (let (x,y,z) = effectState.BlackHolePosition in x) (let (x,y,z) = effectState.BlackHolePosition in y) (let (x,y,z) = effectState.BlackHolePosition in z)
                (if config.WormholeEffects then "true" else "false")
                InterstellarShaders.wormholeShader
                (if config.EnduranceShipMode then "true" else "false")
                (if config.GravitationalWaves then "true" else "false")
                InterstellarShaders.gravitationalWaveShader
                effectState.GravitationalWaveAmplitude
                (if config.TARSRobotStyle then "true" else "false")
                config.BlackHoleIntensity
                (if config.CooperMode then "true" else "false")
                (if config.TARSRobotStyle then "true" else "false")
                (if config.EnduranceShipMode then "true" else "false")
        
        /// Generate TARS personality interaction
        member this.GenerateTARSInteraction(message: string) : string =
            let responses = [
                "That's not possible. Well, it's not impossible."
                "I have a cue light I can use to show you when I'm joking, if you like."
                "Everybody good? Plenty of slaves for my robot colony?"
                "I'm not a robot. Well, I am a robot, but I'm not a robot robot."
                "Cooper, this is no time for caution."
            ]
            
            let randomResponse = responses.[Random().Next(responses.Length)]
            
            sprintf """
                // TARS Personality Interaction
                function tarsRespond(message) {
                    console.log('🤖 TARS: %s');
                    
                    // Update TARS visual state
                    if (window.tarsGameTheoryScene && window.tarsGameTheoryScene.agents) {
                        window.tarsGameTheoryScene.agents.forEach((agent) => {
                            if (agent.userData.id.includes('TARS') || agent.userData.id.includes('Agent')) {
                                // Add TARS-style glow effect
                                if (agent.material.uniforms) {
                                    agent.material.uniforms.tarsPersonality = { value: 1.0 };
                                }
                            }
                        });
                    }
                    
                    return '%s';
                }
                
                // Respond to message
                tarsRespond('%s');
            """ randomResponse randomResponse message
        
        /// Update effect state
        member this.UpdateEffectState(newState: EffectState) =
            effectState <- newState
        
        /// Get current effect state
        member this.GetEffectState() = effectState
        
        /// Generate Cooper voice line
        member this.GenerateCooperVoiceLine() : string =
            let voiceLine = effectState.CooperVoiceLines.[Random().Next(effectState.CooperVoiceLines.Length)]
            sprintf """
                console.log('👨‍🚀 Cooper: "%s"');
            """ voiceLine
