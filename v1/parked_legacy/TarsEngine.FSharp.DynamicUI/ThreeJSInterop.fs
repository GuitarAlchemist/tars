namespace TarsEngine.FSharp.DynamicUI

open Fable.Core
open Fable.Core.JsInterop
open Browser.Dom
open Types

module ThreeJSInterop =
    
    /// Three.js WebGPU renderer interop
    [<Import("*", "three")>]
    let THREE: obj = jsNative
    
    [<Import("WebGPURenderer", "three/examples/jsm/renderers/webgpu/WebGPURenderer.js")>]
    let WebGPURenderer: obj = jsNative
    
    [<Import("OrbitControls", "three/examples/jsm/controls/OrbitControls.js")>]
    let OrbitControls: obj = jsNative
    
    /// Initialize Three.js WebGPU scene
    let initThreeJSScene (containerId: string) =
        promise {
            try
                let container = document.getElementById(containerId)
                
                // Create scene
                let scene = THREE?Scene$()
                scene?background <- THREE?Color$(0x000510)
                
                // Create camera
                let camera = THREE?PerspectiveCamera$(75.0, window.innerWidth / window.innerHeight, 0.1, 1000.0)
                camera?position?set(8.0, 5.0, 8.0)
                
                // Create WebGPU renderer
                let renderer = WebGPURenderer$({ antialias = true })
                renderer?setSize(window.innerWidth, window.innerHeight)
                renderer?setPixelRatio(window.devicePixelRatio)
                
                // Initialize WebGPU
                do! renderer?init()
                
                container.appendChild(renderer?domElement) |> ignore
                
                // Add controls
                let controls = OrbitControls$(camera, renderer?domElement)
                controls?enableDamping <- true
                controls?dampingFactor <- 0.05
                
                // Add lighting
                let ambientLight = THREE?AmbientLight$(0x404040, 0.4)
                scene?add(ambientLight)
                
                let pointLight = THREE?PointLight$(0x00ff88, 1.0, 100.0)
                pointLight?position?set(0.0, 5.0, 5.0)
                scene?add(pointLight)
                
                return Some { Scene = scene; Camera = camera; Renderer = renderer; Controls = controls }
            with
            | ex ->
                console.error("Failed to initialize Three.js WebGPU:", ex)
                return None
        }
    
    /// Add agent node to 3D scene
    let addAgentNode (scene: obj) (agent: TarsAgent) =
        let (x, y, z) = agent.Position
        
        // Create sphere geometry
        let geometry = THREE?SphereGeometry$(0.3, 16, 16)
        let material = THREE?MeshBasicMaterial$({
            color = if agent.Status = Running then 0x00ff88 else 0x333333
            transparent = true
            opacity = if agent.Status = Running then 1.0 else 0.5
        })
        
        let sphere = THREE?Mesh$(geometry, material)
        sphere?position?set(x, y, z)
        sphere?userData <- agent
        
        // Add glow effect for running agents
        if agent.Status = Running then
            let glowGeometry = THREE?SphereGeometry$(0.5, 16, 16)
            let glowMaterial = THREE?MeshBasicMaterial$({
                color = 0x00ff88
                transparent = true
                opacity = 0.2
            })
            let glow = THREE?Mesh$(glowGeometry, glowMaterial)
            glow?position?copy(sphere?position)
            scene?add(glow)
        
        scene?add(sphere)
        sphere
    
    /// Add connection line between agents
    let addConnectionLine (scene: obj) (fromPos: float * float * float) (toPos: float * float * float) (isActive: bool) =
        let (x1, y1, z1) = fromPos
        let (x2, y2, z2) = toPos
        
        let points = [|
            THREE?Vector3$(x1, y1, z1)
            THREE?Vector3$(x2, y2, z2)
        |]
        
        let geometry = THREE?BufferGeometry$()
        geometry?setFromPoints(points)
        
        let material = THREE?LineBasicMaterial$({
            color = if isActive then 0x00ff88 else 0x333333
            transparent = true
            opacity = if isActive then 0.6 else 0.2
        })
        
        let line = THREE?Line$(geometry, material)
        scene?add(line)
        line
    
    /// Start animation loop
    let startAnimationLoop (scene: obj) (camera: obj) (renderer: obj) (controls: obj) (onFrame: unit -> unit) =
        let rec animate () =
            window.requestAnimationFrame(fun _ -> animate()) |> ignore
            
            // Update controls
            controls?update()
            
            // Custom frame logic
            onFrame()
            
            // Render scene
            renderer?render(scene, camera)
        
        animate()
    
    /// Update agent node status
    let updateAgentNode (node: obj) (agent: TarsAgent) =
        let color = if agent.Status = Running then 0x00ff88 else 0x333333
        let opacity = if agent.Status = Running then 1.0 else 0.5
        
        node?material?color?setHex(color)
        node?material?opacity <- opacity
        node?userData <- agent
    
    type ThreeJSScene = {
        Scene: obj
        Camera: obj
        Renderer: obj
        Controls: obj
    }
