import { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import HelloWorldScene from './components/HelloWorldScene'

function App() {
  const [isWebGPUSupported, setIsWebGPUSupported] = useState<boolean | null>(null)

  useEffect(() => {
    const checkWebGPUSupport = async () => {
      try {
        if (!navigator.gpu) {
          setIsWebGPUSupported(false)
          return
        }

        // Try to get an adapter to confirm WebGPU is available
        const adapter = await navigator.gpu.requestAdapter()
        setIsWebGPUSupported(!!adapter)
      } catch (error) {
        console.error('Error checking WebGPU support:', error)
        setIsWebGPUSupported(false)
      }
    }

    checkWebGPUSupport()
  }, [])

  if (isWebGPUSupported === null) {
    return <div className="webgpu-not-supported">Checking WebGPU support...</div>
  }

  if (isWebGPUSupported === false) {
    return (
      <div className="webgpu-not-supported">
        <div>
          <h2>WebGPU is not supported in your browser</h2>
          <p>Please use a browser that supports WebGPU, such as Chrome 113+ or Edge 113+</p>
        </div>
      </div>
    )
  }

  return (
    <>
      <Canvas gl={{
        powerPreference: 'high-performance',
        antialias: true,
        // Force WebGPU rendering
        // @ts-ignore - TypeScript doesn't know about the WebGPU properties yet
        webgpu: true,
        // Disable WebGL fallback
        // @ts-ignore
        failIfMajorPerformanceCaveat: true
      }}>
        <HelloWorldScene />
      </Canvas>
      <div className="info-overlay">
        <p><strong>TARS - React Three.js WebGPU Demo</strong></p>
        <p>✅ Using WebGPU rendering mode</p>
        <p>Browser: {navigator.userAgent.split(' ').slice(-1)[0]}</p>
      </div>
    </>
  )
}

export default App
