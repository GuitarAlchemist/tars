import { useRef, useEffect, useState } from 'react'
import { useFrame, useThree, useLoader } from '@react-three/fiber'
import { Text, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'

// Constants for the scene
const SUNLIGHT_COLOR = 0xFDB813 // Golden yellow color similar to the sun
const ANODIZED_BLUE_COLOR = 0x0077CC // Anodized blue color
const YELLOW_TEXT_COLOR = 0xFFFF00 // Bright yellow for text

// Skybox component with parallax effect (follows camera rotation but not zoom)
const MilkyWaySkybox = () => {
  const { scene } = useThree()
  const [texture] = useState(() => new THREE.TextureLoader().load('./textures/milky_way.jpg'))

  useEffect(() => {
    // Create a skybox using CubeTextureLoader or by using a single texture on a large sphere
    const skyGeometry = new THREE.SphereGeometry(1000, 60, 40) // Very large radius
    const skyMaterial = new THREE.MeshBasicMaterial({
      map: texture,
      side: THREE.BackSide, // Render on the inside of the sphere
      depthWrite: false,    // Don't write to depth buffer
    })

    const sky = new THREE.Mesh(skyGeometry, skyMaterial)
    sky.name = 'milkyWaySkybox'

    // Add to scene
    scene.add(sky)

    // Cleanup function
    return () => {
      scene.remove(sky)
      skyGeometry.dispose()
      skyMaterial.dispose()
      texture.dispose()
    }
  }, [scene, texture])

  // No need to return anything as we're adding directly to the scene
  return null
}

const HelloWorldScene = () => {
  const { gl } = useThree()
  const textRef = useRef<THREE.Mesh>(null)
  const textMaterialRef = useRef<THREE.MeshStandardMaterial>(null)

  // Set up WebGPU renderer
  useEffect(() => {
    // Check if we're using WebGPU
    if ('webgpu' in gl) {
      try {
        // @ts-ignore - TypeScript doesn't know about the webgpu property yet
        gl.outputEncoding = THREE.sRGBEncoding
        console.log('✅ WebGPU renderer is active')

        // Add additional WebGPU-specific configurations here
        // @ts-ignore
        if (gl.webgpuRenderer) {
          console.log('WebGPU renderer details:', {
            // @ts-ignore
            device: gl.webgpuRenderer.device,
            // @ts-ignore
            format: gl.webgpuRenderer.format
          })
        }
      } catch (error) {
        console.error('Error configuring WebGPU:', error)
        alert('Error configuring WebGPU. Please check console for details.')
      }
    } else {
      console.error('❌ WebGPU is not available in this renderer')
      alert('This application requires WebGPU, but it appears to be using WebGL instead. Please use a browser with WebGPU support.')
    }
  }, [gl])

  // Animation loop
  useFrame((state, delta) => {
    if (textRef.current) {
      textRef.current.rotation.y += delta * 0.2
    }
  })

  return (
    <>
      {/* Camera setup */}
      <PerspectiveCamera makeDefault position={[0, 2, 5]} />
      <OrbitControls enableDamping dampingFactor={0.05} enableZoom={true} enableRotate={true} />

      {/* Milky Way background with parallax effect */}
      <MilkyWaySkybox />

      {/* Scene lighting */}
      <ambientLight intensity={0.7} />
      <directionalLight
        position={[5, 10, 5]}
        intensity={2.5}
        color={SUNLIGHT_COLOR}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[-5, 5, -5]} intensity={1.5} color={0x0088ff} />
      <pointLight position={[0, 3, 3]} intensity={1.0} color={0xffffff} />
      <hemisphereLight args={[0x0088ff, 0xfdb813, 0.5]} />

      {/* Hello World 3D Text */}
      <Text
        ref={textRef}
        fontSize={1}
        position={[0, 0, 0]}
        castShadow
        receiveShadow
        characters="abcdefghijklmnopqrstuvwxyz0123456789!"
      >
        Hello World
        <meshStandardMaterial
          ref={textMaterialRef}
          color={ANODIZED_BLUE_COLOR}
          metalness={0.8}
          roughness={0.2}
          envMapIntensity={1}
        />
      </Text>

      {/* Background plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#303030" roughness={0.8} metalness={0.2} />
      </mesh>

      {/* Additional visual elements */}
      <mesh position={[-3, 0, 0]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color={ANODIZED_BLUE_COLOR} metalness={0.8} roughness={0.2} />
      </mesh>

      <group position={[3, 0, 0]}>
        <mesh>
          <sphereGeometry args={[1.2, 32, 32]} />
          <meshStandardMaterial color={SUNLIGHT_COLOR} metalness={0.6} roughness={0.3} />
        </mesh>

        {/* Text on the sphere */}
        <Text
          position={[0, 0, 1.3]}
          fontSize={0.3}
          maxWidth={2}
          lineHeight={1}
          letterSpacing={0.02}
          textAlign="center"
          font={undefined}
        >
          Hello World
          <meshBasicMaterial color={YELLOW_TEXT_COLOR} toneMapped={false} />
        </Text>
      </group>
    </>
  )
}

export default HelloWorldScene
