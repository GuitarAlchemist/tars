import React, { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';

function Planet({ 
  name, 
  size, 
  distance, 
  color, 
  orbitSpeed, 
  rotationSpeed, 
  texture,
  moons = []
}) {
  const planetRef = useRef();
  const orbitRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  useFrame((state) => {
    if (orbitRef.current) {
      orbitRef.current.rotation.y += orbitSpeed;
    }
    if (planetRef.current) {
      planetRef.current.rotation.y += rotationSpeed;
    }
  });

  return (
    <group ref={orbitRef}>
      {/* Orbit line */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[distance - 0.05, distance + 0.05, 64]} />
        <meshBasicMaterial color="#444444" transparent opacity={0.3} />
      </mesh>
      
      {/* Planet */}
      <group position={[distance, 0, 0]}>
        <mesh
          ref={planetRef}
          onPointerOver={() => setHovered(true)}
          onPointerOut={() => setHovered(false)}
          scale={hovered ? 1.2 : 1}
        >
          <sphereGeometry args={[size, 32, 32]} />
          <meshStandardMaterial 
            color={color}
            roughness={0.8}
            metalness={0.1}
          />
        </mesh>
        
        {/* Planet label */}
        {hovered && (
          <Text
            position={[0, size + 1, 0]}
            fontSize={0.5}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {name}
          </Text>
        )}
        
        {/* Moons */}
        {moons.map((moon, index) => (
          <group key={moon.name}>
            <mesh position={[moon.distance, 0, 0]}>
              <sphereGeometry args={[moon.size, 16, 16]} />
              <meshStandardMaterial color={moon.color} />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  );
}

export default Planet;