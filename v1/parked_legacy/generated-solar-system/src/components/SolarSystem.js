import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import Planet from './Planet';
import { planetData } from '../data/planetData';

function SolarSystem() {
  const systemRef = useRef();
  
  useFrame((state) => {
    if (systemRef.current) {
      systemRef.current.rotation.y += 0.001;
    }
  });

  const planets = useMemo(() => 
    planetData.map((planet, index) => (
      <Planet
        key={planet.name}
        {...planet}
        index={index}
      />
    )), []
  );

  return (
    <group ref={systemRef}>
      {/* Sun */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[2, 32, 32]} />
        <meshBasicMaterial 
          color="#FDB813"
          emissive="#FDB813"
          emissiveIntensity={0.3}
        />
      </mesh>
      
      {/* Planets */}
      {planets}
      
      {/* Asteroid Belt */}
      <group>
        {Array.from({ length: 1000 }, (_, i) => (
          <mesh
            key={i}
            position={[
              Math.cos(i * 0.1) * (15 + Math.random() * 3),
              (Math.random() - 0.5) * 0.5,
              Math.sin(i * 0.1) * (15 + Math.random() * 3)
            ]}
          >
            <sphereGeometry args={[0.02, 4, 4]} />
            <meshStandardMaterial color="#8C7853" />
          </mesh>
        ))}
      </group>
    </group>
  );
}

export default SolarSystem;