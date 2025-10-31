import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import SolarSystem from './components/SolarSystem';
import UI from './components/UI';
import './App.css';

function App() {
  return (
    <div className="App">
      <Canvas
        camera={{ position: [0, 0, 50], fov: 75 }}
        style={{ background: 'linear-gradient(to bottom, #000428, #004e92)' }}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={0.1} />
          <pointLight position={[0, 0, 0]} intensity={2} />
          <Stars radius={300} depth={60} count={20000} factor={7} />
          <SolarSystem />
          <OrbitControls 
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            zoomSpeed={0.6}
            panSpeed={0.8}
            rotateSpeed={0.4}
          />
        </Suspense>
      </Canvas>
      <UI />
    </div>
  );
}

export default App;