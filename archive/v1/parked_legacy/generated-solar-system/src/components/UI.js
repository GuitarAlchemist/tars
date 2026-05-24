import React, { useState } from 'react';
import styled from 'styled-components';

const UIContainer = styled.div`
  position: absolute;
  top: 20px;
  left: 20px;
  color: white;
  font-family: 'Arial', sans-serif;
  z-index: 100;
  background: rgba(0, 0, 0, 0.7);
  padding: 20px;
  border-radius: 10px;
  backdrop-filter: blur(10px);
`;

const Title = styled.h1`
  margin: 0 0 20px 0;
  font-size: 24px;
  color: #FDB813;
`;

const Controls = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const Button = styled.button`
  background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
  border: none;
  color: white;
  padding: 10px 20px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 14px;
  transition: transform 0.2s;
  
  &:hover {
    transform: scale(1.05);
  }
`;

const Info = styled.div`
  margin-top: 20px;
  font-size: 12px;
  opacity: 0.8;
`;

function UI() {
  const [showInfo, setShowInfo] = useState(true);
  
  return (
    <UIContainer>
      <Title>🌌 3D Solar System</Title>
      <Controls>
        <Button onClick={() => setShowInfo(!showInfo)}>
          {showInfo ? 'Hide Info' : 'Show Info'}
        </Button>
        <Button onClick={() => window.location.reload()}>
          Reset View
        </Button>
      </Controls>
      
      {showInfo && (
        <Info>
          <p><strong>Controls:</strong></p>
          <p>• Mouse: Rotate view</p>
          <p>• Scroll: Zoom in/out</p>
          <p>• Drag: Pan around</p>
          <p>• Hover: Planet info</p>
          <br />
          <p><strong>Features:</strong></p>
          <p>• Real-time 3D rendering</p>
          <p>• Accurate orbital mechanics</p>
          <p>• Interactive planet exploration</p>
          <p>• WebGPU optimized performance</p>
        </Info>
      )}
    </UIContainer>
  );
}

export default UI;