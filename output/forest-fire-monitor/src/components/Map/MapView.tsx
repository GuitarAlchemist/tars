import React, { useEffect, useRef, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { MapComponentProps } from '../../types/FireTypes';

// ArcGIS imports - these would be the actual imports in a real implementation
// For demo purposes, we'll create a mock implementation
interface MockMapView {
  container: HTMLDivElement;
  center: [number, number];
  zoom: number;
  when: (callback: () => void) => void;
  destroy: () => void;
  ui: {
    add: (widget: any, position: string) => void;
  };
}

interface MockGraphic {
  geometry: any;
  symbol: any;
  attributes: any;
  popupTemplate: any;
}

interface MockGraphicsLayer {
  title: string;
  id: string;
  removeAll: () => void;
  add: (graphic: MockGraphic) => void;
}

// Mock ArcGIS classes for demonstration
class MockMap {
  basemap: string;
  layers: MockGraphicsLayer[] = [];

  constructor(options: { basemap: string }) {
    this.basemap = options.basemap;
  }

  add(layer: MockGraphicsLayer) {
    this.layers.push(layer);
  }
}

class MockMapViewClass implements MockMapView {
  container: HTMLDivElement;
  map: MockMap;
  center: [number, number];
  zoom: number;
  ui = {
    add: (widget: any, position: string) => {
      console.log(`Added widget to ${position}`);
    }
  };

  constructor(options: { container: HTMLDivElement; map: MockMap; center: [number, number]; zoom: number }) {
    this.container = options.container;
    this.map = options.map;
    this.center = options.center;
    this.zoom = options.zoom;
  }

  when(callback: () => void) {
    setTimeout(callback, 1000); // Simulate loading time
  }

  destroy() {
    console.log('Map view destroyed');
  }
}

class MockGraphicsLayerClass implements MockGraphicsLayer {
  title: string;
  id: string;
  graphics: MockGraphic[] = [];

  constructor(options: { title: string; id: string }) {
    this.title = options.title;
    this.id = options.id;
  }

  removeAll() {
    this.graphics = [];
  }

  add(graphic: MockGraphic) {
    this.graphics.push(graphic);
  }
}

const MapView: React.FC<MapComponentProps> = ({
  fires,
  onFireSelect,
  visualizationMode,
  filterOptions
}) => {
  const mapDiv = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<MockMapView | null>(null);
  const [loading, setLoading] = useState(true);
  const [fireLayer, setFireLayer] = useState<MockGraphicsLayer | null>(null);

  useEffect(() => {
    if (mapDiv.current) {
      console.log('🗺️ Initializing ArcGIS Map...');
      
      // Create the map with hybrid basemap for satellite imagery
      const map = new MockMap({
        basemap: 'hybrid'
      });

      // Create graphics layer for fire data
      const fireGraphicsLayer = new MockGraphicsLayerClass({
        title: 'Active Forest Fires',
        id: 'fire-layer'
      });

      map.add(fireGraphicsLayer);
      setFireLayer(fireGraphicsLayer);

      // Create the map view centered on North America
      const mapView = new MockMapViewClass({
        container: mapDiv.current,
        map: map,
        center: [-106.3468, 56.1304], // Center on North America
        zoom: 4
      });

      // Add mock widgets
      const widgets = [
        { name: 'Zoom', position: 'top-left' },
        { name: 'Home', position: 'top-left' },
        { name: 'Compass', position: 'top-left' },
        { name: 'Search', position: 'top-right' },
        { name: 'ScaleBar', position: 'bottom-left' },
        { name: 'Legend', position: 'bottom-right' },
        { name: 'Measurement', position: 'top-right' },
        { name: 'LayerList', position: 'top-right' }
      ];

      widgets.forEach(widget => {
        mapView.ui.add({ name: widget.name }, widget.position);
      });

      mapView.when(() => {
        setView(mapView);
        setLoading(false);
        console.log('✅ ArcGIS Map initialized successfully');
      });

      return () => {
        if (mapView) {
          mapView.destroy();
        }
      };
    }
  }, []);

  // Update fire graphics when fires data changes
  useEffect(() => {
    if (fireLayer && fires.length > 0) {
      console.log(`🔥 Updating map with ${fires.length} fires`);
      
      // Clear existing graphics
      fireLayer.removeAll();

      // Add fire graphics
      fires.forEach(fire => {
        const fireGraphic: MockGraphic = {
          geometry: {
            type: 'point',
            longitude: fire.longitude,
            latitude: fire.latitude
          },
          symbol: {
            type: 'simple-marker',
            style: 'circle',
            color: getFireColor(fire.intensity),
            size: getFireSize(fire.frp, visualizationMode.symbolSize),
            outline: {
              color: [255, 255, 255, 0.8],
              width: 1
            }
          },
          attributes: {
            ...fire,
            title: `Fire ${fire.id}`,
            description: `Intensity: ${fire.intensity}, FRP: ${fire.frp.toFixed(1)} MW`
          },
          popupTemplate: {
            title: 'Active Forest Fire 🔥',
            content: createFirePopupContent(fire)
          }
        };

        fireLayer.add(fireGraphic);
      });

      console.log(`✅ Added ${fires.length} fire graphics to map`);
    }
  }, [fires, fireLayer, visualizationMode]);

  const getFireColor = (intensity: string): [number, number, number, number] => {
    switch (intensity) {
      case 'extreme':
        return [139, 0, 0, 0.9]; // Dark red
      case 'high':
        return [255, 0, 0, 0.8]; // Red
      case 'medium':
        return [255, 165, 0, 0.8]; // Orange
      case 'low':
        return [255, 255, 0, 0.8]; // Yellow
      default:
        return [255, 165, 0, 0.8];
    }
  };

  const getFireSize = (frp: number, symbolSize: string): number => {
    if (symbolSize === 'fixed') {
      return 10;
    }
    
    // Proportional sizing based on Fire Radiative Power
    if (frp > 100) return 20;
    if (frp > 50) return 16;
    if (frp > 25) return 12;
    if (frp > 10) return 8;
    return 6;
  };

  const createFirePopupContent = (fire: any): string => {
    return `
      <div style="padding: 15px; font-family: 'Roboto', sans-serif; max-width: 300px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
          <span style="font-size: 24px; margin-right: 8px;">🔥</span>
          <h3 style="margin: 0; color: #ff6b35;">Active Forest Fire</h3>
        </div>
        
        <div style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
          <p style="margin: 5px 0;"><strong>📍 Location:</strong> ${fire.location}</p>
          <p style="margin: 5px 0;"><strong>🌡️ Intensity:</strong> 
            <span style="color: ${getIntensityColor(fire.intensity)}; font-weight: bold;">
              ${fire.intensity.toUpperCase()}
            </span>
          </p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
          <div>
            <p style="margin: 2px 0; font-size: 12px; color: #666;">Fire Radiative Power</p>
            <p style="margin: 2px 0; font-weight: bold;">${fire.frp.toFixed(1)} MW</p>
          </div>
          <div>
            <p style="margin: 2px 0; font-size: 12px; color: #666;">Brightness</p>
            <p style="margin: 2px 0; font-weight: bold;">${fire.brightness.toFixed(1)} K</p>
          </div>
          <div>
            <p style="margin: 2px 0; font-size: 12px; color: #666;">Confidence</p>
            <p style="margin: 2px 0; font-weight: bold;">${fire.confidence.toFixed(1)}%</p>
          </div>
          <div>
            <p style="margin: 2px 0; font-size: 12px; color: #666;">Detection Time</p>
            <p style="margin: 2px 0; font-weight: bold;">${fire.acq_time}</p>
          </div>
        </div>
        
        <div style="border-top: 1px solid #ddd; padding-top: 10px;">
          <p style="margin: 2px 0; font-size: 12px;"><strong>📡 Satellite:</strong> ${fire.satellite}</p>
          <p style="margin: 2px 0; font-size: 12px;"><strong>📊 Source:</strong> ${fire.source}</p>
          <p style="margin: 2px 0; font-size: 12px;"><strong>📅 Date:</strong> ${fire.acq_date}</p>
        </div>
      </div>
    `;
  };

  const getIntensityColor = (intensity: string): string => {
    switch (intensity) {
      case 'extreme': return '#8B0000';
      case 'high': return '#FF0000';
      case 'medium': return '#FFA500';
      case 'low': return '#FFD700';
      default: return '#FFA500';
    }
  };

  return (
    <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2
          }}
        >
          <CircularProgress size={60} sx={{ color: '#ff6b35' }} />
          <Typography variant="h6" sx={{ color: 'white', textAlign: 'center' }}>
            Loading Forest Fire Map...
          </Typography>
          <Typography variant="body2" sx={{ color: '#b0b0b0', textAlign: 'center' }}>
            Powered by TARS Autonomous Intelligence
          </Typography>
        </Box>
      )}
      
      <div
        ref={mapDiv}
        style={{
          height: '100%',
          width: '100%',
          background: loading ? 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)' : 'transparent'
        }}
      />
      
      {/* Map overlay info */}
      {!loading && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            zIndex: 1000
          }}
        >
          <Typography variant="caption">
            🔥 {fires.length} Active Fires | 🛰️ Real-time Data | 🤖 TARS AI
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default MapView;
