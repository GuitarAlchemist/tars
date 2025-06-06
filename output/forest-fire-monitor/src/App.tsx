import React, { useState, useEffect, useCallback } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Snackbar, Alert } from '@mui/material';
import MapView from './components/Map/MapView';
import Header from './components/UI/Header';
import FirePanel from './components/UI/FirePanel';
import ControlPanel from './components/UI/ControlPanel';
import LoadingOverlay from './components/UI/LoadingOverlay';
import { FireDataService } from './services/FireDataService';
import { 
  FireData, 
  FireStatistics, 
  FireAlert, 
  FilterOptions, 
  VisualizationMode,
  FireSource,
  FireIntensity 
} from './types/FireTypes';
import './styles/App.css';

// Create dark theme for professional appearance
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#ff6b35',
      light: '#ff8a65',
      dark: '#e64a19',
    },
    secondary: {
      main: '#f7931e',
      light: '#ffb74d',
      dark: '#f57c00',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
    error: {
      main: '#f44336',
    },
    warning: {
      main: '#ff9800',
    },
    success: {
      main: '#4caf50',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

const App: React.FC = () => {
  // State management
  const [fires, setFires] = useState<FireData[]>([]);
  const [selectedFire, setSelectedFire] = useState<FireData | undefined>();
  const [statistics, setStatistics] = useState<FireStatistics>({
    total: 0,
    byIntensity: { low: 0, medium: 0, high: 0, extreme: 0 },
    bySource: { nasa: 0, canada: 0, nifc: 0, other: 0 },
    byRegion: {},
    lastUpdated: new Date(),
  });
  const [alerts, setAlerts] = useState<FireAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Filter and visualization state
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    dateRange: {
      start: new Date(Date.now() - 24 * 60 * 60 * 1000), // Last 24 hours
      end: new Date(),
    },
    intensityFilter: [FireIntensity.LOW, FireIntensity.MEDIUM, FireIntensity.HIGH],
    sourceFilter: [FireSource.NASA_FIRMS, FireSource.CANADA_CWFIS],
    confidenceThreshold: 50,
    frpThreshold: 0,
    regionFilter: [],
  });

  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>({
    type: 'points',
    showLabels: false,
    showAnimation: true,
    colorScheme: 'intensity',
    symbolSize: 'proportional',
  });

  // Fire data service
  const [fireService] = useState(() => new FireDataService());

  // Load initial fire data
  useEffect(() => {
    const loadFireData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        console.log('🔥 Loading forest fire data...');
        const fireData = await fireService.loadAllFireData();
        
        setFires(fireData);
        setStatistics(fireService.getFireStats());
        setLastUpdate(new Date());
        
        // Generate some sample alerts
        const sampleAlerts: FireAlert[] = [
          {
            id: 'alert_1',
            fireId: fireData[0]?.id || 'unknown',
            type: 'new_fire',
            message: 'New high-intensity fire detected in California',
            timestamp: new Date(),
            severity: 'critical',
            location: 'California, USA',
          },
          {
            id: 'alert_2',
            fireId: fireData[1]?.id || 'unknown',
            type: 'intensity_change',
            message: 'Fire intensity increased in British Columbia',
            timestamp: new Date(Date.now() - 30 * 60 * 1000),
            severity: 'warning',
            location: 'British Columbia, Canada',
          },
        ];
        setAlerts(sampleAlerts);
        
        console.log(`✅ Loaded ${fireData.length} active fires`);
      } catch (err) {
        console.error('❌ Error loading fire data:', err);
        setError('Failed to load fire data. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    loadFireData();

    // Set up auto-refresh every 15 minutes
    const refreshInterval = setInterval(loadFireData, 15 * 60 * 1000);

    return () => {
      clearInterval(refreshInterval);
      fireService.destroy();
    };
  }, [fireService]);

  // Handle fire selection
  const handleFireSelect = useCallback((fire: FireData) => {
    setSelectedFire(fire);
  }, []);

  // Handle filter changes
  const handleFilterChange = useCallback((newFilters: FilterOptions) => {
    setFilterOptions(newFilters);
  }, []);

  // Handle visualization mode changes
  const handleVisualizationChange = useCallback((newMode: VisualizationMode) => {
    setVisualizationMode(newMode);
  }, []);

  // Handle manual refresh
  const handleRefresh = useCallback(async () => {
    try {
      setLoading(true);
      const fireData = await fireService.loadAllFireData();
      setFires(fireData);
      setStatistics(fireService.getFireStats());
      setLastUpdate(new Date());
      console.log('🔄 Fire data refreshed');
    } catch (err) {
      console.error('❌ Error refreshing data:', err);
      setError('Failed to refresh data. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [fireService]);

  // Handle alert dismissal
  const handleAlertDismiss = useCallback((alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  }, []);

  // Handle error dismissal
  const handleErrorDismiss = useCallback(() => {
    setError(null);
  }, []);

  // Filter fires based on current filter options
  const filteredFires = fires.filter(fire => {
    // Date range filter
    const fireDate = new Date(fire.acq_date);
    if (fireDate < filterOptions.dateRange.start || fireDate > filterOptions.dateRange.end) {
      return false;
    }

    // Intensity filter
    if (!filterOptions.intensityFilter.includes(fire.intensity as FireIntensity)) {
      return false;
    }

    // Source filter
    if (!filterOptions.sourceFilter.includes(fire.source)) {
      return false;
    }

    // Confidence threshold
    if (fire.confidence < filterOptions.confidenceThreshold) {
      return false;
    }

    // FRP threshold
    if (fire.frp < filterOptions.frpThreshold) {
      return false;
    }

    return true;
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <Header 
          fireCount={filteredFires.length}
          lastUpdate={lastUpdate}
          isLoading={loading}
        />
        
        {/* Main content area */}
        <Box sx={{ flex: 1, display: 'flex', position: 'relative', overflow: 'hidden' }}>
          {/* Map view */}
          <MapView
            fires={filteredFires}
            onFireSelect={handleFireSelect}
            visualizationMode={visualizationMode}
            filterOptions={filterOptions}
          />
          
          {/* Fire information panel */}
          <FirePanel
            selectedFire={selectedFire}
            statistics={statistics}
            alerts={alerts}
            onClose={() => setSelectedFire(undefined)}
            onAlertDismiss={handleAlertDismiss}
          />
          
          {/* Control panel */}
          <ControlPanel
            filterOptions={filterOptions}
            visualizationMode={visualizationMode}
            onFilterChange={handleFilterChange}
            onVisualizationChange={handleVisualizationChange}
            onRefresh={handleRefresh}
            isLoading={loading}
            lastUpdate={lastUpdate}
          />
        </Box>
        
        {/* Loading overlay */}
        {loading && <LoadingOverlay />}
        
        {/* Error snackbar */}
        <Snackbar
          open={!!error}
          autoHideDuration={6000}
          onClose={handleErrorDismiss}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={handleErrorDismiss} severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
};

export default App;
