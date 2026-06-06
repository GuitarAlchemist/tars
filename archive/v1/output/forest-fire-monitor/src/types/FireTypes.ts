// Fire data types for the Forest Fire Monitor application

export enum FireSource {
  NASA_FIRMS = 'NASA_FIRMS',
  CANADA_CWFIS = 'CANADA_CWFIS',
  NIFC = 'NIFC',
  MODIS = 'MODIS',
  VIIRS = 'VIIRS'
}

export enum FireIntensity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  EXTREME = 'extreme'
}

export interface FireData {
  id: string;
  latitude: number;
  longitude: number;
  brightness: number;
  confidence: number;
  frp: number; // Fire Radiative Power in MW
  acq_date: string;
  acq_time: string;
  satellite: string;
  source: FireSource;
  intensity: 'low' | 'medium' | 'high' | 'extreme';
  location: string;
  track?: number;
  version?: string;
  bright_t31?: number;
  daynight?: 'D' | 'N';
}

export interface FireStatistics {
  total: number;
  byIntensity: {
    low: number;
    medium: number;
    high: number;
    extreme: number;
  };
  bySource: {
    nasa: number;
    canada: number;
    nifc: number;
    other: number;
  };
  byRegion: {
    [region: string]: number;
  };
  lastUpdated: Date;
}

export interface FireAlert {
  id: string;
  fireId: string;
  type: 'new_fire' | 'intensity_change' | 'size_change' | 'containment';
  message: string;
  timestamp: Date;
  severity: 'info' | 'warning' | 'critical';
  location: string;
}

export interface WeatherData {
  temperature: number;
  humidity: number;
  windSpeed: number;
  windDirection: number;
  precipitation: number;
  pressure: number;
  visibility: number;
  uvIndex: number;
  timestamp: Date;
  location: {
    latitude: number;
    longitude: number;
  };
}

export interface FirePrediction {
  fireId: string;
  predictedSpread: {
    direction: number; // degrees
    speed: number; // km/h
    area: number; // hectares
  };
  riskLevel: 'low' | 'moderate' | 'high' | 'extreme';
  confidence: number; // 0-100
  timeframe: number; // hours
  factors: string[];
}

export interface MapViewState {
  center: [number, number];
  zoom: number;
  rotation: number;
  extent: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
}

export interface FilterOptions {
  dateRange: {
    start: Date;
    end: Date;
  };
  intensityFilter: FireIntensity[];
  sourceFilter: FireSource[];
  confidenceThreshold: number;
  frpThreshold: number;
  regionFilter: string[];
}

export interface VisualizationMode {
  type: 'points' | 'heatmap' | 'clusters' | 'density';
  showLabels: boolean;
  showAnimation: boolean;
  colorScheme: 'intensity' | 'source' | 'age' | 'confidence';
  symbolSize: 'fixed' | 'proportional';
}

export interface UserPreferences {
  theme: 'light' | 'dark';
  units: 'metric' | 'imperial';
  language: string;
  notifications: {
    newFires: boolean;
    intensityChanges: boolean;
    nearbyFires: boolean;
    systemUpdates: boolean;
  };
  defaultView: MapViewState;
  autoRefresh: boolean;
  refreshInterval: number; // minutes
}

export interface APIResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  timestamp: Date;
  source: string;
  count?: number;
}

export interface FireDataResponse extends APIResponse<FireData[]> {
  metadata: {
    totalRecords: number;
    filteredRecords: number;
    lastUpdate: Date;
    sources: FireSource[];
    coverage: {
      north: number;
      south: number;
      east: number;
      west: number;
    };
  };
}

// Event types for real-time updates
export interface FireUpdateEvent {
  type: 'fire_added' | 'fire_updated' | 'fire_removed' | 'bulk_update';
  data: FireData | FireData[];
  timestamp: Date;
}

export interface SystemEvent {
  type: 'data_refresh' | 'api_error' | 'connection_lost' | 'connection_restored';
  message: string;
  timestamp: Date;
  severity: 'info' | 'warning' | 'error';
}

// Component prop types
export interface MapComponentProps {
  fires: FireData[];
  onFireSelect?: (fire: FireData) => void;
  onMapMove?: (viewState: MapViewState) => void;
  visualizationMode: VisualizationMode;
  filterOptions: FilterOptions;
  showWeather?: boolean;
  showPredictions?: boolean;
}

export interface FirePanelProps {
  selectedFire?: FireData;
  statistics: FireStatistics;
  alerts: FireAlert[];
  onClose?: () => void;
  onAlertDismiss?: (alertId: string) => void;
}

export interface ControlPanelProps {
  filterOptions: FilterOptions;
  visualizationMode: VisualizationMode;
  onFilterChange: (filters: FilterOptions) => void;
  onVisualizationChange: (mode: VisualizationMode) => void;
  onRefresh: () => void;
  isLoading: boolean;
  lastUpdate?: Date;
}

// Utility types
export type FireDataKey = keyof FireData;
export type SortDirection = 'asc' | 'desc';
export type SortOptions = {
  key: FireDataKey;
  direction: SortDirection;
};

export interface ExportOptions {
  format: 'csv' | 'json' | 'kml' | 'geojson';
  includeMetadata: boolean;
  dateRange?: {
    start: Date;
    end: Date;
  };
  filters?: FilterOptions;
}
