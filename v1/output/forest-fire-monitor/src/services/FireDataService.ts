import { FireData, FireSource, FireStatistics } from '../types/FireTypes';

export class FireDataService {
  private readonly NASA_FIRMS_URL = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/c6f775e93e0bb107dd292a6b94eb642e/VIIRS_SNPP_NRT/world/1';
  private readonly CANADA_FIRES_URL = 'https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/activefires.csv';
  
  private fireData: FireData[] = [];
  private updateInterval: NodeJS.Timeout | null = null;

  async loadAllFireData(): Promise<FireData[]> {
    try {
      console.log('🔥 Loading fire data from multiple sources...');
      
      // Load NASA FIRMS data
      const nasaData = await this.loadNASAFirms();
      
      // Load Canadian fire data
      const canadaData = await this.loadCanadianFires();
      
      // Combine and process data
      this.fireData = [...nasaData, ...canadaData];
      
      console.log(`✅ Loaded ${this.fireData.length} active fires`);
      return this.fireData;
    } catch (error) {
      console.error('❌ Error loading fire data:', error);
      // Use comprehensive mock data for demo
      this.fireData = this.generateComprehensiveMockData();
      return this.fireData;
    }
  }

  private async loadNASAFirms(): Promise<FireData[]> {
    try {
      // For demo purposes, create realistic mock data
      // In production, this would make actual API calls
      return this.generateMockNASAData();
    } catch (error) {
      console.error('Error loading NASA FIRMS data:', error);
      return [];
    }
  }

  private async loadCanadianFires(): Promise<FireData[]> {
    try {
      // For demo purposes, create realistic mock data
      // In production, this would make actual API calls
      return this.generateMockCanadianData();
    } catch (error) {
      console.error('Error loading Canadian fire data:', error);
      return [];
    }
  }

  private generateComprehensiveMockData(): FireData[] {
    const fires: FireData[] = [];
    
    // Major fire-prone regions with realistic coordinates
    const fireRegions = [
      // California
      { lat: 34.0522, lon: -118.2437, state: 'California', country: 'USA', intensity: 'high' },
      { lat: 37.7749, lon: -122.4194, state: 'California', country: 'USA', intensity: 'medium' },
      { lat: 36.7783, lon: -119.4179, state: 'California', country: 'USA', intensity: 'high' },
      
      // Oregon & Washington
      { lat: 45.5152, lon: -122.6784, state: 'Oregon', country: 'USA', intensity: 'medium' },
      { lat: 47.0379, lon: -122.9007, state: 'Washington', country: 'USA', intensity: 'low' },
      
      // Colorado & Montana
      { lat: 39.7392, lon: -104.9903, state: 'Colorado', country: 'USA', intensity: 'medium' },
      { lat: 46.8059, lon: -110.3626, state: 'Montana', country: 'USA', intensity: 'low' },
      
      // Canadian Provinces
      { lat: 53.5461, lon: -113.4938, state: 'Alberta', country: 'Canada', intensity: 'high' },
      { lat: 49.2827, lon: -123.1207, state: 'British Columbia', country: 'Canada', intensity: 'extreme' },
      { lat: 50.4452, lon: -104.6189, state: 'Saskatchewan', country: 'Canada', intensity: 'medium' },
      { lat: 53.7609, lon: -98.8139, state: 'Manitoba', country: 'Canada', intensity: 'low' },
      { lat: 51.2538, lon: -85.3232, state: 'Ontario', country: 'Canada', intensity: 'medium' },
      
      // Additional US States
      { lat: 33.4484, lon: -112.0740, state: 'Arizona', country: 'USA', intensity: 'high' },
      { lat: 39.3210, lon: -111.0937, state: 'Utah', country: 'USA', intensity: 'medium' },
      { lat: 43.0759, lon: -107.2903, state: 'Wyoming', country: 'USA', intensity: 'low' },
      { lat: 64.0685, lon: -152.2782, state: 'Alaska', country: 'USA', intensity: 'medium' },
    ];

    fireRegions.forEach((region, regionIndex) => {
      // Generate multiple fires per region
      const fireCount = Math.floor(Math.random() * 8) + 3; // 3-10 fires per region
      
      for (let i = 0; i < fireCount; i++) {
        const baseId = `fire_${regionIndex}_${i}`;
        const source = Math.random() > 0.6 ? FireSource.NASA_FIRMS : FireSource.CANADA_CWFIS;
        
        // Add realistic variation to coordinates
        const latVariation = (Math.random() - 0.5) * 2; // ±1 degree
        const lonVariation = (Math.random() - 0.5) * 2; // ±1 degree
        
        const intensity = this.getRandomIntensity(region.intensity);
        const frp = this.generateRealisticFRP(intensity);
        const brightness = this.generateRealisticBrightness(intensity);
        const confidence = Math.random() * 30 + 70; // 70-100%
        
        fires.push({
          id: baseId,
          latitude: region.lat + latVariation,
          longitude: region.lon + lonVariation,
          brightness: brightness,
          confidence: confidence,
          frp: frp,
          acq_date: this.getRandomRecentDate(),
          acq_time: this.getRandomTime(),
          satellite: source === FireSource.NASA_FIRMS ? 'VIIRS_SNPP_NRT' : 'CWFIS',
          source: source,
          intensity: intensity,
          location: `${region.state}, ${region.country}`,
          track: Math.floor(Math.random() * 100),
          version: '2.0',
          bright_t31: brightness - Math.random() * 50,
          daynight: Math.random() > 0.7 ? 'N' : 'D'
        });
      }
    });

    // Add some random fires in remote areas
    for (let i = 0; i < 15; i++) {
      const randomLat = Math.random() * 40 + 30; // 30-70 degrees north
      const randomLon = Math.random() * 60 - 130; // -130 to -70 degrees west
      const intensity = this.getRandomIntensity('medium');
      
      fires.push({
        id: `random_fire_${i}`,
        latitude: randomLat,
        longitude: randomLon,
        brightness: this.generateRealisticBrightness(intensity),
        confidence: Math.random() * 25 + 60,
        frp: this.generateRealisticFRP(intensity),
        acq_date: this.getRandomRecentDate(),
        acq_time: this.getRandomTime(),
        satellite: 'VIIRS_SNPP_NRT',
        source: FireSource.NASA_FIRMS,
        intensity: intensity,
        location: 'Remote Area',
        track: Math.floor(Math.random() * 100),
        version: '2.0',
        daynight: Math.random() > 0.5 ? 'N' : 'D'
      });
    }

    return fires;
  }

  private generateMockNASAData(): FireData[] {
    // This would be replaced with actual NASA FIRMS API integration
    return this.generateComprehensiveMockData().filter(fire => fire.source === FireSource.NASA_FIRMS);
  }

  private generateMockCanadianData(): FireData[] {
    // This would be replaced with actual Canadian fire data integration
    return this.generateComprehensiveMockData().filter(fire => fire.source === FireSource.CANADA_CWFIS);
  }

  private getRandomIntensity(baseIntensity: string): 'low' | 'medium' | 'high' | 'extreme' {
    const intensities = ['low', 'medium', 'high', 'extreme'];
    const baseIndex = intensities.indexOf(baseIntensity);
    
    // Add some variation around the base intensity
    const variation = Math.floor(Math.random() * 3) - 1; // -1, 0, or 1
    const newIndex = Math.max(0, Math.min(intensities.length - 1, baseIndex + variation));
    
    return intensities[newIndex] as 'low' | 'medium' | 'high' | 'extreme';
  }

  private generateRealisticFRP(intensity: string): number {
    switch (intensity) {
      case 'extreme':
        return Math.random() * 200 + 100; // 100-300 MW
      case 'high':
        return Math.random() * 100 + 50;  // 50-150 MW
      case 'medium':
        return Math.random() * 50 + 20;   // 20-70 MW
      case 'low':
        return Math.random() * 20 + 5;    // 5-25 MW
      default:
        return Math.random() * 50 + 10;
    }
  }

  private generateRealisticBrightness(intensity: string): number {
    switch (intensity) {
      case 'extreme':
        return Math.random() * 200 + 400; // 400-600 K
      case 'high':
        return Math.random() * 150 + 350; // 350-500 K
      case 'medium':
        return Math.random() * 100 + 300; // 300-400 K
      case 'low':
        return Math.random() * 50 + 280;  // 280-330 K
      default:
        return Math.random() * 100 + 300;
    }
  }

  private getRandomRecentDate(): string {
    const now = new Date();
    const daysBack = Math.floor(Math.random() * 3); // 0-2 days back
    const date = new Date(now.getTime() - daysBack * 24 * 60 * 60 * 1000);
    return date.toISOString().split('T')[0];
  }

  private getRandomTime(): string {
    const hours = Math.floor(Math.random() * 24).toString().padStart(2, '0');
    const minutes = Math.floor(Math.random() * 60).toString().padStart(2, '0');
    return `${hours}:${minutes}`;
  }

  public getFireData(): FireData[] {
    return this.fireData;
  }

  public getFireStats(): FireStatistics {
    const total = this.fireData.length;
    
    const byIntensity = {
      low: this.fireData.filter(f => f.intensity === 'low').length,
      medium: this.fireData.filter(f => f.intensity === 'medium').length,
      high: this.fireData.filter(f => f.intensity === 'high').length,
      extreme: this.fireData.filter(f => f.intensity === 'extreme').length
    };
    
    const bySource = {
      nasa: this.fireData.filter(f => f.source === FireSource.NASA_FIRMS).length,
      canada: this.fireData.filter(f => f.source === FireSource.CANADA_CWFIS).length,
      nifc: this.fireData.filter(f => f.source === FireSource.NIFC).length,
      other: this.fireData.filter(f => ![FireSource.NASA_FIRMS, FireSource.CANADA_CWFIS, FireSource.NIFC].includes(f.source)).length
    };
    
    const byRegion: { [region: string]: number } = {};
    this.fireData.forEach(fire => {
      byRegion[fire.location] = (byRegion[fire.location] || 0) + 1;
    });

    return {
      total,
      byIntensity,
      bySource,
      byRegion,
      lastUpdated: new Date()
    };
  }

  public destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
}
