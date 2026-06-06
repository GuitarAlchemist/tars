import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  User, 
  LoginCredentials, 
  ApiResponse, 
  DashboardMetrics, 
  ChartData, 
  RealtimeData,
  PaginatedResponse 
} from '@/types';
import { tokenStorage } from '@/utils/storage';

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = tokenStorage.getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      tokenStorage.removeToken();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Mock data for demonstration (will be replaced with real API calls)
const mockUsers: User[] = [
  {
    id: '1',
    email: 'admin@tars.ai',
    name: 'TARS Administrator',
    role: 'admin',
    avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150',
    createdAt: '2024-01-01T00:00:00Z',
    lastLogin: new Date().toISOString(),
    isActive: true,
  },
  {
    id: '2',
    email: 'user@tars.ai',
    name: 'Demo User',
    role: 'user',
    avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150',
    createdAt: '2024-01-15T00:00:00Z',
    lastLogin: new Date(Date.now() - 86400000).toISOString(),
    isActive: true,
  },
];

const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwibmFtZSI6IlRBUlMgQWRtaW5pc3RyYXRvciIsImlhdCI6MTUxNjIzOTAyMn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c';

// Authentication API
export const authApi = {
  async login(credentials: LoginCredentials): Promise<ApiResponse<{ user: User; token: string }>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Mock authentication logic
    if (credentials.email === 'admin@tars.ai' && credentials.password === 'admin123') {
      return {
        success: true,
        data: {
          user: mockUsers[0],
          token: mockToken,
        },
        timestamp: new Date().toISOString(),
      };
    } else if (credentials.email === 'user@tars.ai' && credentials.password === 'user123') {
      return {
        success: true,
        data: {
          user: mockUsers[1],
          token: mockToken,
        },
        timestamp: new Date().toISOString(),
      };
    } else {
      throw new Error('Invalid credentials');
    }
  },

  async validateToken(token: string): Promise<User> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Mock token validation
    if (token === mockToken) {
      return mockUsers[0];
    } else {
      throw new Error('Invalid token');
    }
  },

  async refreshToken(token: string): Promise<ApiResponse<{ user: User; token: string }>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    return {
      success: true,
      data: {
        user: mockUsers[0],
        token: mockToken,
      },
      timestamp: new Date().toISOString(),
    };
  },

  async logout(): Promise<ApiResponse> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    return {
      success: true,
      message: 'Logged out successfully',
      timestamp: new Date().toISOString(),
    };
  },
};

// Dashboard API
export const dashboardApi = {
  async getMetrics(): Promise<ApiResponse<DashboardMetrics>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    const metrics: DashboardMetrics = {
      totalUsers: 12847,
      totalRevenue: 284750.50,
      conversionRate: 3.24,
      activeUsers: 1847,
      growthRate: 12.5,
      lastUpdated: new Date().toISOString(),
    };
    
    return {
      success: true,
      data: metrics,
      timestamp: new Date().toISOString(),
    };
  },

  async getChartData(type: 'revenue' | 'users' | 'conversions'): Promise<ApiResponse<ChartData>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    const generateMockData = (type: string) => {
      const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
      let data: number[];
      
      switch (type) {
        case 'revenue':
          data = [45000, 52000, 48000, 61000, 55000, 67000];
          break;
        case 'users':
          data = [1200, 1900, 1500, 2100, 1800, 2400];
          break;
        case 'conversions':
          data = [2.1, 3.2, 2.8, 3.7, 3.1, 4.2];
          break;
        default:
          data = [10, 20, 15, 25, 20, 30];
      }
      
      return {
        labels,
        datasets: [
          {
            label: type.charAt(0).toUpperCase() + type.slice(1),
            data,
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderColor: 'rgba(59, 130, 246, 1)',
            borderWidth: 2,
            fill: true,
          },
        ],
      };
    };
    
    return {
      success: true,
      data: generateMockData(type),
      timestamp: new Date().toISOString(),
    };
  },
};

// Users API
export const usersApi = {
  async getUsers(page = 1, limit = 10): Promise<PaginatedResponse<User>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    return {
      success: true,
      data: mockUsers,
      pagination: {
        page,
        limit,
        total: mockUsers.length,
        totalPages: Math.ceil(mockUsers.length / limit),
      },
      timestamp: new Date().toISOString(),
    };
  },

  async createUser(userData: Partial<User>): Promise<ApiResponse<User>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    const newUser: User = {
      id: Date.now().toString(),
      email: userData.email || '',
      name: userData.name || '',
      role: userData.role || 'user',
      createdAt: new Date().toISOString(),
      isActive: true,
      ...userData,
    };
    
    return {
      success: true,
      data: newUser,
      timestamp: new Date().toISOString(),
    };
  },

  async updateUser(id: string, userData: Partial<User>): Promise<ApiResponse<User>> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    const user = mockUsers.find(u => u.id === id);
    if (!user) {
      throw new Error('User not found');
    }
    
    const updatedUser = { ...user, ...userData };
    
    return {
      success: true,
      data: updatedUser,
      timestamp: new Date().toISOString(),
    };
  },

  async deleteUser(id: string): Promise<ApiResponse> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    return {
      success: true,
      message: 'User deleted successfully',
      timestamp: new Date().toISOString(),
    };
  },
};

// Realtime API (WebSocket simulation)
export const realtimeApi = {
  connect(onMessage: (data: RealtimeData) => void): () => void {
    const interval = setInterval(() => {
      const data: RealtimeData = {
        timestamp: new Date().toISOString(),
        activeUsers: Math.floor(Math.random() * 100) + 1800,
        pageViews: Math.floor(Math.random() * 500) + 2000,
        conversions: Math.floor(Math.random() * 20) + 50,
        revenue: Math.floor(Math.random() * 1000) + 5000,
      };
      onMessage(data);
    }, 3000);

    return () => clearInterval(interval);
  },
};

export default api;
