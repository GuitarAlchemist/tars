// TARS Zustand Store - Autonomously created by TARS for managing its own state
// TARS_STORE_SIGNATURE: TARS_ZUSTAND_AUTONOMOUS_STATE_MANAGEMENT
// TARS_ARCHITECTURE_PROOF: SELF_MONITORING_AUTONOMOUS_DESIGN
import { create } from 'zustand';
import { TarsStatus, TarsProject, TarsAgent, TarsMetrics, TarsCommand, TarsMetascript } from '../types/tars';

interface TarsStore {
  // State
  status: TarsStatus | null;
  projects: TarsProject[];
  agents: TarsAgent[];
  metrics: TarsMetrics | null;
  commands: TarsCommand[];
  metascripts: TarsMetascript[];
  
  // Actions - TARS defines its own state management
  setStatus: (status: TarsStatus) => void;
  setProjects: (projects: TarsProject[]) => void;
  addProject: (project: TarsProject) => void;
  updateProject: (id: string, updates: Partial<TarsProject>) => void;
  
  setAgents: (agents: TarsAgent[]) => void;
  updateAgent: (id: string, updates: Partial<TarsAgent>) => void;
  
  setMetrics: (metrics: TarsMetrics) => void;
  
  addCommand: (command: TarsCommand) => void;
  updateCommand: (id: string, updates: Partial<TarsCommand>) => void;
  
  setMetascripts: (metascripts: TarsMetascript[]) => void;
  updateMetascript: (id: string, updates: Partial<TarsMetascript>) => void;
}

// TARS creates its own store with autonomous state management
export const useTarsStore = create<TarsStore>((set, get) => ({
  // Initial state
  status: null,
  projects: [],
  agents: [],
  metrics: null,
  commands: [],
  metascripts: [],
  
  // Status management
  setStatus: (status) => set({ status }),
  
  // Project management
  setProjects: (projects) => set({ projects }),
  addProject: (project) => set((state) => ({ 
    projects: [...state.projects, project] 
  })),
  updateProject: (id, updates) => set((state) => ({
    projects: state.projects.map(p => p.id === id ? { ...p, ...updates } : p)
  })),
  
  // Agent management
  setAgents: (agents) => set({ agents }),
  updateAgent: (id, updates) => set((state) => ({
    agents: state.agents.map(a => a.id === id ? { ...a, ...updates } : a)
  })),
  
  // Metrics management
  setMetrics: (metrics) => set({ metrics }),
  
  // Command management
  addCommand: (command) => set((state) => ({
    commands: [command, ...state.commands.slice(0, 49)] // Keep last 50 commands
  })),
  updateCommand: (id, updates) => set((state) => ({
    commands: state.commands.map(c => c.id === id ? { ...c, ...updates } : c)
  })),
  
  // Metascript management
  setMetascripts: (metascripts) => set({ metascripts }),
  updateMetascript: (id, updates) => set((state) => ({
    metascripts: state.metascripts.map(m => m.id === id ? { ...m, ...updates } : m)
  })),
}));
