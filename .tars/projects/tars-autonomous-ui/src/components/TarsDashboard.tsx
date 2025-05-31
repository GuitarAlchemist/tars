// TARS Dashboard - Autonomously designed and coded by TARS to monitor itself
import React from 'react';
import { Activity, Users, FolderOpen, Zap, FileText, Terminal, Cpu } from 'lucide-react';
import { useTarsStore } from '../stores/tarsStore';

export const TarsDashboard: React.FC = () => {
  const { status, projects, agents, metrics, commands, metascripts } = useTarsStore();
  
  return (
    <div className="space-y-6">
      {/* TARS Dashboard Header */}
      <div className="flex items-center space-x-3">
        <Cpu className="h-8 w-8 text-cyan-400" />
        <h2 className="text-3xl font-bold text-cyan-400 font-mono">TARS Control Center</h2>
        <span className="text-gray-400">Autonomous System Dashboard</span>
      </div>
      
      {/* Status Cards - TARS monitors its own capabilities */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 hover:border-cyan-400 transition-colors">
          <div className="flex items-center space-x-3">
            <Activity className="h-8 w-8 text-green-400" />
            <div>
              <p className="text-gray-400 text-sm">System Status</p>
              <p className="text-xl font-bold text-white">
                {status?.online ? 'Online' : 'Offline'}
              </p>
              <p className="text-xs text-gray-500">
                Uptime: {status?.uptime ? Math.floor(status.uptime / 3600) : 0}h
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 hover:border-blue-400 transition-colors">
          <div className="flex items-center space-x-3">
            <Users className="h-8 w-8 text-blue-400" />
            <div>
              <p className="text-gray-400 text-sm">Active Agents</p>
              <p className="text-xl font-bold text-white">{agents.length}</p>
              <p className="text-xs text-gray-500">
                {agents.filter(a => a.status === 'busy').length} working
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 hover:border-purple-400 transition-colors">
          <div className="flex items-center space-x-3">
            <FolderOpen className="h-8 w-8 text-purple-400" />
            <div>
              <p className="text-gray-400 text-sm">Projects</p>
              <p className="text-xl font-bold text-white">{projects.length}</p>
              <p className="text-xs text-gray-500">
                {projects.filter(p => p.status === 'active').length} active
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 hover:border-yellow-400 transition-colors">
          <div className="flex items-center space-x-3">
            <Zap className="h-8 w-8 text-yellow-400" />
            <div>
              <p className="text-gray-400 text-sm">CUDA Performance</p>
              <p className="text-xl font-bold text-white">
                {metrics?.cuda_searches_per_sec ? 
                  `${(metrics.cuda_searches_per_sec / 1000000).toFixed(0)}M/s` : 
                  'N/A'
                }
              </p>
              <p className="text-xs text-gray-500">
                GPU: {metrics?.gpu_memory_used || 0}MB used
              </p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Agent Activity */}
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <Users className="h-5 w-5 text-blue-400" />
            <h3 className="text-xl font-bold text-white">Agent Activity</h3>
          </div>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {agents.length > 0 ? agents.map((agent) => (
              <div key={agent.id} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                <div>
                  <p className="text-white font-medium">{agent.name}</p>
                  <p className="text-gray-400 text-sm">{agent.persona}</p>
                  <p className="text-gray-500 text-xs">{agent.task || 'Idle'}</p>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  agent.status === 'busy' ? 'bg-yellow-600 text-yellow-100' :
                  agent.status === 'idle' ? 'bg-green-600 text-green-100' :
                  'bg-red-600 text-red-100'
                }`}>
                  {agent.status}
                </span>
              </div>
            )) : (
              <p className="text-gray-400 text-center py-4">No agents active</p>
            )}
          </div>
        </div>
        
        {/* Recent Projects */}
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <FolderOpen className="h-5 w-5 text-purple-400" />
            <h3 className="text-xl font-bold text-white">Recent Projects</h3>
          </div>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {projects.length > 0 ? projects.map((project) => (
              <div key={project.id} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                <div>
                  <p className="text-white font-medium">{project.name}</p>
                  <p className="text-gray-400 text-sm">{project.description}</p>
                  <p className="text-gray-500 text-xs">Created: {project.created}</p>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  project.status === 'active' ? 'bg-blue-600 text-blue-100' :
                  project.status === 'completed' ? 'bg-green-600 text-green-100' :
                  project.status === 'creating' ? 'bg-yellow-600 text-yellow-100' :
                  'bg-red-600 text-red-100'
                }`}>
                  {project.status}
                </span>
              </div>
            )) : (
              <p className="text-gray-400 text-center py-4">No projects yet</p>
            )}
          </div>
        </div>
      </div>
      
      {/* Command History */}
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <div className="flex items-center space-x-2 mb-4">
          <Terminal className="h-5 w-5 text-green-400" />
          <h3 className="text-xl font-bold text-white">Command History</h3>
        </div>
        <div className="space-y-2 max-h-48 overflow-y-auto font-mono text-sm">
          {commands.length > 0 ? commands.slice(0, 10).map((command) => (
            <div key={command.id} className="flex items-center space-x-3 p-2 bg-gray-900 rounded">
              <span className={`w-2 h-2 rounded-full ${
                command.status === 'completed' ? 'bg-green-400' :
                command.status === 'running' ? 'bg-yellow-400' :
                command.status === 'error' ? 'bg-red-400' :
                'bg-gray-400'
              }`} />
              <span className="text-cyan-400">$</span>
              <span className="text-white flex-1">{command.command}</span>
              <span className="text-gray-500 text-xs">{command.timestamp}</span>
            </div>
          )) : (
            <p className="text-gray-400 text-center py-4">No commands executed yet</p>
          )}
        </div>
      </div>
    </div>
  );
};
