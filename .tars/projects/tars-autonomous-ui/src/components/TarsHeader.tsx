// TARS Header Component - Autonomously designed and coded by TARS
// TARS_COMPONENT_SIGNATURE: TARS_HEADER_AUTONOMOUS_UI_GENERATION
// TARS_DESIGN_PROOF: CYAN_DARK_THEME_AUTONOMOUS_AESTHETIC_CHOICE
import React from 'react';
import { Cpu, Zap, Activity, Users } from 'lucide-react';
import { useTarsStore } from '../stores/tarsStore';

export const TarsHeader: React.FC = () => {
  const { status, agents, metrics } = useTarsStore();
  
  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* TARS Branding - Autonomously designed */}
        <div className="flex items-center space-x-3">
          <Cpu className="h-8 w-8 text-cyan-400 animate-pulse" />
          <div>
            <h1 className="text-2xl font-bold text-cyan-400 font-mono">TARS</h1>
            <p className="text-xs text-gray-400">Autonomous System v{status?.version || '2.0.0'}</p>
          </div>
        </div>
        
        {/* System Status Indicators - TARS monitors itself */}
        <div className="flex items-center space-x-6">
          {/* CUDA Status */}
          {status?.cuda && (
            <div className="flex items-center space-x-2 text-green-400">
              <Zap className="h-4 w-4" />
              <div className="text-sm">
                <div className="font-medium">CUDA</div>
                <div className="text-xs text-gray-400">
                  {metrics?.cuda_searches_per_sec ? 
                    `${(metrics.cuda_searches_per_sec / 1000000).toFixed(0)}M/s` : 
                    'Active'
                  }
                </div>
              </div>
            </div>
          )}
          
          {/* Agent Status */}
          <div className="flex items-center space-x-2 text-blue-400">
            <Users className="h-4 w-4" />
            <div className="text-sm">
              <div className="font-medium">{agents.length} Agents</div>
              <div className="text-xs text-gray-400">
                {agents.filter(a => a.status === 'busy').length} Active
              </div>
            </div>
          </div>
          
          {/* System Health */}
          <div className="flex items-center space-x-2">
            <Activity className="h-4 w-4 text-yellow-400" />
            <div className="text-sm">
              <div className="font-medium text-white">
                CPU: {status?.cpuUsage?.toFixed(1) || '0.0'}%
              </div>
              <div className="text-xs text-gray-400">
                RAM: {status?.memoryUsage?.toFixed(1) || '0.0'}%
              </div>
            </div>
          </div>
          
          {/* Online Status */}
          <div className="flex items-center space-x-2">
            <div className={`h-3 w-3 rounded-full ${status?.online ? 'bg-green-400' : 'bg-red-400'}`} />
            <span className="text-sm text-gray-400">
              {status?.online ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
