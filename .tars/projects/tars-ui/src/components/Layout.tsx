import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  FolderOpen, 
  FileText, 
  Users, 
  Activity, 
  Settings,
  Cpu,
  Zap
} from 'lucide-react';
import { useSystemStatus } from '../hooks/useSystemStatus';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  const { data: systemStatus } = useSystemStatus();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Home },
    { name: 'Projects', href: '/projects', icon: FolderOpen },
    { name: 'Metascripts', href: '/metascripts', icon: FileText },
    { name: 'Agents', href: '/agents', icon: Users },
    { name: 'Performance', href: '/performance', icon: Activity },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Cpu className="h-8 w-8 text-tars-cyan animate-pulse-cyan" />
                <h1 className="text-2xl font-bold text-tars-cyan terminal-text">TARS</h1>
                <span className="text-sm text-gray-400">Autonomous System</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {systemStatus?.cudaAvailable && (
                <div className="flex items-center space-x-1 text-green-400">
                  <Zap className="h-4 w-4" />
                  <span className="text-sm">CUDA</span>
                </div>
              )}
              <div className={`h-3 w-3 rounded-full ${systemStatus?.isOnline ? 'bg-green-400' : 'bg-red-400'}`} />
              <span className="text-sm text-gray-400">
                {systemStatus?.isOnline ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <nav className="w-64 bg-gray-800 min-h-screen border-r border-gray-700">
          <div className="p-4">
            <ul className="space-y-2">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-tars-cyan text-white'
                          : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                      }`}
                    >
                      <item.icon className="h-5 w-5" />
                      <span>{item.name}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {children}
        </main>
      </div>
    </div>
  );
};
