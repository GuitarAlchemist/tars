import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Users, Eye, TrendingUp, Wifi, WifiOff } from 'lucide-react';
import { RealtimeData } from '@/types';

interface RealtimeWidgetProps {
  data: RealtimeData | null;
}

function RealtimeWidget({ data }: RealtimeWidgetProps) {
  const isConnected = !!data;

  const metrics = [
    {
      label: 'Active Users',
      value: data?.activeUsers?.toLocaleString() || '0',
      icon: Users,
      color: 'text-blue-600 dark:text-blue-400',
      bg: 'bg-blue-50 dark:bg-blue-900/20',
    },
    {
      label: 'Page Views',
      value: data?.pageViews?.toLocaleString() || '0',
      icon: Eye,
      color: 'text-green-600 dark:text-green-400',
      bg: 'bg-green-50 dark:bg-green-900/20',
    },
    {
      label: 'Conversions',
      value: data?.conversions?.toLocaleString() || '0',
      icon: TrendingUp,
      color: 'text-purple-600 dark:text-purple-400',
      bg: 'bg-purple-50 dark:bg-purple-900/20',
    },
    {
      label: 'Revenue',
      value: data ? `$${data.revenue.toLocaleString()}` : '$0',
      icon: Activity,
      color: 'text-orange-600 dark:text-orange-400',
      bg: 'bg-orange-50 dark:bg-orange-900/20',
    },
  ];

  return (
    <div className="card h-full">
      <div className="card-header">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="card-title text-lg">Real-time Analytics</h3>
            <p className="card-description">Live data updates</p>
          </div>
          
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <Wifi className="w-5 h-5 text-green-500" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-500" />
            )}
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`} />
          </div>
        </div>
      </div>

      <div className="card-content">
        <div className="space-y-4">
          {metrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50"
            >
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${metric.bg}`}>
                  <metric.icon className={`w-4 h-4 ${metric.color}`} />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {metric.label}
                  </p>
                </div>
              </div>
              
              <div className="text-right">
                <p className="text-lg font-bold text-gray-900 dark:text-white">
                  {metric.value}
                </p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Connection status */}
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500 dark:text-gray-400">
              Status
            </span>
            <span className={`font-medium ${
              isConnected 
                ? 'text-green-600 dark:text-green-400' 
                : 'text-red-600 dark:text-red-400'
            }`}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          {data && (
            <div className="flex items-center justify-between text-sm mt-2">
              <span className="text-gray-500 dark:text-gray-400">
                Last Update
              </span>
              <span className="text-gray-700 dark:text-gray-300">
                {new Date(data.timestamp).toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>

        {/* Pulse animation for live updates */}
        {isConnected && (
          <div className="mt-4 flex items-center justify-center">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" />
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RealtimeWidget;
