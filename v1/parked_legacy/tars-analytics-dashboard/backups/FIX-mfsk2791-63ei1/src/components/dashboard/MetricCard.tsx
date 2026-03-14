import React from 'react';
import { LucideIcon, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { motion } from 'framer-motion';

interface MetricCardProps {
  title: string;
  value: string;
  change: number;
  icon: LucideIcon;
  color: 'green' | 'blue' | 'purple' | 'orange' | 'red';
  subtitle?: string;
}

const colorClasses = {
  green: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    icon: 'text-green-600 dark:text-green-400',
    border: 'border-green-200 dark:border-green-800',
  },
  blue: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    icon: 'text-blue-600 dark:text-blue-400',
    border: 'border-blue-200 dark:border-blue-800',
  },
  purple: {
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    icon: 'text-purple-600 dark:text-purple-400',
    border: 'border-purple-200 dark:border-purple-800',
  },
  orange: {
    bg: 'bg-orange-50 dark:bg-orange-900/20',
    icon: 'text-orange-600 dark:text-orange-400',
    border: 'border-orange-200 dark:border-orange-800',
  },
  red: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    icon: 'text-red-600 dark:text-red-400',
    border: 'border-red-200 dark:border-red-800',
  },
};

function MetricCard({ title, value, change, icon: Icon, color, subtitle }: MetricCardProps) {
  const isPositive = change >= 0;
  const classes = colorClasses[color];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="card hover:shadow-md transition-shadow duration-200"
    >
      <div className="card-content">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
              {title}
            </p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
              {value}
            </p>
            {subtitle && (
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                {subtitle}
              </p>
            )}
          </div>
          
          <div className={`p-3 rounded-lg ${classes.bg} ${classes.border} border`}>
            <Icon className={`w-6 h-6 ${classes.icon}`} />
          </div>
        </div>

        {/* Change indicator */}
        <div className="flex items-center mt-4">
          <div className={`flex items-center space-x-1 ${
            isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
          }`}>
            {isPositive ? (
              <ArrowUpRight className="w-4 h-4" />
            ) : (
              <ArrowDownRight className="w-4 h-4" />
            )}
            <span className="text-sm font-medium">
              {Math.abs(change).toFixed(1)}%
            </span>
          </div>
          <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
            vs last month
          </span>
        </div>
      </div>
    </motion.div>
  );
}

export default MetricCard;
