import React from 'react';
import { BaseComponentProps } from '@/types';

interface LoadingSpinnerProps extends BaseComponentProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'secondary' | 'white';
  text?: string;
}

const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
  xl: 'w-12 h-12',
};

const colorClasses = {
  primary: 'border-primary-600',
  secondary: 'border-secondary-600',
  white: 'border-white',
};

function LoadingSpinner({ 
  size = 'md', 
  color = 'primary', 
  text, 
  className = '',
  testId = 'loading-spinner'
}: LoadingSpinnerProps) {
  return (
    <div 
      className={`flex flex-col items-center justify-center space-y-2 ${className}`}
      data-testid={testId}
    >
      <div
        className={`
          loading-spinner
          ${sizeClasses[size]}
          ${colorClasses[color]}
          border-t-transparent
        `}
        role="status"
        aria-label={text || 'Loading'}
      />
      {text && (
        <p className="text-sm text-gray-600 dark:text-gray-400 animate-pulse">
          {text}
        </p>
      )}
    </div>
  );
}

export default LoadingSpinner;
