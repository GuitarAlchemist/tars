import React from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { ChartData } from '@/types';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface ChartContainerProps {
  title: string;
  subtitle?: string;
  data?: ChartData;
  type: 'line' | 'bar' | 'doughnut' | 'area';
  isLoading?: boolean;
  height?: number;
}

function ChartContainer({ 
  title, 
  subtitle, 
  data, 
  type, 
  isLoading = false,
  height = 300 
}: ChartContainerProps) {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          color: 'rgb(107, 114, 128)', // gray-500
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
      },
    },
    scales: type !== 'doughnut' ? {
      x: {
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
        },
      },
      y: {
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
        },
        beginAtZero: true,
      },
    } : undefined,
    elements: {
      point: {
        radius: 4,
        hoverRadius: 6,
      },
      line: {
        tension: 0.4,
      },
    },
  };

  const renderChart = () => {
    if (!data) return null;

    // Modify data for area chart
    const chartData = type === 'area' ? {
      ...data,
      datasets: data.datasets.map(dataset => ({
        ...dataset,
        fill: true,
        backgroundColor: dataset.backgroundColor || 'rgba(59, 130, 246, 0.1)',
      }))
    } : data;

    switch (type) {
      case 'line':
      case 'area':
        return <Line data={chartData} options={chartOptions} />;
      case 'bar':
        return <Bar data={chartData} options={chartOptions} />;
      case 'doughnut':
        return <Doughnut data={chartData} options={chartOptions} />;
      default:
        return null;
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h3 className="card-title text-lg">{title}</h3>
          {subtitle && (
            <p className="card-description">{subtitle}</p>
          )}
        </div>
      </div>
      
      <div className="card-content">
        <div style={{ height: `${height}px` }} className="relative">
          {isLoading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <LoadingSpinner size="md" text="Loading chart..." />
            </div>
          ) : data ? (
            renderChart()
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 dark:text-gray-400">
              <div className="text-center">
                <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center mx-auto mb-2">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <p className="text-sm">No data available</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ChartContainer;
