import React, { useEffect, useState } from 'react';
import { useQuery } from 'react-query';
import { 
  TrendingUp, 
  Users, 
  DollarSign, 
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw
} from 'lucide-react';
import { dashboardApi, realtimeApi } from '@/lib/api';
import { DashboardMetrics, RealtimeData } from '@/types';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import MetricCard from '@/components/dashboard/MetricCard';
import ChartContainer from '@/components/dashboard/ChartContainer';
import RealtimeWidget from '@/components/dashboard/RealtimeWidget';

function Dashboard() {
  const [realtimeData, setRealtimeData] = useState<RealtimeData | null>(null);

  // Fetch dashboard metrics
  const {
    data: metricsResponse,
    isLoading: metricsLoading,
    error: metricsError,
    refetch: refetchMetrics,
  } = useQuery('dashboard-metrics', dashboardApi.getMetrics, {
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Fetch chart data
  const {
    data: revenueChartResponse,
    isLoading: revenueLoading,
  } = useQuery('revenue-chart', () => dashboardApi.getChartData('revenue'));

  const {
    data: usersChartResponse,
    isLoading: usersLoading,
  } = useQuery('users-chart', () => dashboardApi.getChartData('users'));

  const {
    data: conversionsChartResponse,
    isLoading: conversionsLoading,
  } = useQuery('conversions-chart', () => dashboardApi.getChartData('conversions'));

  // Setup realtime connection
  useEffect(() => {
    const disconnect = realtimeApi.connect((data) => {
      setRealtimeData(data);
    });

    return disconnect;
  }, []);

  const metrics = metricsResponse?.data;
  const revenueChart = revenueChartResponse?.data;
  const usersChart = usersChartResponse?.data;
  const conversionsChart = conversionsChartResponse?.data;

  if (metricsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Loading dashboard..." />
      </div>
    );
  }

  if (metricsError) {
    return (
      <div className="text-center py-12">
        <div className="text-red-500 mb-4">
          <Activity className="w-12 h-12 mx-auto" />
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Failed to load dashboard
        </h3>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          There was an error loading the dashboard data.
        </p>
        <button
          onClick={() => refetchMetrics()}
          className="btn btn-primary flex items-center space-x-2 mx-auto"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Retry</span>
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Welcome back! Here's what's happening with your analytics.
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => refetchMetrics()}
            className="btn btn-outline flex items-center space-x-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Revenue"
          value={`$${metrics?.totalRevenue?.toLocaleString() || '0'}`}
          change={metrics?.growthRate || 0}
          icon={DollarSign}
          color="green"
        />
        
        <MetricCard
          title="Total Users"
          value={metrics?.totalUsers?.toLocaleString() || '0'}
          change={8.2}
          icon={Users}
          color="blue"
        />
        
        <MetricCard
          title="Active Users"
          value={metrics?.activeUsers?.toLocaleString() || '0'}
          change={-2.1}
          icon={Activity}
          color="purple"
        />
        
        <MetricCard
          title="Conversion Rate"
          value={`${metrics?.conversionRate?.toFixed(2) || '0'}%`}
          change={1.8}
          icon={TrendingUp}
          color="orange"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Revenue Chart */}
        <ChartContainer
          title="Revenue Trend"
          subtitle="Monthly revenue over time"
          isLoading={revenueLoading}
          data={revenueChart}
          type="line"
        />

        {/* Users Chart */}
        <ChartContainer
          title="User Growth"
          subtitle="New users per month"
          isLoading={usersLoading}
          data={usersChart}
          type="bar"
        />
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Conversions Chart */}
        <div className="lg:col-span-2">
          <ChartContainer
            title="Conversion Rate"
            subtitle="Conversion rate percentage over time"
            isLoading={conversionsLoading}
            data={conversionsChart}
            type="area"
          />
        </div>

        {/* Realtime Widget */}
        <div className="lg:col-span-1">
          <RealtimeWidget data={realtimeData} />
        </div>
      </div>

      {/* Last Updated */}
      {metrics?.lastUpdated && (
        <div className="text-center text-sm text-gray-500 dark:text-gray-400">
          Last updated: {new Date(metrics.lastUpdated).toLocaleString()}
        </div>
      )}
    </div>
  );
}

export default Dashboard;
