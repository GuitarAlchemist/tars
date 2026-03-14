import React, { Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import Layout from '@/components/layout/Layout';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import ErrorBoundary from '@/components/ui/ErrorBoundary';
import NotificationContainer from '@/components/ui/NotificationContainer';

// Lazy load components for better performance
const Dashboard = React.lazy(() => import('@/pages/Dashboard'));
const Users = React.lazy(() => import('@/pages/Users'));
const Reports = React.lazy(() => import('@/pages/Reports'));
const Settings = React.lazy(() => import('@/pages/Settings'));
const Login = React.lazy(() => import('@/pages/Login'));
const NotFound = React.lazy(() => import('@/pages/NotFound'));

// Protected Route Component
interface ProtectedRouteProps {
  children: React.ReactNode;
}

function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

// Public Route Component (redirects to dashboard if authenticated)
interface PublicRouteProps {
  children: React.ReactNode;
}

function PublicRoute({ children }: PublicRouteProps) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}

function App() {
  const { theme } = useTheme();

  return (
    <ErrorBoundary>
      <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200 ${theme.mode}`}>
        <Routes>
          {/* Public Routes */}
          <Route
            path="/login"
            element={
              <PublicRoute>
                <Suspense fallback={<LoadingSpinner size="lg" />}>
                  <Login />
                </Suspense>
              </PublicRoute>
            }
          />

          {/* Protected Routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route
              path="dashboard"
              element={
                <Suspense fallback={<LoadingSpinner size="lg" />}>
                  <Dashboard />
                </Suspense>
              }
            />
            <Route
              path="users"
              element={
                <Suspense fallback={<LoadingSpinner size="lg" />}>
                  <Users />
                </Suspense>
              }
            />
            <Route
              path="reports"
              element={
                <Suspense fallback={<LoadingSpinner size="lg" />}>
                  <Reports />
                </Suspense>
              }
            />
            <Route
              path="settings"
              element={
                <Suspense fallback={<LoadingSpinner size="lg" />}>
                  <Settings />
                </Suspense>
              }
            />
          </Route>

          {/* 404 Route */}
          <Route
            path="*"
            element={
              <Suspense fallback={<LoadingSpinner size="lg" />}>
                <NotFound />
              </Suspense>
            }
          />
        </Routes>

        {/* Global Notification Container */}
        <NotificationContainer />
      </div>
    </ErrorBoundary>
  );
}

export default App;
