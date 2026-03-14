import React from 'react';
import { Link } from 'react-router-dom';
import { Home, ArrowLeft } from 'lucide-react';

function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="max-w-md w-full mx-4 text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-primary-600 dark:text-primary-400">
            404
          </h1>
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mt-4">
            Page Not Found
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            The page you're looking for doesn't exist or has been moved.
          </p>
        </div>

        <div className="space-y-4">
          <Link
            to="/dashboard"
            className="btn btn-primary w-full flex items-center justify-center space-x-2"
          >
            <Home className="w-4 h-4" />
            <span>Go to Dashboard</span>
          </Link>
          
          <button
            onClick={() => window.history.back()}
            className="btn btn-outline w-full flex items-center justify-center space-x-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Go Back</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default NotFound;
