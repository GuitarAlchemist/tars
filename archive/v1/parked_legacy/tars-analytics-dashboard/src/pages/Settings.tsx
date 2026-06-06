import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { useAuth } from '@/contexts/AuthContext';
import { Settings as SettingsIcon, User, Bell, Shield, Palette } from 'lucide-react';

function Settings() {
  const { theme, toggleMode } = useTheme();
  const { user } = useAuth();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Manage your account settings and preferences.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Profile Settings */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center space-x-2">
                <User className="w-5 h-5 text-gray-500" />
                <h3 className="card-title">Profile Information</h3>
              </div>
            </div>
            <div className="card-content space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  defaultValue={user?.name}
                  className="input w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Email
                </label>
                <input
                  type="email"
                  defaultValue={user?.email}
                  className="input w-full"
                />
              </div>
              <button className="btn btn-primary">
                Save Changes
              </button>
            </div>
          </div>

          {/* Appearance Settings */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center space-x-2">
                <Palette className="w-5 h-5 text-gray-500" />
                <h3 className="card-title">Appearance</h3>
              </div>
            </div>
            <div className="card-content">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    Dark Mode
                  </h4>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Toggle between light and dark themes
                  </p>
                </div>
                <button
                  onClick={toggleMode}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    theme.mode === 'dark' ? 'bg-primary-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      theme.mode === 'dark' ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {/* Quick Actions */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Quick Actions</h3>
            </div>
            <div className="card-content space-y-3">
              <button className="btn btn-outline w-full flex items-center space-x-2">
                <Bell className="w-4 h-4" />
                <span>Notification Settings</span>
              </button>
              <button className="btn btn-outline w-full flex items-center space-x-2">
                <Shield className="w-4 h-4" />
                <span>Security Settings</span>
              </button>
              <button className="btn btn-outline w-full flex items-center space-x-2">
                <SettingsIcon className="w-4 h-4" />
                <span>Advanced Settings</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
