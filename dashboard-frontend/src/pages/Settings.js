import React, { useState, useEffect } from 'react';
import { 
  CogIcon, 
  BellIcon,
  ShieldCheckIcon,
  ServerIcon,
  UserIcon,
  KeyIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';

export default function Settings() {
  const [activeTab, setActiveTab] = useState('general');
  const [settings, setSettings] = useState({
    general: {
      pipelineAutoStart: true,
      maxConcurrentExecutions: 5,
      executionTimeout: 3600,
      logLevel: 'info'
    },
    notifications: {
      emailNotifications: true,
      slackNotifications: false,
      webhookUrl: '',
      alertThreshold: 0.8
    },
    security: {
      requireAuth: true,
      sessionTimeout: 1800,
      apiKeyRotation: 30,
      auditLogging: true
    },
    system: {
      dataRetention: 90,
      backupFrequency: 'daily',
      monitoringInterval: 60,
      healthCheckTimeout: 30
    }
  });

  const tabs = [
    { id: 'general', name: 'General', icon: CogIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon },
    { id: 'security', name: 'Security', icon: ShieldCheckIcon },
    { id: 'system', name: 'System', icon: ServerIcon }
  ];

  const handleSettingChange = (category, key, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value
      }
    }));
  };

  const handleSaveSettings = () => {
    // In a real app, this would save to the backend
    toast.success('Settings saved successfully');
  };

  const handleResetSettings = () => {
    if (window.confirm('Are you sure you want to reset all settings to default?')) {
      // Reset to default values
      toast.success('Settings reset to default');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Settings
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Configure your AI/ML pipeline management system
          </p>
        </div>
        <div className="mt-4 flex space-x-3 md:mt-0 md:ml-4">
          <button
            onClick={handleResetSettings}
            className="btn btn-secondary"
          >
            Reset
          </button>
          <button
            onClick={handleSaveSettings}
            className="btn btn-primary"
          >
            Save Changes
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeTab === tab.id
                    ? 'bg-primary-100 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <tab.icon className="h-5 w-5 mr-3" />
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2 }}
            className="card"
          >
            {activeTab === 'general' && <GeneralSettings settings={settings.general} onChange={(key, value) => handleSettingChange('general', key, value)} />}
            {activeTab === 'notifications' && <NotificationSettings settings={settings.notifications} onChange={(key, value) => handleSettingChange('notifications', key, value)} />}
            {activeTab === 'security' && <SecuritySettings settings={settings.security} onChange={(key, value) => handleSettingChange('security', key, value)} />}
            {activeTab === 'system' && <SystemSettings settings={settings.system} onChange={(key, value) => handleSettingChange('system', key, value)} />}
          </motion.div>
        </div>
      </div>
    </div>
  );
}

// General Settings Component
function GeneralSettings({ settings, onChange }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">General Settings</h3>
        <p className="mt-1 text-sm text-gray-500">
          Configure basic pipeline behavior and preferences
        </p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Auto-start Pipelines
            </label>
            <p className="text-sm text-gray-500">
              Automatically start pipelines when they are created
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.pipelineAutoStart}
              onChange={(e) => onChange('pipelineAutoStart', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Max Concurrent Executions
          </label>
          <p className="text-sm text-gray-500">
            Maximum number of pipelines that can run simultaneously
          </p>
          <input
            type="number"
            min="1"
            max="20"
            value={settings.maxConcurrentExecutions}
            onChange={(e) => onChange('maxConcurrentExecutions', parseInt(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Execution Timeout (seconds)
          </label>
          <p className="text-sm text-gray-500">
            Maximum time a pipeline execution can run before being terminated
          </p>
          <input
            type="number"
            min="60"
            max="86400"
            value={settings.executionTimeout}
            onChange={(e) => onChange('executionTimeout', parseInt(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Log Level
          </label>
          <p className="text-sm text-gray-500">
            Level of detail in system logs
          </p>
          <select
            value={settings.logLevel}
            onChange={(e) => onChange('logLevel', e.target.value)}
            className="mt-1 input w-48"
          >
            <option value="debug">Debug</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>
    </div>
  );
}

// Notification Settings Component
function NotificationSettings({ settings, onChange }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Notification Settings</h3>
        <p className="mt-1 text-sm text-gray-500">
          Configure how you receive alerts and updates
        </p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Email Notifications
            </label>
            <p className="text-sm text-gray-500">
              Receive notifications via email
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.emailNotifications}
              onChange={(e) => onChange('emailNotifications', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Slack Notifications
            </label>
            <p className="text-sm text-gray-500">
              Send notifications to Slack channel
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.slackNotifications}
              onChange={(e) => onChange('slackNotifications', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Webhook URL
          </label>
          <p className="text-sm text-gray-500">
            Custom webhook endpoint for notifications
          </p>
          <input
            type="url"
            value={settings.webhookUrl}
            onChange={(e) => onChange('webhookUrl', e.target.value)}
            className="mt-1 input"
            placeholder="https://hooks.slack.com/services/..."
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Alert Threshold
          </label>
          <p className="text-sm text-gray-500">
            Success rate below which alerts are triggered
          </p>
          <input
            type="number"
            min="0"
            max="1"
            step="0.1"
            value={settings.alertThreshold}
            onChange={(e) => onChange('alertThreshold', parseFloat(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>
      </div>
    </div>
  );
}

// Security Settings Component
function SecuritySettings({ settings, onChange }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Security Settings</h3>
        <p className="mt-1 text-sm text-gray-500">
          Configure security and access control settings
        </p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Require Authentication
            </label>
            <p className="text-sm text-gray-500">
              Require users to authenticate before accessing the system
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.requireAuth}
              onChange={(e) => onChange('requireAuth', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Session Timeout (seconds)
          </label>
          <p className="text-sm text-gray-500">
            How long before users are automatically logged out
          </p>
          <input
            type="number"
            min="300"
            max="86400"
            value={settings.sessionTimeout}
            onChange={(e) => onChange('sessionTimeout', parseInt(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            API Key Rotation (days)
          </label>
          <p className="text-sm text-gray-500">
            How often API keys are automatically rotated
          </p>
          <input
            type="number"
            min="1"
            max="365"
            value={settings.apiKeyRotation}
            onChange={(e) => onChange('apiKeyRotation', parseInt(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>

        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Audit Logging
            </label>
            <p className="text-sm text-gray-500">
              Log all user actions and system events
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.auditLogging}
              onChange={(e) => onChange('auditLogging', e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>
      </div>
    </div>
  );
}

// System Settings Component
function SystemSettings({ settings, onChange }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">System Settings</h3>
        <p className="mt-1 text-sm text-gray-500">
          Configure system behavior and maintenance settings
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Data Retention (days)
          </label>
          <p className="text-sm text-gray-500">
            How long to keep execution logs and metrics
          </p>
          <input
            type="number"
            min="1"
            max="3650"
            value={settings.dataRetention}
            onChange={(e) => onChange('dataRetention', parseInt(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Backup Frequency
          </label>
          <p className="text-sm text-gray-500">
            How often to create system backups
          </p>
          <select
            value={settings.backupFrequency}
            onChange={(e) => onChange('backupFrequency', e.target.value)}
            className="mt-1 input w-48"
          >
            <option value="hourly">Hourly</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Monitoring Interval (seconds)
          </label>
          <p className="text-sm text-gray-500">
            How often to check system health and performance
          </p>
          <input
            type="number"
            min="10"
            max="3600"
            value={settings.monitoringInterval}
            onChange={(e) => onChange('monitoringInterval', parseInt(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Health Check Timeout (seconds)
          </label>
          <p className="text-sm text-gray-500">
            Maximum time to wait for health check responses
          </p>
          <input
            type="number"
            min="5"
            max="300"
            value={settings.healthCheckTimeout}
            onChange={(e) => onChange('healthCheckTimeout', parseInt(e.target.value))}
            className="mt-1 input w-32"
          />
        </div>
      </div>
    </div>
  );
}
