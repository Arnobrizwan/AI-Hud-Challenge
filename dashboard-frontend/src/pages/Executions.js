import React, { useState, useEffect } from 'react';
import { 
  PlayIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon,
  ClockIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { toast } from 'react-hot-toast';
import api from '../services/api';

export default function Executions() {
  const [executions, setExecutions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedExecution, setSelectedExecution] = useState(null);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    fetchExecutions();
    const interval = setInterval(fetchExecutions, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchExecutions = async () => {
    try {
      const response = await api.get('/dashboard/executions?limit=50');
      setExecutions(response.data);
    } catch (error) {
      console.error('Failed to fetch executions:', error);
      toast.error('Failed to load executions');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-success-600 bg-success-100';
      case 'running':
        return 'text-warning-600 bg-warning-100';
      case 'failed':
        return 'text-error-600 bg-error-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5" />;
      case 'running':
        return <ClockIcon className="h-5 w-5" />;
      case 'failed':
        return <ExclamationTriangleIcon className="h-5 w-5" />;
      default:
        return <PlayIcon className="h-5 w-5" />;
    }
  };

  const filteredExecutions = executions.filter(execution => {
    if (filter === 'all') return true;
    return execution.status === filter;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner"></div>
        <span className="ml-2 text-gray-600">Loading executions...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Execution History
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Monitor and analyze pipeline execution results
          </p>
        </div>
        <div className="mt-4 flex md:mt-0 md:ml-4">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="input"
          >
            <option value="all">All Executions</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Executions Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Execution ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pipeline
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Start Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Duration
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Progress
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredExecutions.map((execution, index) => (
                <motion.tr
                  key={execution.execution_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="hover:bg-gray-50"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                    {execution.execution_id.substring(0, 12)}...
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {execution.pipeline_id.substring(0, 12)}...
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(execution.status)}`}>
                      {getStatusIcon(execution.status)}
                      <span className="ml-1 capitalize">{execution.status}</span>
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {format(new Date(execution.start_time), 'MMM d, HH:mm:ss')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {execution.duration ? `${execution.duration.toFixed(1)}s` : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-300 ${
                            execution.status === 'completed' ? 'bg-success-500' :
                            execution.status === 'running' ? 'bg-warning-500' :
                            execution.status === 'failed' ? 'bg-error-500' : 'bg-gray-400'
                          }`}
                          style={{ 
                            width: `${(execution.components_completed / execution.total_components) * 100}%` 
                          }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-500">
                        {execution.components_completed}/{execution.total_components}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button
                      onClick={() => setSelectedExecution(execution)}
                      className="text-primary-600 hover:text-primary-900"
                    >
                      <EyeIcon className="h-4 w-4" />
                    </button>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Empty State */}
        {filteredExecutions.length === 0 && (
          <div className="text-center py-12">
            <PlayIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No executions found</h3>
            <p className="mt-1 text-sm text-gray-500">
              {filter === 'all' ? 'No executions have been run yet.' : `No ${filter} executions found.`}
            </p>
          </div>
        )}
      </motion.div>

      {/* Execution Details Modal */}
      {selectedExecution && (
        <ExecutionDetailsModal
          execution={selectedExecution}
          onClose={() => setSelectedExecution(null)}
        />
      )}
    </div>
  );
}

// Execution Details Modal Component
function ExecutionDetailsModal({ execution, onClose }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExecutionDetails();
  }, [execution.execution_id]);

  const fetchExecutionDetails = async () => {
    try {
      const response = await api.get(`/api/v1/pipeline/executions/${execution.execution_id}`);
      setDetails(response.data);
    } catch (error) {
      console.error('Failed to fetch execution details:', error);
      toast.error('Failed to load execution details');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
        <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
          <div className="flex items-center justify-center h-32">
            <div className="loading-spinner"></div>
            <span className="ml-2 text-gray-600">Loading details...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-10 mx-auto p-5 border w-11/12 max-w-4xl shadow-lg rounded-md bg-white">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium text-gray-900">
            Execution Details: {execution.execution_id.substring(0, 12)}...
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <span className="sr-only">Close</span>
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="space-y-6">
          {/* Execution Info */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-gray-500">Status</h4>
              <p className="mt-1 text-sm text-gray-900 capitalize">{execution.status}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500">Start Time</h4>
              <p className="mt-1 text-sm text-gray-900">
                {format(new Date(execution.start_time), 'MMM d, yyyy HH:mm:ss')}
              </p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500">Duration</h4>
              <p className="mt-1 text-sm text-gray-900">
                {execution.duration ? `${execution.duration.toFixed(1)}s` : 'N/A'}
              </p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500">Progress</h4>
              <p className="mt-1 text-sm text-gray-900">
                {execution.components_completed}/{execution.total_components} components
              </p>
            </div>
          </div>

          {/* Metrics */}
          {details?.metrics && Object.keys(details.metrics).length > 0 && (
            <div>
              <h4 className="text-md font-medium text-gray-900 mb-3">Execution Metrics</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(details.metrics).map(([key, value]) => (
                  <div key={key} className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm font-medium text-gray-900 capitalize">
                      {key.replace(/_/g, ' ')}
                    </p>
                    <p className="text-lg text-gray-600">
                      {typeof value === 'number' ? value.toFixed(2) : String(value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error Message */}
          {execution.status === 'failed' && details?.error_message && (
            <div>
              <h4 className="text-md font-medium text-gray-900 mb-3">Error Details</h4>
              <div className="p-3 bg-error-50 border border-error-200 rounded-lg">
                <p className="text-sm text-error-800">{details.error_message}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
