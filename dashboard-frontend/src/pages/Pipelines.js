import React, { useState, useEffect } from 'react';
import { 
  PlusIcon, 
  PlayIcon, 
  PauseIcon, 
  TrashIcon,
  CogIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import api from '../services/api';

export default function Pipelines() {
  const [pipelines, setPipelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedPipeline, setSelectedPipeline] = useState(null);

  useEffect(() => {
    fetchPipelines();
  }, []);

  const fetchPipelines = async () => {
    try {
      const response = await api.get('/dashboard/pipelines');
      setPipelines(response.data);
    } catch (error) {
      console.error('Failed to fetch pipelines:', error);
      toast.error('Failed to load pipelines');
    } finally {
      setLoading(false);
    }
  };

  const handleExecutePipeline = async (pipelineId) => {
    try {
      await api.post(`/api/v1/pipeline/pipelines/${pipelineId}/execute`);
      toast.success('Pipeline execution started');
      fetchPipelines(); // Refresh data
    } catch (error) {
      console.error('Failed to execute pipeline:', error);
      toast.error('Failed to start pipeline execution');
    }
  };

  const handleDeletePipeline = async (pipelineId) => {
    if (window.confirm('Are you sure you want to delete this pipeline?')) {
      try {
        await api.delete(`/api/v1/pipeline/pipelines/${pipelineId}`);
        toast.success('Pipeline deleted successfully');
        fetchPipelines();
      } catch (error) {
        console.error('Failed to delete pipeline:', error);
        toast.error('Failed to delete pipeline');
      }
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'text-success-600 bg-success-100';
      case 'paused':
        return 'text-warning-600 bg-warning-100';
      case 'empty':
        return 'text-gray-600 bg-gray-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner"></div>
        <span className="ml-2 text-gray-600">Loading pipelines...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Pipeline Management
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Create, manage, and monitor your AI/ML pipelines
          </p>
        </div>
        <div className="mt-4 flex md:mt-0 md:ml-4">
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn btn-primary"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            Create Pipeline
          </button>
        </div>
      </div>

      {/* Pipelines Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {pipelines.map((pipeline, index) => (
          <motion.div
            key={pipeline.pipeline_id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card hover:shadow-md transition-shadow duration-200"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <CogIcon className="h-8 w-8 text-primary-600" />
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    {pipeline.name}
                  </h3>
                  <p className="text-sm text-gray-500">
                    {pipeline.components_count} components
                  </p>
                </div>
              </div>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(pipeline.status)}`}>
                {pipeline.status}
              </span>
            </div>

            <div className="mt-4">
              <div className="flex items-center justify-between text-sm text-gray-500">
                <span>Success Rate</span>
                <span className="font-medium">
                  {(pipeline.success_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-2 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${pipeline.success_rate * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="mt-4">
              <p className="text-sm text-gray-500">
                Last execution: {pipeline.last_execution ? 
                  new Date(pipeline.last_execution).toLocaleDateString() : 
                  'Never'
                }
              </p>
            </div>

            <div className="mt-6 flex space-x-3">
              <button
                onClick={() => handleExecutePipeline(pipeline.pipeline_id)}
                className="flex-1 btn btn-primary"
              >
                <PlayIcon className="h-4 w-4 mr-2" />
                Execute
              </button>
              <button
                onClick={() => setSelectedPipeline(pipeline)}
                className="btn btn-secondary"
              >
                <EyeIcon className="h-4 w-4" />
              </button>
              <button
                onClick={() => handleDeletePipeline(pipeline.pipeline_id)}
                className="btn btn-error"
              >
                <TrashIcon className="h-4 w-4" />
              </button>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Empty State */}
      {pipelines.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <CogIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No pipelines</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by creating a new pipeline.
          </p>
          <div className="mt-6">
            <button
              onClick={() => setShowCreateModal(true)}
              className="btn btn-primary"
            >
              <PlusIcon className="h-4 w-4 mr-2" />
              Create Pipeline
            </button>
          </div>
        </motion.div>
      )}

      {/* Create Pipeline Modal */}
      {showCreateModal && (
        <CreatePipelineModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false);
            fetchPipelines();
          }}
        />
      )}

      {/* Pipeline Details Modal */}
      {selectedPipeline && (
        <PipelineDetailsModal
          pipeline={selectedPipeline}
          onClose={() => setSelectedPipeline(null)}
        />
      )}
    </div>
  );
}

// Create Pipeline Modal Component
function CreatePipelineModal({ onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    component_type: 'data_processor',
    config: {}
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Create pipeline
      const pipelineResponse = await api.post('/api/v1/pipeline/pipelines', {
        name: formData.name,
        description: formData.description
      });

      const pipelineId = pipelineResponse.data.pipeline_id;

      // Add initial component
      await api.post(`/api/v1/pipeline/pipelines/${pipelineId}/components`, {
        name: `${formData.name} Component`,
        component_type: formData.component_type,
        config: formData.config
      });

      toast.success('Pipeline created successfully');
      onSuccess();
    } catch (error) {
      console.error('Failed to create pipeline:', error);
      toast.error('Failed to create pipeline');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Create New Pipeline
          </h3>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Pipeline Name
              </label>
              <input
                type="text"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="mt-1 input"
                placeholder="Enter pipeline name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Description
              </label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="mt-1 input"
                rows={3}
                placeholder="Enter pipeline description"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Initial Component Type
              </label>
              <select
                value={formData.component_type}
                onChange={(e) => setFormData({ ...formData, component_type: e.target.value })}
                className="mt-1 input"
              >
                <option value="data_processor">Data Processor</option>
                <option value="model_trainer">Model Trainer</option>
                <option value="evaluator">Evaluator</option>
                <option value="deployer">Deployer</option>
                <option value="monitor">Monitor</option>
              </select>
            </div>
            <div className="flex justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="btn btn-secondary"
                disabled={loading}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading}
              >
                {loading ? 'Creating...' : 'Create Pipeline'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

// Pipeline Details Modal Component
function PipelineDetailsModal({ pipeline, onClose }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPipelineDetails();
  }, [pipeline.pipeline_id]);

  const fetchPipelineDetails = async () => {
    try {
      const response = await api.get(`/api/v1/pipeline/pipelines/${pipeline.pipeline_id}/status`);
      setDetails(response.data);
    } catch (error) {
      console.error('Failed to fetch pipeline details:', error);
      toast.error('Failed to load pipeline details');
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
            Pipeline Details: {pipeline.name}
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
          {/* Components */}
          <div>
            <h4 className="text-md font-medium text-gray-900 mb-3">Components</h4>
            <div className="space-y-2">
              {details?.components?.map((component, index) => (
                <div key={component.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <CogIcon className="h-5 w-5 text-primary-600 mr-3" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">{component.name}</p>
                      <p className="text-xs text-gray-500">{component.component_type}</p>
                    </div>
                  </div>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    component.status === 'completed' ? 'bg-success-100 text-success-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {component.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Executions */}
          <div>
            <h4 className="text-md font-medium text-gray-900 mb-3">Recent Executions</h4>
            <div className="space-y-2">
              {details?.recent_executions?.map((execution) => (
                <div key={execution.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      Execution {execution.id.substring(0, 8)}...
                    </p>
                    <p className="text-xs text-gray-500">
                      {new Date(execution.start_time).toLocaleString()}
                    </p>
                  </div>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    execution.status === 'completed' ? 'bg-success-100 text-success-800' : 
                    execution.status === 'running' ? 'bg-warning-100 text-warning-800' : 
                    'bg-error-100 text-error-800'
                  }`}>
                    {execution.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
