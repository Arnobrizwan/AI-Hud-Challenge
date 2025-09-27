import React, { useState, useEffect } from 'react';
import { 
  ChartBarIcon, 
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell,
  AreaChart,
  Area
} from 'recharts';
import { format } from 'date-fns';
import api from '../services/api';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

export default function Analytics() {
  const [overview, setOverview] = useState(null);
  const [trends, setTrends] = useState(null);
  const [pipelines, setPipelines] = useState([]);
  const [executions, setExecutions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7');

  useEffect(() => {
    fetchAnalyticsData();
  }, [timeRange]);

  const fetchAnalyticsData = async () => {
    try {
      const [overviewData, trendsData, pipelinesData, executionsData] = await Promise.all([
        api.get('/dashboard/overview'),
        api.get(`/dashboard/metrics/trends?days=${timeRange}`),
        api.get('/dashboard/pipelines'),
        api.get('/dashboard/executions?limit=100')
      ]);

      setOverview(overviewData.data);
      setTrends(trendsData.data);
      setPipelines(pipelinesData.data);
      setExecutions(executionsData.data);
    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSuccessRateData = () => {
    if (!trends?.success_rate) return [];
    
    return trends.success_rate.map((rate, index) => ({
      date: trends.dates[index],
      successRate: rate * 100
    }));
  };

  const getExecutionCountData = () => {
    if (!trends?.execution_count) return [];
    
    return trends.execution_count.map((count, index) => ({
      date: trends.dates[index],
      executions: count
    }));
  };

  const getPipelinePerformanceData = () => {
    return pipelines.map(pipeline => ({
      name: pipeline.name,
      successRate: pipeline.success_rate * 100,
      components: pipeline.components_count
    }));
  };

  const getExecutionStatusData = () => {
    const statusCounts = executions.reduce((acc, execution) => {
      acc[execution.status] = (acc[execution.status] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(statusCounts).map(([status, count]) => ({
      name: status.charAt(0).toUpperCase() + status.slice(1),
      value: count,
      color: status === 'completed' ? '#10b981' : 
             status === 'running' ? '#f59e0b' : 
             status === 'failed' ? '#ef4444' : '#6b7280'
    }));
  };

  const getAverageExecutionTime = () => {
    const completedExecutions = executions.filter(exec => exec.status === 'completed' && exec.duration);
    if (completedExecutions.length === 0) return 0;
    
    const totalTime = completedExecutions.reduce((sum, exec) => sum + exec.duration, 0);
    return totalTime / completedExecutions.length;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner"></div>
        <span className="ml-2 text-gray-600">Loading analytics...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Analytics & Insights
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Comprehensive analysis of your AI/ML pipeline performance
          </p>
        </div>
        <div className="mt-4 flex md:mt-0 md:ml-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="input"
          >
            <option value="7">Last 7 days</option>
            <option value="14">Last 14 days</option>
            <option value="30">Last 30 days</option>
          </select>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CheckCircleIcon className="h-8 w-8 text-success-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Overall Success Rate
                </dt>
                <dd className="text-lg font-medium text-gray-900">
                  {overview?.success_rate ? `${(overview.success_rate * 100).toFixed(1)}%` : '0%'}
                </dd>
              </dl>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ClockIcon className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Avg Execution Time
                </dt>
                <dd className="text-lg font-medium text-gray-900">
                  {getAverageExecutionTime().toFixed(1)}s
                </dd>
              </dl>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-warning-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Total Executions
                </dt>
                <dd className="text-lg font-medium text-gray-900">
                  {executions.length}
                </dd>
              </dl>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ExclamationTriangleIcon className="h-8 w-8 text-error-600" />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Failed Executions
                </dt>
                <dd className="text-lg font-medium text-gray-900">
                  {executions.filter(exec => exec.status === 'failed').length}
                </dd>
              </dl>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Success Rate Trend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Success Rate Trend</h3>
            <p className="mt-1 text-sm text-gray-500">Pipeline success rate over time</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={getSuccessRateData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => format(new Date(value), 'MMM d')}
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value) => [`${value.toFixed(1)}%`, 'Success Rate']}
                  labelFormatter={(value) => format(new Date(value), 'MMM d, yyyy')}
                />
                <Area 
                  type="monotone" 
                  dataKey="successRate" 
                  stroke="#10b981" 
                  fill="#10b981"
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Execution Count Trend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Execution Volume</h3>
            <p className="mt-1 text-sm text-gray-500">Number of executions over time</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getExecutionCountData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => format(new Date(value), 'MMM d')}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value) => [value, 'Executions']}
                  labelFormatter={(value) => format(new Date(value), 'MMM d, yyyy')}
                />
                <Bar dataKey="executions" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pipeline Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Pipeline Performance</h3>
            <p className="mt-1 text-sm text-gray-500">Success rate by pipeline</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getPipelinePerformanceData()} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={100} />
                <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Success Rate']} />
                <Bar dataKey="successRate" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Execution Status Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Execution Status</h3>
            <p className="mt-1 text-sm text-gray-500">Distribution of execution statuses</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={getExecutionStatusData()}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {getExecutionStatusData().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Performance Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="card"
      >
        <div className="card-header">
          <h3 className="text-lg font-medium text-gray-900">Performance Insights</h3>
          <p className="mt-1 text-sm text-gray-500">Key insights and recommendations</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="p-4 bg-success-50 rounded-lg">
            <h4 className="text-sm font-medium text-success-800">Best Performing Pipeline</h4>
            <p className="mt-1 text-sm text-success-700">
              {pipelines.length > 0 ? 
                pipelines.reduce((best, current) => 
                  current.success_rate > best.success_rate ? current : best
                ).name : 'N/A'
              }
            </p>
          </div>
          <div className="p-4 bg-warning-50 rounded-lg">
            <h4 className="text-sm font-medium text-warning-800">Average Components per Pipeline</h4>
            <p className="mt-1 text-sm text-warning-700">
              {pipelines.length > 0 ? 
                (pipelines.reduce((sum, p) => sum + p.components_count, 0) / pipelines.length).toFixed(1) : '0'
              } components
            </p>
          </div>
          <div className="p-4 bg-primary-50 rounded-lg">
            <h4 className="text-sm font-medium text-primary-800">System Uptime</h4>
            <p className="mt-1 text-sm text-primary-700">
              {overview?.system_health === 'healthy' ? '99.9%' : '99.5%'}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
