import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Progress, List, Typography, Spin } from 'antd';
import { 
  MessageOutlined, 
  CheckCircleOutlined, 
  ClockCircleOutlined,
  UserOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalFeedback: 0,
    processedToday: 0,
    pendingTasks: 0,
    activeAnnotators: 0,
    averageQuality: 0
  });
  const [recentActivity, setRecentActivity] = useState([]);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch feedback stats
      const feedbackResponse = await axios.get('/api/v1/feedback/stats?hours=24');
      const feedbackStats = feedbackResponse.data;

      // Fetch recent activity (mock data for now)
      const activity = [
        { id: 1, type: 'feedback', message: 'New feedback received for content #123', time: '2 minutes ago' },
        { id: 2, type: 'annotation', message: 'Annotation task completed by user@example.com', time: '5 minutes ago' },
        { id: 3, type: 'review', message: 'Content #456 approved by editor', time: '10 minutes ago' },
        { id: 4, type: 'quality', message: 'Quality score updated for content #789', time: '15 minutes ago' }
      ];

      setStats({
        totalFeedback: feedbackStats.total_feedback || 0,
        processedToday: feedbackStats.by_type?.explicit?.count || 0,
        pendingTasks: 12, // Mock data
        activeAnnotators: 8, // Mock data
        averageQuality: 4.2 // Mock data
      });

      setRecentActivity(activity);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: '16px' }}>Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div>
      <Title level={2}>Dashboard</Title>
      <Text type="secondary">
        <span className="realtime-indicator"></span>
        Real-time feedback processing and quality control
      </Text>

      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Feedback"
              value={stats.totalFeedback}
              prefix={<MessageOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Processed Today"
              value={stats.processedToday}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Pending Tasks"
              value={stats.pendingTasks}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Active Annotators"
              value={stats.activeAnnotators}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col xs={24} lg={12}>
          <Card title="Quality Metrics" extra={<BarChartOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Text strong>Average Quality Score</Text>
              <div style={{ marginTop: '8px' }}>
                <Progress 
                  percent={stats.averageQuality * 20} 
                  format={() => `${stats.averageQuality}/5.0`}
                  strokeColor={{
                    '0%': '#ff4d4f',
                    '50%': '#faad14',
                    '100%': '#52c41a',
                  }}
                />
              </div>
            </div>
            <div>
              <Text strong>Processing Efficiency</Text>
              <div style={{ marginTop: '8px' }}>
                <Progress percent={85} format={() => '85%'} />
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Recent Activity" extra={<ClockCircleOutlined />}>
            <List
              dataSource={recentActivity}
              renderItem={item => (
                <List.Item>
                  <List.Item.Meta
                    title={
                      <Text style={{ fontSize: '14px' }}>
                        {item.message}
                      </Text>
                    }
                    description={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {item.time}
                      </Text>
                    }
                  />
                </List.Item>
              )}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card title="System Status">
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <div className="metrics-card">
                  <div className="metrics-value" style={{ color: '#52c41a' }}>
                    <span className="realtime-indicator"></span>
                    Online
                  </div>
                  <div className="metrics-label">WebSocket Status</div>
                </div>
              </Col>
              <Col xs={24} sm={8}>
                <div className="metrics-card">
                  <div className="metrics-value" style={{ color: '#1890ff' }}>
                    45ms
                  </div>
                  <div className="metrics-label">Avg Response Time</div>
                </div>
              </Col>
              <Col xs={24} sm={8}>
                <div className="metrics-card">
                  <div className="metrics-value" style={{ color: '#722ed1' }}>
                    99.9%
                  </div>
                  <div className="metrics-label">Uptime</div>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
