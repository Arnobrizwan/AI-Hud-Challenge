import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Table, 
  Tag, 
  Typography, 
  Select,
  DatePicker,
  Space,
  Button,
  Tooltip
} from 'antd';
import { 
  BarChartOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  WarningOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;

const QualityDashboard = () => {
  const [loading, setLoading] = useState(false);
  const [qualityMetrics, setQualityMetrics] = useState({
    averageScore: 4.2,
    totalAssessments: 1250,
    highQuality: 850,
    mediumQuality: 300,
    lowQuality: 100,
    needsReview: 45
  });
  const [qualityTrends, setQualityTrends] = useState([]);
  const [recentAssessments, setRecentAssessments] = useState([]);

  useEffect(() => {
    fetchQualityData();
  }, []);

  const fetchQualityData = async () => {
    try {
      setLoading(true);
      
      // Mock quality trends data
      const trendsData = [
        { date: '2024-01-01', score: 4.1, assessments: 120 },
        { date: '2024-01-02', score: 4.2, assessments: 135 },
        { date: '2024-01-03', score: 4.0, assessments: 110 },
        { date: '2024-01-04', score: 4.3, assessments: 145 },
        { date: '2024-01-05', score: 4.2, assessments: 130 },
        { date: '2024-01-06', score: 4.4, assessments: 155 },
        { date: '2024-01-07', score: 4.2, assessments: 140 }
      ];
      setQualityTrends(trendsData);

      // Mock recent assessments
      const assessments = [
        {
          id: 'qa-001',
          content_id: 'content-123',
          overall_score: 4.5,
          factual_accuracy: 4.8,
          readability: 4.2,
          bias_score: 0.1,
          needs_review: false,
          created_at: '2024-01-07T10:30:00Z'
        },
        {
          id: 'qa-002',
          content_id: 'content-124',
          overall_score: 3.2,
          factual_accuracy: 3.0,
          readability: 3.5,
          bias_score: 0.3,
          needs_review: true,
          created_at: '2024-01-07T09:15:00Z'
        },
        {
          id: 'qa-003',
          content_id: 'content-125',
          overall_score: 4.8,
          factual_accuracy: 4.9,
          readability: 4.7,
          bias_score: 0.05,
          needs_review: false,
          created_at: '2024-01-07T08:45:00Z'
        }
      ];
      setRecentAssessments(assessments);
      
    } catch (error) {
      console.error('Error fetching quality data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getQualityColor = (score) => {
    if (score >= 4.0) return '#52c41a';
    if (score >= 3.0) return '#faad14';
    return '#ff4d4f';
  };

  const getQualityLevel = (score) => {
    if (score >= 4.0) return 'High';
    if (score >= 3.0) return 'Medium';
    return 'Low';
  };

  const assessmentColumns = [
    {
      title: 'Assessment ID',
      dataIndex: 'id',
      key: 'id',
      width: 100,
    },
    {
      title: 'Content ID',
      dataIndex: 'content_id',
      key: 'content_id',
      width: 120,
    },
    {
      title: 'Overall Score',
      dataIndex: 'overall_score',
      key: 'overall_score',
      width: 120,
      render: (score) => (
        <Tag color={getQualityColor(score)}>
          {score}/5.0
        </Tag>
      )
    },
    {
      title: 'Factual Accuracy',
      dataIndex: 'factual_accuracy',
      key: 'factual_accuracy',
      width: 120,
      render: (score) => (
        <Progress 
          percent={score * 20} 
          size="small" 
          strokeColor={getQualityColor(score)}
          format={() => `${score}/5.0`}
        />
      )
    },
    {
      title: 'Readability',
      dataIndex: 'readability',
      key: 'readability',
      width: 120,
      render: (score) => (
        <Progress 
          percent={score * 20} 
          size="small" 
          strokeColor={getQualityColor(score)}
          format={() => `${score}/5.0`}
        />
      )
    },
    {
      title: 'Bias Score',
      dataIndex: 'bias_score',
      key: 'bias_score',
      width: 100,
      render: (score) => (
        <Tag color={score < 0.2 ? 'green' : score < 0.5 ? 'orange' : 'red'}>
          {score}
        </Tag>
      )
    },
    {
      title: 'Status',
      dataIndex: 'needs_review',
      key: 'needs_review',
      width: 100,
      render: (needsReview) => (
        <Tag color={needsReview ? 'red' : 'green'}>
          {needsReview ? 'Review Needed' : 'Approved'}
        </Tag>
      )
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (date) => new Date(date).toLocaleString()
    }
  ];

  return (
    <div>
      <Title level={2}>Quality Dashboard</Title>
      
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Average Quality Score"
              value={qualityMetrics.averageScore}
              precision={1}
              suffix="/5.0"
              valueStyle={{ color: getQualityColor(qualityMetrics.averageScore) }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Total Assessments"
              value={qualityMetrics.totalAssessments}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="High Quality"
              value={qualityMetrics.highQuality}
              valueStyle={{ color: '#52c41a' }}
              suffix={`(${Math.round(qualityMetrics.highQuality / qualityMetrics.totalAssessments * 100)}%)`}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Needs Review"
              value={qualityMetrics.needsReview}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={12}>
          <Card title="Quality Distribution">
            <div style={{ marginBottom: '16px' }}>
              <Text strong>High Quality (≥4.0)</Text>
              <Progress 
                percent={Math.round(qualityMetrics.highQuality / qualityMetrics.totalAssessments * 100)} 
                strokeColor="#52c41a"
                style={{ marginTop: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <Text strong>Medium Quality (3.0-3.9)</Text>
              <Progress 
                percent={Math.round(qualityMetrics.mediumQuality / qualityMetrics.totalAssessments * 100)} 
                strokeColor="#faad14"
                style={{ marginTop: '4px' }}
              />
            </div>
            <div>
              <Text strong>Low Quality (<3.0)</Text>
              <Progress 
                percent={Math.round(qualityMetrics.lowQuality / qualityMetrics.totalAssessments * 100)} 
                strokeColor="#ff4d4f"
                style={{ marginTop: '4px' }}
              />
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Quality Trends (Last 7 Days)">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={qualityTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={[0, 5]} />
                <RechartsTooltip />
                <Line 
                  type="monotone" 
                  dataKey="score" 
                  stroke="#1890ff" 
                  strokeWidth={2}
                  dot={{ fill: '#1890ff' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={24}>
          <Card 
            title="Assessment Volume" 
            extra={
              <Space>
                <RangePicker />
                <Button icon={<ReloadOutlined />} onClick={fetchQualityData}>
                  Refresh
                </Button>
              </Space>
            }
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={qualityTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="assessments" fill="#1890ff" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      <Card 
        title="Recent Quality Assessments" 
        extra={<Button onClick={fetchQualityData}>Refresh</Button>}
      >
        <Table
          columns={assessmentColumns}
          dataSource={recentAssessments}
          loading={loading}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true
          }}
        />
      </Card>
    </div>
  );
};

export default QualityDashboard;
