import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Table, 
  Button, 
  Tag, 
  Space, 
  Typography, 
  Modal, 
  Form, 
  Input, 
  Select,
  message,
  Badge,
  Tooltip
} from 'antd';
import { 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  EditOutlined,
  EyeOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

const EditorialDashboard = () => {
  const [loading, setLoading] = useState(false);
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState(null);
  const [reviewModalVisible, setReviewModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchTasks();
  }, []);

  const fetchTasks = async () => {
    try {
      setLoading(true);
      // Mock data - in real app, this would come from API
      const mockTasks = [
        {
          id: 'task-001',
          content_id: 'content-123',
          task_type: 'content_review',
          priority: 'high',
          assigned_to: 'editor1',
          status: 'assigned',
          due_date: '2024-01-15T23:59:59Z',
          created_at: '2024-01-14T10:00:00Z',
          content_preview: 'This is a sample content that needs editorial review...'
        },
        {
          id: 'task-002',
          content_id: 'content-124',
          task_type: 'fact_check',
          priority: 'normal',
          assigned_to: 'editor2',
          status: 'in_progress',
          due_date: '2024-01-16T23:59:59Z',
          created_at: '2024-01-14T11:00:00Z',
          content_preview: 'Another piece of content requiring fact checking...'
        },
        {
          id: 'task-003',
          content_id: 'content-125',
          task_type: 'quality_review',
          priority: 'low',
          assigned_to: 'editor1',
          status: 'completed',
          due_date: '2024-01-13T23:59:59Z',
          created_at: '2024-01-13T09:00:00Z',
          content_preview: 'Content that has been reviewed and approved...'
        }
      ];
      setTasks(mockTasks);
    } catch (error) {
      console.error('Error fetching tasks:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReview = (task) => {
    setSelectedTask(task);
    setReviewModalVisible(true);
    form.resetFields();
  };

  const handleReviewSubmit = async (values) => {
    try {
      setLoading(true);
      
      const reviewData = {
        task_id: selectedTask.id,
        reviewer_id: 'current-user',
        decision: values.decision,
        comments: values.comments,
        changes_requested: values.changes_requested
      };

      // Submit review
      await axios.post(`/api/v1/editorial/tasks/${selectedTask.id}/complete`, reviewData);
      
      message.success('Review submitted successfully');
      setReviewModalVisible(false);
      fetchTasks();
      
    } catch (error) {
      console.error('Error submitting review:', error);
      message.error('Failed to submit review');
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return 'red';
      case 'normal': return 'blue';
      case 'low': return 'green';
      default: return 'default';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending': return 'default';
      case 'assigned': return 'blue';
      case 'in_progress': return 'orange';
      case 'completed': return 'green';
      case 'overdue': return 'red';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: 'Task ID',
      dataIndex: 'id',
      key: 'id',
      width: 100,
    },
    {
      title: 'Type',
      dataIndex: 'task_type',
      key: 'task_type',
      width: 120,
      render: (type) => <Tag color="blue">{type.replace('_', ' ').toUpperCase()}</Tag>
    },
    {
      title: 'Priority',
      dataIndex: 'priority',
      key: 'priority',
      width: 100,
      render: (priority) => (
        <Tag color={getPriorityColor(priority)}>
          {priority.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {status.replace('_', ' ').toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Content Preview',
      dataIndex: 'content_preview',
      key: 'content_preview',
      ellipsis: true,
      render: (text) => (
        <Tooltip title={text}>
          {text}
        </Tooltip>
      )
    },
    {
      title: 'Due Date',
      dataIndex: 'due_date',
      key: 'due_date',
      width: 150,
      render: (date) => new Date(date).toLocaleDateString()
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleReview(record)}
          >
            Review
          </Button>
        </Space>
      )
    }
  ];

  const pendingTasks = tasks.filter(task => task.status === 'pending' || task.status === 'assigned');
  const overdueTasks = tasks.filter(task => 
    new Date(task.due_date) < new Date() && 
    (task.status === 'assigned' || task.status === 'in_progress')
  );

  return (
    <div>
      <Title level={2}>Editorial Dashboard</Title>
      
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Badge count={pendingTasks.length} showZero>
                <ClockCircleOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
              </Badge>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Pending Tasks</Text>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Badge count={overdueTasks.length} showZero>
                <ExclamationCircleOutlined style={{ fontSize: '24px', color: '#ff4d4f' }} />
              </Badge>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Overdue Tasks</Text>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <CheckCircleOutlined style={{ fontSize: '24px', color: '#52c41a' }} />
              <div style={{ marginTop: '8px' }}>
                <Text strong>Completed Today</Text>
                <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#52c41a' }}>
                  {tasks.filter(task => task.status === 'completed').length}
                </div>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Card title="Review Tasks" extra={<Button onClick={fetchTasks}>Refresh</Button>}>
        <Table
          columns={columns}
          dataSource={tasks}
          loading={loading}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true
          }}
        />
      </Card>

      <Modal
        title={`Review Task: ${selectedTask?.id}`}
        open={reviewModalVisible}
        onCancel={() => setReviewModalVisible(false)}
        footer={null}
        width={600}
      >
        {selectedTask && (
          <div>
            <Card size="small" style={{ marginBottom: '16px' }}>
              <Text strong>Content Preview:</Text>
              <div style={{ marginTop: '8px', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                {selectedTask.content_preview}
              </div>
            </Card>

            <Form
              form={form}
              layout="vertical"
              onFinish={handleReviewSubmit}
            >
              <Form.Item
                name="decision"
                label="Review Decision"
                rules={[{ required: true, message: 'Please select a decision' }]}
              >
                <Select placeholder="Select decision">
                  <Option value="approve">
                    <CheckCircleOutlined style={{ color: '#52c41a' }} /> Approve
                  </Option>
                  <Option value="reject">
                    <CloseCircleOutlined style={{ color: '#ff4d4f' }} /> Reject
                  </Option>
                  <Option value="request_changes">
                    <EditOutlined style={{ color: '#faad14' }} /> Request Changes
                  </Option>
                  <Option value="escalate">
                    <ExclamationCircleOutlined style={{ color: '#722ed1' }} /> Escalate
                  </Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="comments"
                label="Review Comments"
                rules={[{ required: true, message: 'Please provide comments' }]}
              >
                <TextArea 
                  rows={4} 
                  placeholder="Provide detailed comments about your review decision..."
                />
              </Form.Item>

              <Form.Item
                name="changes_requested"
                label="Changes Requested (if applicable)"
              >
                <TextArea 
                  rows={3} 
                  placeholder="Specify what changes are needed..."
                />
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button type="primary" htmlType="submit" loading={loading}>
                    Submit Review
                  </Button>
                  <Button onClick={() => setReviewModalVisible(false)}>
                    Cancel
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default EditorialDashboard;
