import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Button, 
  Form, 
  Input, 
  Select, 
  Radio, 
  Slider, 
  Typography, 
  Space, 
  Divider,
  Alert,
  Progress,
  List,
  Tag,
  Spin
} from 'antd';
import { 
  SendOutlined, 
  ClockCircleOutlined, 
  CheckCircleOutlined,
  InfoCircleOutlined 
} from '@ant-design/icons';
import axios from 'axios';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Option } = Select;

const AnnotationInterface = () => {
  const [loading, setLoading] = useState(false);
  const [currentTask, setCurrentTask] = useState(null);
  const [form] = Form.useForm();
  const [annotationType, setAnnotationType] = useState('sentiment');
  const [confidence, setConfidence] = useState(0.8);
  const [timeSpent, setTimeSpent] = useState(0);

  useEffect(() => {
    fetchCurrentTask();
    startTimer();
  }, []);

  const fetchCurrentTask = async () => {
    try {
      setLoading(true);
      // Mock task data - in real app, this would come from API
      const mockTask = {
        id: 'task-123',
        content: 'This is a sample piece of content that needs to be annotated. It contains various elements that require careful analysis and classification.',
        annotation_type: 'sentiment',
        guidelines: 'Please analyze the sentiment of this content and provide your assessment.',
        deadline: '2024-01-15T23:59:59Z',
        priority: 'normal'
      };
      setCurrentTask(mockTask);
    } catch (error) {
      console.error('Error fetching task:', error);
    } finally {
      setLoading(false);
    }
  };

  const startTimer = () => {
    const timer = setInterval(() => {
      setTimeSpent(prev => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  };

  const handleSubmit = async (values) => {
    try {
      setLoading(true);
      
      const annotationData = {
        task_id: currentTask.id,
        annotator_id: 'current-user',
        annotation_data: {
          ...values,
          confidence: confidence,
          time_spent_seconds: timeSpent
        }
      };

      // Submit annotation
      await axios.post('/api/v1/annotation/tasks/submit', annotationData);
      
      // Show success message and fetch next task
      fetchCurrentTask();
      form.resetFields();
      setTimeSpent(0);
      
    } catch (error) {
      console.error('Error submitting annotation:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderAnnotationForm = () => {
    switch (annotationType) {
      case 'sentiment':
        return (
          <Form.Item
            name="sentiment"
            label="Sentiment Analysis"
            rules={[{ required: true, message: 'Please select sentiment' }]}
          >
            <Radio.Group>
              <Radio value="positive">Positive</Radio>
              <Radio value="negative">Negative</Radio>
              <Radio value="neutral">Neutral</Radio>
            </Radio.Group>
          </Form.Item>
        );
      
      case 'quality':
        return (
          <Form.Item
            name="quality_score"
            label="Quality Score (1-5)"
            rules={[{ required: true, message: 'Please rate quality' }]}
          >
            <Slider
              min={1}
              max={5}
              step={0.1}
              marks={{
                1: 'Poor',
                2: 'Fair',
                3: 'Good',
                4: 'Very Good',
                5: 'Excellent'
              }}
            />
          </Form.Item>
        );
      
      case 'topic':
        return (
          <Form.Item
            name="topic"
            label="Topic Classification"
            rules={[{ required: true, message: 'Please select topic' }]}
          >
            <Select placeholder="Select topic">
              <Option value="technology">Technology</Option>
              <Option value="business">Business</Option>
              <Option value="health">Health</Option>
              <Option value="education">Education</Option>
              <Option value="entertainment">Entertainment</Option>
              <Option value="other">Other</Option>
            </Select>
          </Form.Item>
        );
      
      case 'bias':
        return (
          <Form.Item
            name="bias_level"
            label="Bias Assessment"
            rules={[{ required: true, message: 'Please assess bias' }]}
          >
            <Radio.Group>
              <Radio value="unbiased">Unbiased</Radio>
              <Radio value="slightly_biased">Slightly Biased</Radio>
              <Radio value="heavily_biased">Heavily Biased</Radio>
            </Radio.Group>
          </Form.Item>
        );
      
      default:
        return null;
    }
  };

  if (loading && !currentTask) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: '16px' }}>Loading annotation task...</div>
      </div>
    );
  }

  if (!currentTask) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <CheckCircleOutlined style={{ fontSize: '48px', color: '#52c41a' }} />
          <Title level={3}>No Tasks Available</Title>
          <Text type="secondary">
            You have completed all available annotation tasks. Check back later for new assignments.
          </Text>
        </div>
      </Card>
    );
  }

  return (
    <div className="annotation-interface">
      <Title level={2}>Annotation Interface</Title>
      
      <Row gutter={[24, 24]}>
        <Col xs={24} lg={16}>
          <Card title="Content to Annotate" extra={<Tag color="blue">{currentTask.annotation_type}</Tag>}>
            <div className="content-display">
              <Paragraph>{currentTask.content}</Paragraph>
            </div>
            
            {currentTask.guidelines && (
              <Alert
                message="Annotation Guidelines"
                description={currentTask.guidelines}
                type="info"
                icon={<InfoCircleOutlined />}
                className="annotation-guidelines"
              />
            )}
          </Card>
        </Col>
        
        <Col xs={24} lg={8}>
          <Card title="Task Information">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>Task ID:</Text> {currentTask.id}
              </div>
              <div>
                <Text strong>Type:</Text> {currentTask.annotation_type}
              </div>
              <div>
                <Text strong>Priority:</Text> 
                <Tag color={currentTask.priority === 'high' ? 'red' : 'blue'}>
                  {currentTask.priority}
                </Tag>
              </div>
              <div>
                <Text strong>Deadline:</Text> {new Date(currentTask.deadline).toLocaleString()}
              </div>
              <Divider />
              <div>
                <Text strong>Time Spent:</Text>
                <div style={{ marginTop: '8px' }}>
                  <ClockCircleOutlined /> {Math.floor(timeSpent / 60)}m {timeSpent % 60}s
                </div>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="Annotation Form" className="annotation-form">
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            annotation_type: annotationType
          }}
        >
          <Form.Item
            name="annotation_type"
            label="Annotation Type"
            rules={[{ required: true, message: 'Please select annotation type' }]}
          >
            <Select 
              value={annotationType} 
              onChange={setAnnotationType}
              disabled
            >
              <Option value="sentiment">Sentiment Analysis</Option>
              <Option value="quality">Quality Assessment</Option>
              <Option value="topic">Topic Classification</Option>
              <Option value="bias">Bias Detection</Option>
            </Select>
          </Form.Item>

          {renderAnnotationForm()}

          <Form.Item
            name="comments"
            label="Additional Comments (Optional)"
          >
            <TextArea 
              rows={3} 
              placeholder="Add any additional notes or observations..."
            />
          </Form.Item>

          <Form.Item label="Confidence Level">
            <Slider
              min={0}
              max={1}
              step={0.1}
              value={confidence}
              onChange={setConfidence}
              marks={{
                0: '0%',
                0.5: '50%',
                1: '100%'
              }}
            />
            <Text type="secondary">How confident are you in this annotation?</Text>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading}
                icon={<SendOutlined />}
                size="large"
              >
                Submit Annotation
              </Button>
              <Button onClick={() => form.resetFields()}>
                Reset Form
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card title="Recent Annotations" style={{ marginTop: '24px' }}>
        <List
          dataSource={[
            { id: 1, content: 'Sample content 1', sentiment: 'positive', time: '2 minutes ago' },
            { id: 2, content: 'Sample content 2', sentiment: 'negative', time: '5 minutes ago' },
            { id: 3, content: 'Sample content 3', sentiment: 'neutral', time: '10 minutes ago' }
          ]}
          renderItem={item => (
            <List.Item>
              <List.Item.Meta
                title={`Content #${item.id}`}
                description={
                  <Space>
                    <Tag color="blue">Sentiment: {item.sentiment}</Tag>
                    <Text type="secondary">{item.time}</Text>
                  </Space>
                }
              />
            </List.Item>
          )}
          pagination={false}
          size="small"
        />
      </Card>
    </div>
  );
};

export default AnnotationInterface;
