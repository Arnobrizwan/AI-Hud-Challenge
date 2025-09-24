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
  Avatar,
  Tooltip,
  Badge
} from 'antd';
import { 
  UserOutlined, 
  EditOutlined, 
  DeleteOutlined,
  PlusOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;
const { Option } = Select;

const UserManagement = () => {
  const [loading, setLoading] = useState(false);
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [userModalVisible, setUserModalVisible] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      // Mock data - in real app, this would come from API
      const mockUsers = [
        {
          id: 'user-001',
          username: 'admin',
          email: 'admin@feedback-service.com',
          full_name: 'System Administrator',
          role: 'admin',
          is_active: true,
          created_at: '2024-01-01T00:00:00Z',
          last_login: '2024-01-07T10:30:00Z',
          annotation_count: 0,
          review_count: 0
        },
        {
          id: 'user-002',
          username: 'editor1',
          email: 'editor1@feedback-service.com',
          full_name: 'Content Editor 1',
          role: 'editor',
          is_active: true,
          created_at: '2024-01-02T00:00:00Z',
          last_login: '2024-01-07T09:15:00Z',
          annotation_count: 0,
          review_count: 15
        },
        {
          id: 'user-003',
          username: 'annotator1',
          email: 'annotator1@feedback-service.com',
          full_name: 'Content Annotator 1',
          role: 'annotator',
          is_active: true,
          created_at: '2024-01-03T00:00:00Z',
          last_login: '2024-01-07T08:45:00Z',
          annotation_count: 45,
          review_count: 0
        },
        {
          id: 'user-004',
          username: 'annotator2',
          email: 'annotator2@feedback-service.com',
          full_name: 'Content Annotator 2',
          role: 'annotator',
          is_active: false,
          created_at: '2024-01-04T00:00:00Z',
          last_login: '2024-01-05T16:20:00Z',
          annotation_count: 23,
          review_count: 0
        }
      ];
      setUsers(mockUsers);
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleEditUser = (user) => {
    setSelectedUser(user);
    setUserModalVisible(true);
    form.setFieldsValue({
      username: user.username,
      email: user.email,
      full_name: user.full_name,
      role: user.role,
      is_active: user.is_active
    });
  };

  const handleCreateUser = () => {
    setSelectedUser(null);
    setUserModalVisible(true);
    form.resetFields();
  };

  const handleUserSubmit = async (values) => {
    try {
      setLoading(true);
      
      if (selectedUser) {
        // Update existing user
        await axios.put(`/api/v1/users/${selectedUser.id}`, values);
        message.success('User updated successfully');
      } else {
        // Create new user
        await axios.post('/api/v1/users', values);
        message.success('User created successfully');
      }
      
      setUserModalVisible(false);
      fetchUsers();
      
    } catch (error) {
      console.error('Error saving user:', error);
      message.error('Failed to save user');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteUser = async (userId) => {
    try {
      await axios.delete(`/api/v1/users/${userId}`);
      message.success('User deleted successfully');
      fetchUsers();
    } catch (error) {
      console.error('Error deleting user:', error);
      message.error('Failed to delete user');
    }
  };

  const getRoleColor = (role) => {
    switch (role) {
      case 'admin': return 'red';
      case 'editor': return 'blue';
      case 'annotator': return 'green';
      case 'viewer': return 'default';
      default: return 'default';
    }
  };

  const getStatusColor = (isActive) => {
    return isActive ? 'green' : 'red';
  };

  const columns = [
    {
      title: 'User',
      key: 'user',
      width: 200,
      render: (_, record) => (
        <Space>
          <Avatar icon={<UserOutlined />} />
          <div>
            <div style={{ fontWeight: 'bold' }}>{record.full_name}</div>
            <div style={{ fontSize: '12px', color: '#666' }}>@{record.username}</div>
          </div>
        </Space>
      )
    },
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
      width: 200,
    },
    {
      title: 'Role',
      dataIndex: 'role',
      key: 'role',
      width: 100,
      render: (role) => (
        <Tag color={getRoleColor(role)}>
          {role.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Status',
      dataIndex: 'is_active',
      key: 'is_active',
      width: 100,
      render: (isActive) => (
        <Tag color={getStatusColor(isActive)}>
          {isActive ? 'Active' : 'Inactive'}
        </Tag>
      )
    },
    {
      title: 'Activity',
      key: 'activity',
      width: 150,
      render: (_, record) => (
        <div>
          <div>Annotations: {record.annotation_count}</div>
          <div>Reviews: {record.review_count}</div>
        </div>
      )
    },
    {
      title: 'Last Login',
      dataIndex: 'last_login',
      key: 'last_login',
      width: 150,
      render: (date) => new Date(date).toLocaleDateString()
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEditUser(record)}
          >
            Edit
          </Button>
          <Button
            type="primary"
            danger
            size="small"
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteUser(record.id)}
          >
            Delete
          </Button>
        </Space>
      )
    }
  ];

  const roleStats = {
    admin: users.filter(u => u.role === 'admin').length,
    editor: users.filter(u => u.role === 'editor').length,
    annotator: users.filter(u => u.role === 'annotator').length,
    viewer: users.filter(u => u.role === 'viewer').length
  };

  return (
    <div>
      <Title level={2}>User Management</Title>
      
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Badge count={roleStats.admin} showZero>
                <UserOutlined style={{ fontSize: '24px', color: '#ff4d4f' }} />
              </Badge>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Administrators</Text>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Badge count={roleStats.editor} showZero>
                <UserOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
              </Badge>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Editors</Text>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Badge count={roleStats.annotator} showZero>
                <UserOutlined style={{ fontSize: '24px', color: '#52c41a' }} />
              </Badge>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Annotators</Text>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Badge count={roleStats.viewer} showZero>
                <UserOutlined style={{ fontSize: '24px', color: '#722ed1' }} />
              </Badge>
              <div style={{ marginTop: '8px' }}>
                <Text strong>Viewers</Text>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Card 
        title="Users" 
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />} 
            onClick={handleCreateUser}
          >
            Add User
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={users}
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
        title={selectedUser ? 'Edit User' : 'Create User'}
        open={userModalVisible}
        onCancel={() => setUserModalVisible(false)}
        footer={null}
        width={500}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUserSubmit}
        >
          <Form.Item
            name="username"
            label="Username"
            rules={[{ required: true, message: 'Please enter username' }]}
          >
            <Input placeholder="Enter username" />
          </Form.Item>

          <Form.Item
            name="email"
            label="Email"
            rules={[
              { required: true, message: 'Please enter email' },
              { type: 'email', message: 'Please enter valid email' }
            ]}
          >
            <Input placeholder="Enter email" />
          </Form.Item>

          <Form.Item
            name="full_name"
            label="Full Name"
            rules={[{ required: true, message: 'Please enter full name' }]}
          >
            <Input placeholder="Enter full name" />
          </Form.Item>

          <Form.Item
            name="role"
            label="Role"
            rules={[{ required: true, message: 'Please select role' }]}
          >
            <Select placeholder="Select role">
              <Option value="admin">Administrator</Option>
              <Option value="editor">Editor</Option>
              <Option value="annotator">Annotator</Option>
              <Option value="viewer">Viewer</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="is_active"
            label="Status"
            valuePropName="checked"
          >
            <Select placeholder="Select status">
              <Option value={true}>Active</Option>
              <Option value={false}>Inactive</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                {selectedUser ? 'Update User' : 'Create User'}
              </Button>
              <Button onClick={() => setUserModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default UserManagement;
