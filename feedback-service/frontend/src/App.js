import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { Layout, Menu } from 'antd';
import {
  DashboardOutlined,
  EditOutlined,
  CheckCircleOutlined,
  BarChartOutlined,
  UserOutlined
} from '@ant-design/icons';

import Dashboard from './components/Dashboard';
import AnnotationInterface from './components/AnnotationInterface';
import EditorialDashboard from './components/EditorialDashboard';
import QualityDashboard from './components/QualityDashboard';
import UserManagement from './components/UserManagement';

import './App.css';

const { Header, Sider, Content } = Layout;

function App() {
  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
      path: '/'
    },
    {
      key: 'annotation',
      icon: <EditOutlined />,
      label: 'Annotation',
      path: '/annotation'
    },
    {
      key: 'editorial',
      icon: <CheckCircleOutlined />,
      label: 'Editorial',
      path: '/editorial'
    },
    {
      key: 'quality',
      icon: <BarChartOutlined />,
      label: 'Quality',
      path: '/quality'
    },
    {
      key: 'users',
      icon: <UserOutlined />,
      label: 'Users',
      path: '/users'
    }
  ];

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 6,
        },
      }}
    >
      <Router>
        <Layout style={{ minHeight: '100vh' }}>
          <Sider
            breakpoint="lg"
            collapsedWidth="0"
            style={{
              background: '#fff',
              boxShadow: '2px 0 8px rgba(0,0,0,0.15)'
            }}
          >
            <div style={{ 
              height: '64px', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              borderBottom: '1px solid #f0f0f0'
            }}>
              <h2 style={{ margin: 0, color: '#1890ff' }}>Feedback AI</h2>
            </div>
            <Menu
              mode="inline"
              defaultSelectedKeys={['dashboard']}
              style={{ borderRight: 0 }}
              items={menuItems.map(item => ({
                key: item.key,
                icon: item.icon,
                label: item.label,
                onClick: () => window.location.href = item.path
              }))}
            />
          </Sider>
          <Layout>
            <Header style={{ 
              background: '#fff', 
              padding: '0 24px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <h1 style={{ margin: 0, fontSize: '20px' }}>
                Feedback & Human-in-the-Loop Service
              </h1>
              <div style={{ color: '#666' }}>
                Real-time Annotation & Quality Control
              </div>
            </Header>
            <Content style={{ 
              margin: '24px 16px',
              padding: '24px',
              background: '#fff',
              borderRadius: '8px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
            }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/annotation" element={<AnnotationInterface />} />
                <Route path="/editorial" element={<EditorialDashboard />} />
                <Route path="/quality" element={<QualityDashboard />} />
                <Route path="/users" element={<UserManagement />} />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </Router>
    </ConfigProvider>
  );
}

export default App;
