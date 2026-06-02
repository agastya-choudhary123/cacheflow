import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { MetricCard } from './components/MetricCard';
import { DataTable, TableColumn } from './components/DataTable';
import { BarChart3, Zap, TrendingUp, Activity } from 'lucide-react';
import './styles/design-system.css';
import './styles/app.css';

interface DashboardData {
  metrics: {
    total_tokens: number;
    total_saved: number;
    savings_pct: number;
    total_sessions: number;
    agent_count: number;
  };
  agents: any[];
  sessions: any[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/data');
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US', { notation: 'compact' }).format(num);
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const sessionColumns: TableColumn<any>[] = [
    {
      key: 'agent_name',
      label: 'Agent',
      width: '20%',
      sortable: true,
    },
    {
      key: 'task',
      label: 'Task',
      width: '40%',
      render: (value) => (
        <span className="text-secondary" title={value}>
          {value?.substring(0, 50)}...
        </span>
      ),
    },
    {
      key: 'tokens_in',
      label: 'Input Tokens',
      width: '15%',
      sortable: true,
      align: 'right',
      render: (value) => <span className="text-accent">{formatNumber(value)}</span>,
    },
    {
      key: 'tokens_out',
      label: 'Output Tokens',
      width: '15%',
      sortable: true,
      align: 'right',
      render: (value) => <span className="text-accent">{formatNumber(value)}</span>,
    },
    {
      key: 'duration_ms',
      label: 'Duration',
      width: '10%',
      sortable: true,
      align: 'right',
      render: (value) => formatDuration(value),
    },
  ];

  const agentColumns: TableColumn<any>[] = [
    {
      key: 'name',
      label: 'Agent Name',
      width: '25%',
      sortable: true,
    },
    {
      key: 'model',
      label: 'Model',
      width: '20%',
      render: (value) => <span className="text-secondary">{value}</span>,
    },
    {
      key: 'session_count',
      label: 'Sessions',
      width: '15%',
      sortable: true,
      align: 'center',
    },
    {
      key: 'total_tokens_used',
      label: 'Tokens Used',
      width: '15%',
      sortable: true,
      align: 'right',
      render: (value) => formatNumber(value),
    },
    {
      key: 'total_tokens_saved',
      label: 'Tokens Saved',
      width: '15%',
      sortable: true,
      align: 'right',
      render: (value) => (
        <span className="text-success">{formatNumber(value)}</span>
      ),
    },
    {
      key: 'savings_pct',
      label: 'Savings %',
      width: '10%',
      sortable: true,
      align: 'right',
      render: (value) => (
        <span className="text-accent">{value.toFixed(1)}%</span>
      ),
    },
  ];

  const renderOverview = () => (
    <div className="main-content">
      <Header
        title="Dashboard Overview"
        subtitle="Real-time insights into your CacheFlow agents"
        onRefresh={fetchDashboardData}
        isLoading={loading}
      />

      <div className="content-area">
        {/* KPI Metrics Grid */}
        <div className="metrics-grid">
          <MetricCard
            label="Total Tokens Used"
            value={formatNumber(data?.metrics.total_tokens || 0)}
            subtext="Cumulative across all agents"
            variant="primary"
            icon={<Zap size={18} />}
            size="lg"
            delay={0}
          />
          <MetricCard
            label="Tokens Saved"
            value={formatNumber(data?.metrics.total_saved || 0)}
            subtext="Through KV cache optimization"
            variant="success"
            icon={<TrendingUp size={18} />}
            highlight
            size="lg"
            delay={100}
          />
          <MetricCard
            label="Savings Percentage"
            value={`${(data?.metrics.savings_pct || 0).toFixed(1)}%`}
            subtext="Efficiency gain"
            variant="primary"
            trend={{
              direction: 'up',
              value: (data?.metrics.savings_pct || 0) > 50 ? 12 : 5,
            }}
            size="md"
            delay={200}
          />
          <MetricCard
            label="Total Sessions"
            value={data?.metrics.total_sessions || 0}
            subtext="Across all agents"
            variant="secondary"
            icon={<Activity size={18} />}
            size="md"
            delay={300}
          />
          <MetricCard
            label="Active Agents"
            value={data?.metrics.agent_count || 0}
            subtext="Concurrent agents"
            variant="warning"
            icon={<BarChart3 size={18} />}
            size="md"
            delay={400}
          />
        </div>

        {/* Agents Table */}
        <DataTable
          columns={agentColumns}
          data={data?.agents || []}
          title="Agents Performance"
          loading={loading}
          empty="No agents found"
        />

        {/* Recent Sessions */}
        <DataTable
          columns={sessionColumns}
          data={data?.sessions || []}
          title="Recent Sessions"
          loading={loading}
          empty="No sessions recorded"
        />
      </div>
    </div>
  );

  const renderAgents = () => (
    <div className="main-content">
      <Header title="Agents Management" subtitle="Manage your CacheFlow agents" />
      <div className="content-area">
        <div className="placeholder-section">
          <h3>Agent Details Coming Soon</h3>
          <p>Detailed agent management interface will be available here</p>
        </div>
      </div>
    </div>
  );

  const renderSessions = () => (
    <div className="main-content">
      <Header title="Session History" subtitle="View all recorded sessions" />
      <div className="content-area">
        <DataTable
          columns={sessionColumns}
          data={data?.sessions || []}
          title="All Sessions"
          loading={loading}
          empty="No sessions found"
        />
      </div>
    </div>
  );

  const renderSettings = () => (
    <div className="main-content">
      <Header title="Settings" subtitle="Configure dashboard preferences" />
      <div className="content-area">
        <div className="placeholder-section">
          <h3>Settings Coming Soon</h3>
          <p>Dashboard settings and preferences will be available here</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="app">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="app-main">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'agents' && renderAgents()}
        {activeTab === 'sessions' && renderSessions()}
        {activeTab === 'settings' && renderSettings()}
      </main>
    </div>
  );
}
