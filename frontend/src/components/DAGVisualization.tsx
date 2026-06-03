import React, { useEffect, useRef, useState } from 'react';
import './dag-styles.css';

interface DAGNode {
  id: string;
  label: string;
  title: string;
  color: string;
  tokens_used: number;
  tokens_saved: number;
  index: number;
}

interface DAGEdge {
  from: string;
  to: string;
  label: string;
}

interface DAGVisualizationProps {
  agentName: string;
  onNodeClick?: (nodeId: string) => void;
}

declare global {
  interface Window {
    vis: any;
  }
}

export const DAGVisualization: React.FC<DAGVisualizationProps> = ({ agentName, onNodeClick }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [libLoaded, setLibLoaded] = useState(false);
  const [dataFetched, setDataFetched] = useState(false);

  // Load vis library once
  useEffect(() => {
    if (window.vis) {
      setLibLoaded(true);
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://unpkg.com/vis-network/standalone/umd/vis-network.min.js';
    script.async = true;
    script.onload = () => {
      console.log('[DAG] vis library loaded');
      setLibLoaded(true);
    };
    script.onerror = () => {
      console.error('[DAG] Failed to load vis library');
      setError('Failed to load visualization library');
    };
    document.head.appendChild(script);
  }, []);

  // Fetch data when lib is loaded
  useEffect(() => {
    if (!libLoaded || dataFetched) return;

    const fetchData = async () => {
      try {
        console.log(`[DAG] Fetching for agent: ${agentName}`);
        const response = await fetch(`/api/agents/${encodeURIComponent(agentName)}/dag`);
        if (!response.ok) throw new Error('Failed to fetch DAG');

        const data = await response.json();
        console.log(`[DAG] Got ${data.nodes?.length} nodes`);

        if (!data.nodes || data.nodes.length === 0) {
          setError('No commits found for this agent');
          return;
        }

        // Store the data and show the container
        window.dagData = data;
        setDataFetched(true);
        // Don't set loading to false yet - wait for container to be rendered
      } catch (err) {
        console.error('[DAG] Fetch error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load DAG');
      }
    };

    fetchData();
  }, [libLoaded, dataFetched, agentName]);

  // Render DAG after container is mounted
  useEffect(() => {
    if (!containerRef.current || !dataFetched || !window.vis) {
      console.log('[DAG] Waiting - container:', !!containerRef.current, 'data:', dataFetched, 'vis:', !!window.vis);
      return;
    }

    const renderDAG = () => {
      const { nodes: nodesList, edges: edgesList } = window.dagData;

      try {
        console.log('[DAG] Creating vis DataSet objects');

        const nodes = new window.vis.DataSet(
          nodesList.map((n: DAGNode) => ({
            id: n.id,
            label: n.label,
            title: n.title,
            color: {
              background: n.color,
              border: n.color,
              highlight: { background: n.color, border: '#fff' },
            },
            font: { color: '#fff', size: 13, face: 'system-ui' },
            shadow: { enabled: true, color: 'rgba(0,0,0,0.3)', size: 10, x: 0, y: 0 },
            borderWidth: 2,
            margin: 12,
          }))
        );

        const edges = new window.vis.DataSet(
          edgesList.map((e: DAGEdge) => ({
            from: e.from,
            to: e.to,
            arrows: { to: { enabled: true, scaleFactor: 0.8 } },
            color: { color: '#4a5568', highlight: '#63b3ed' },
            width: 2.5,
            smooth: { type: 'continuous', forceDirection: 'vertical' },
          }))
        );

        const options = {
          physics: {
            enabled: false,
          },
          layout: {
            hierarchical: {
              direction: 'UD',
              sortMethod: 'hubsize',
              nodeSpacing: 200,
              levelSeparation: 250,
            },
          },
          nodes: {
            shape: 'box',
            padding: 15,
            widthConstraint: { maximum: 180, minimum: 120 },
          },
          edges: {
            smooth: { type: 'continuous', forceDirection: 'vertical' },
          },
          interaction: {
            navigationButtons: true,
            keyboard: true,
            hover: true,
          },
        };

        console.log('[DAG] Creating network');

        if (networkRef.current) {
          networkRef.current.destroy();
        }

        networkRef.current = new window.vis.Network(
          containerRef.current,
          { nodes, edges },
          options
        );

        networkRef.current.on('click', (params: any) => {
          if (params.nodes && params.nodes.length > 0) {
            console.log('[DAG] Node clicked:', params.nodes[0]);
            onNodeClick?.(params.nodes[0]);
          }
        });

        console.log('[DAG] Network created successfully');
        setLoading(false);
      } catch (err) {
        console.error('[DAG] Render error:', err);
        setError(err instanceof Error ? err.message : 'Failed to render DAG');
      }
    };

    renderDAG();
  }, [containerRef, dataFetched]);

  useEffect(() => {
    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
      }
    };
  }, []);

  if (error) {
    return (
      <div className="dag-container">
        <div className="dag-error">
          <p>⚠ {error}</p>
        </div>
      </div>
    );
  }

  // Always show the container so ref gets attached
  return (
    <div>
      {loading && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(248, 250, 252, 0.8)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '16px',
          zIndex: 1000,
          borderRadius: '16px',
        }}>
          <div className="loading-spinner"></div>
          <p style={{ color: '#64748b', fontSize: '14px', fontWeight: '500' }}>
            Loading commit DAG for {agentName}...
          </p>
        </div>
      )}
      <div
        ref={containerRef}
        className="dag-container"
        style={{ width: '100%', height: '700px', position: 'relative', background: '#f8fafc' }}
      />
    </div>
  );
};
