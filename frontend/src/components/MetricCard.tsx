import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import '../styles/metric-card.css';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  variant?: 'primary' | 'secondary' | 'success' | 'warning';
  trend?: {
    direction: 'up' | 'down';
    value: number;
  };
  icon?: React.ReactNode;
  highlight?: boolean;
  size?: 'sm' | 'md' | 'lg';
  delay?: number;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  variant = 'primary',
  trend,
  icon,
  highlight = false,
  size = 'md',
  delay = 0,
}) => {
  return (
    <div
      className={`metric-card metric-card-${size} metric-card-${variant} ${
        highlight ? 'highlight' : ''
      } animate-fade-in-up animation-delay-${delay}`}
    >
      <div className="metric-header">
        <div className="metric-label">{label}</div>
        {icon && <div className="metric-icon">{icon}</div>}
      </div>

      <div className="metric-body">
        <div className="metric-value">{value}</div>
        {subtext && <div className="metric-subtext">{subtext}</div>}
      </div>

      {trend && (
        <div className={`metric-trend metric-trend-${trend.direction}`}>
          {trend.direction === 'up' ? (
            <TrendingUp size={14} />
          ) : (
            <TrendingDown size={14} />
          )}
          <span>{trend.value}%</span>
        </div>
      )}
    </div>
  );
};
