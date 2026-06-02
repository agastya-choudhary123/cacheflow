import React from 'react';
import { RefreshCw } from 'lucide-react';
import '../styles/header.css';

interface HeaderProps {
  title: string;
  subtitle?: string;
  onRefresh?: () => void;
  isLoading?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  onRefresh,
  isLoading = false,
}) => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-text">
          <h1 className="header-title">{title}</h1>
          {subtitle && <p className="header-subtitle">{subtitle}</p>}
        </div>

        <div className="header-actions">
          {onRefresh && (
            <button
              className={`btn btn-secondary ${isLoading ? 'loading' : ''}`}
              onClick={onRefresh}
              disabled={isLoading}
              aria-label="Refresh dashboard"
            >
              <RefreshCw size={16} />
              Refresh
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
