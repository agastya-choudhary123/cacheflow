import React, { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import '../styles/data-table.css';

export interface TableColumn<T> {
  key: keyof T;
  label: string;
  width?: string;
  render?: (value: any, row: T) => React.ReactNode;
  sortable?: boolean;
  align?: 'left' | 'center' | 'right';
}

interface DataTableProps<T extends { id?: string | number }> {
  columns: TableColumn<T>[];
  data: T[];
  title?: string;
  loading?: boolean;
  empty?: string;
  onRowClick?: (row: T) => void;
}

type SortOrder = 'asc' | 'desc' | null;

export const DataTable = React.forwardRef<
  HTMLDivElement,
  DataTableProps<any>
>(function DataTable(
  {
    columns,
    data,
    title,
    loading = false,
    empty = 'No data available',
    onRowClick,
  },
  ref
) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<SortOrder>(null);

  const handleSort = (key: string) => {
    if (sortKey === key) {
      if (sortOrder === 'asc') {
        setSortOrder('desc');
      } else if (sortOrder === 'desc') {
        setSortKey(null);
        setSortOrder(null);
      }
    } else {
      setSortKey(key);
      setSortOrder('asc');
    }
  };

  const sortedData = React.useMemo(() => {
    if (!sortKey || !sortOrder) return data;

    return [...data].sort((a, b) => {
      const aVal = a[sortKey as keyof typeof a];
      const bVal = b[sortKey as keyof typeof b];

      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
      }

      const aStr = String(aVal).toLowerCase();
      const bStr = String(bVal).toLowerCase();
      return sortOrder === 'asc'
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });
  }, [data, sortKey, sortOrder]);

  return (
    <div className="data-table-container" ref={ref}>
      {title && <h2 className="data-table-title">{title}</h2>}

      <div className="data-table-wrapper">
        {loading ? (
          <div className="data-table-loading">Loading...</div>
        ) : data.length === 0 ? (
          <div className="data-table-empty">{empty}</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th
                    key={String(col.key)}
                    style={{ width: col.width, textAlign: col.align || 'left' }}
                    className={col.sortable ? 'sortable' : ''}
                    onClick={() => col.sortable && handleSort(String(col.key))}
                  >
                    <div className="th-content">
                      <span>{col.label}</span>
                      {col.sortable && (
                        <div className="sort-icon">
                          {sortKey === String(col.key) && sortOrder === 'asc' && (
                            <ChevronUp size={14} />
                          )}
                          {sortKey === String(col.key) && sortOrder === 'desc' && (
                            <ChevronDown size={14} />
                          )}
                        </div>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedData.map((row, idx) => (
                <tr
                  key={row.id || idx}
                  className={onRowClick ? 'clickable' : ''}
                  onClick={() => onRowClick?.(row)}
                >
                  {columns.map((col) => (
                    <td
                      key={String(col.key)}
                      style={{
                        width: col.width,
                        textAlign: col.align || 'left',
                      }}
                    >
                      {col.render
                        ? col.render(row[col.key], row)
                        : String(row[col.key] ?? '-')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
});

DataTable.displayName = 'DataTable';
