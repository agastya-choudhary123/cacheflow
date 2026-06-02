# CacheFlow Premium Dashboard

A hyper-premium, data-rich React + TypeScript dashboard for CacheFlow with dark glassmorphism aesthetic and electric accent colors.

## Quick Start

### Build the frontend

```bash
cd frontend
npm install
npm run build
```

This creates an optimized production build in `frontend/dist/`.

### Run the dashboard

```bash
cd ..
cf dashboard
```

The dashboard will be available at `http://localhost:8080`

## Development

### Start the development server

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on `http://localhost:5173` with automatic proxy to the Flask backend at `http://localhost:8080/api`.

### Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   │   ├── Sidebar.tsx   # Navigation sidebar
│   │   ├── Header.tsx    # Page header
│   │   ├── MetricCard.tsx # KPI metric card
│   │   └── DataTable.tsx # Sortable data table
│   ├── styles/           # CSS files
│   │   ├── design-system.css # CSS variables & base styles
│   │   ├── app.css      # Main layout
│   │   ├── sidebar.css  # Sidebar styling
│   │   ├── header.css   # Header styling
│   │   ├── metric-card.css
│   │   └── data-table.css
│   ├── App.tsx          # Main app component
│   └── main.tsx         # Entry point
├── index.html           # HTML template
├── vite.config.ts       # Vite configuration
└── package.json         # Dependencies
```

## Design System

### Colors

- **Primary Accent**: Electric Violet (`#a78bfa`)
- **Secondary Accent**: Neon Cyan (`#22d3ee`)
- **Background**: Deep Black (`#0f0f12`) with subtle gradients
- **Text**: Off-white (`#f5f5f7`)

### Typography

- **Display Font**: Space Grotesk (headings, metrics)
- **Body Font**: Inter (all text content)

### Components

- **Glass Cards**: Frosted glass effect with backdrop blur
- **Gradient Borders**: Subtle violet-to-cyan blend on hover
- **Metric Cards**: Large, prominent KPI displays with animations
- **Data Tables**: Sortable with smooth interactions

## Build Output

The `npm run build` command creates:
- Minified JavaScript bundle
- Optimized CSS
- Inline critical styles
- Image optimization

Output is in `frontend/dist/` and is served directly by Flask.

## Deployment

1. Build the frontend: `npm run build`
2. The Flask dashboard automatically serves files from `frontend/dist/`
3. No separate web server needed

## API Endpoints

The frontend communicates with these Flask API endpoints:

- `GET /api/data` - Dashboard metrics and tables
- `GET /api/agents/<agent_name>/dag` - Commit DAG visualization
- `GET /api/agents/<agent_name>/commits/<commit_id>/summary` - Snapshot summary
- `GET /api/agents/<agent_name>/commits/<commit_id>/deep` - Deep analysis
- `GET /api/query?q=<query>&agent=<name>` - Semantic search
