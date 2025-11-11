# Facial Authentication System - Frontend

Modern, responsive web application for enterprise-grade facial authentication.

## Features

- 🎨 **Beautiful UI** - Modern design with Tailwind CSS and Framer Motion animations
- 📱 **Responsive** - Works seamlessly on desktop, tablet, and mobile
- 🎥 **Real-time** - Live camera feed with face detection and authentication
- 📊 **Analytics** - Comprehensive dashboard with charts and metrics
- 👥 **User Management** - Full CRUD operations for registered users
- ⚙️ **Settings** - Configurable system parameters and features
- 🚀 **Fast** - Built with Vite for lightning-fast development and builds

## Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Next-generation frontend tooling
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations and transitions
- **Recharts** - Beautiful charts and data visualization
- **React Router** - Client-side routing
- **Axios** - HTTP client for API calls
- **Lucide React** - Beautiful icon library

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on http://localhost:8000

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

The application will be available at http://localhost:5173

### Build for Production

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── public/           # Static files
├── src/
│   ├── api/         # API client and services
│   ├── components/  # Reusable components
│   ├── pages/       # Page components
│   ├── App.jsx      # Main app component
│   ├── main.jsx     # Entry point
│   └── index.css    # Global styles
├── index.html       # HTML template
├── vite.config.js   # Vite configuration
└── tailwind.config.js  # Tailwind configuration
```

## Pages

- **Home** - Landing page with features and system overview
- **Register** - Face registration with camera or file upload
- **Authenticate** - Real-time face authentication
- **Dashboard** - Analytics and system metrics
- **Users** - User management and database viewer
- **Settings** - System configuration and parameters

## API Integration

The frontend communicates with the backend API through the `api/client.js` module:

- Registration: `POST /api/v1/register`
- Authentication: `POST /api/v1/authenticate`
- Identification: `POST /api/v1/identify`
- User Management: `POST /api/v1/delete_user`
- System Info: `GET /api/v1/system/info`
- Health Check: `GET /health`

## WebSocket Support

Real-time face recognition uses WebSocket connections:

```javascript
const ws = createWebSocketConnection(clientId)
ws.send(JSON.stringify({
  type: 'frame',
  frame: base64Image,
  user_id: userId
}))
```

## Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```javascript
theme: {
  extend: {
    colors: {
      primary: { ... },
      dark: { ... }
    }
  }
}
```

### Features

Toggle features in the Settings page or by modifying the system configuration.

## Performance

- Code splitting for optimal loading
- Lazy loading of components
- Image optimization
- Minimal bundle size with tree-shaking
- Fast refresh during development

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)

## License

Part of the Facial Authentication System project.


