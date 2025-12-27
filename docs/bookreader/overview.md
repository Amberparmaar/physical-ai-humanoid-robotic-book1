# BookReader with Urdu Translation

## Project Overview

The BookReader with Urdu Translation is a modern, minimalist web application that provides an exceptional reading experience with integrated Urdu translation capabilities. The application features a clean, distraction-free interface that allows users to enjoy reading content while having access to real-time Urdu translations for selected text.

## Key Features

### Reading Experience
- Clean reading area with smooth typography optimized for long reading sessions
- Support for multiple book formats (plain text, PDF, EPUB)
- Responsive design for cross-device compatibility
- Performance optimized for smooth scrolling and fast interactions

### Urdu Translation
- Floating translation button for instant access to Urdu translations
- Side-by-side or toggle view for bilingual reading
- Translation available within 2 seconds of activation
- Right-to-left text rendering for Urdu content

### Customization
- Adjustable font size (12px to 36px range)
- Light/dark theme switching
- Language preference settings (English, Urdu, or both)
- Reading mode options (side-by-side vs. toggle)

### Technical Highlights
- Built with React 18+ and TypeScript
- Styled with Tailwind CSS following minimalist design principles
- Performance optimized with virtual scrolling for long documents
- Caching implemented with LocalStorage and IndexedDB
- Full accessibility support

## Tech Stack

- **Frontend**: React 18+, TypeScript, Tailwind CSS
- **Build Tool**: Vite
- **State Management**: React Context API with custom hooks
- **API Communication**: Custom API client with error handling
- **Styling**: Tailwind CSS with custom typography components
- **Caching**: LocalStorage for preferences, IndexedDB for content
- **Icons**: Custom SVG icons or Font Awesome if required

## Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── Reader/         # Reading-specific components
│   │   ├── Layout/         # Layout components
│   │   ├── Books/          # Book browsing components
│   │   └── UI/             # General UI components
│   ├── services/           # API and business logic services
│   ├── models/             # TypeScript type definitions for entities
│   ├── hooks/              # Custom React hooks
│   ├── styles/             # CSS styling files
│   ├── pages/              # Page-level components
│   ├── utils/              # Utility functions
│   ├── types/              # Global type definitions
│   ├── assets/             # Images, fonts, etc.
│   └── App.tsx             # Main application component
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn package manager
- Git

### Setting Up Locally

1. **Clone the repository**
````
git clone <repository-url>
cd bookreader-app
````

2. **Install dependencies**
````
npm install
````

3. **Set up environment variables**
Create a `.env` file in the root directory:
````
REACT_APP_API_URL=https://api.bookreader.com
REACT_APP_TRANSLATION_API_KEY=your_translation_api_key_here
````

4. **Start the development server**
````
npm run dev
````

5. **Open your browser** to `http://localhost:3000`

### Development Scripts

- `npm run dev` - Start development server with hot reloading
- `npm run build` - Create production build
- `npm run preview` - Preview production build locally
- `npm run lint` - Lint code for potential issues
- `npm run format` - Format code with Prettier

## API Endpoints

The application integrates with the following API endpoints:

- `GET /api/books` - Retrieve list of available books
- `GET /api/books/{bookId}` - Get content of a specific book
- `POST /api/books` - Upload a new book
- `POST /api/books/{bookId}/translate` - Get Urdu translation for selected text
- `GET /api/user/preferences` - Get current user preferences
- `PUT /api/user/preferences` - Update user preferences
- `GET /api/books/{bookId}/progress` - Get reading progress
- `PUT /api/books/{bookId}/progress` - Update reading progress

## Usage Guide

1. **Browse Books**: Navigate to the library to see all available books
2. **Select a Book**: Click on any book card to start reading
3. **Translate Text**: 
   - Select any paragraph in the reading area
   - Click the floating translation button
   - View the Urdu translation in the panel
4. **Adjust Settings**:
   - Use the settings icon to adjust font size
   - Toggle between light and dark themes
   - Change language display preferences (English/Urdu/Both)
5. **Save Progress**: Your reading progress is automatically saved

## Design Principles

This application follows the BookReader UI Constitution:

1. **User Experience First**: Every feature is designed with the user's reading experience as priority
2. **Accessibility & Inclusion**: Full support for Urdu translation and multiple language displays
3. **Minimalist Design**: Clean UI with soft colors and rounded corners
4. **Performance & Responsiveness**: Optimized for fast loading and smooth interactions
5. **Clean Typography**: Optimized for long reading sessions
6. **Consistent & Predictable Behavior**: All interactions are consistent and intuitive

## Contributing

We welcome contributions to improve the BookReader application!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Guidelines

- Follow TypeScript best practices
- Write accessible components with proper ARIA attributes
- Use Tailwind CSS utility classes for styling
- Write tests for new components and features
- Follow the existing code structure and formatting
- Document new components and functions

## Testing

The application includes various levels of testing:

- Unit tests for individual components and utility functions
- Integration tests for component interactions
- API mocking for testing service layers
- Accessibility testing with automated tools

To run tests:
````
npm run test
````

## Deployment

The application can be deployed to any static hosting service:

1. Build the application: `npm run build`
2. Host the contents of the `dist/` directory
3. Configure your domain and SSL certificates
4. Monitor performance and errors

## Performance

The application implements several performance optimizations:

- Virtual scrolling for long documents
- Caching of book content and translations
- Lazy loading of non-essential components
- Efficient rendering with React's reconciliation algorithm
- Tree-shaking to minimize bundle size

## Security

Security considerations include:

- Input sanitization for user-generated content
- Secure API communication with HTTPS
- Sanitization of translation results
- Protection against XSS attacks
- Proper error handling that doesn't expose system information

## Troubleshooting

### Common Issues

- **API requests failing**: Check if the API URL is correctly configured in environment variables
- **Translation not appearing**: Verify that translations are available for the selected text
- **Performance issues**: Large documents might require virtual scrolling implementation
- **Font rendering problems**: Ensure proper font loading and fallbacks

### Browser Compatibility

The application is designed to work with modern browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+).

## Support

For support, please check:
- Our documentation for solutions to common issues
- The issue tracker to see if others have faced similar problems
- Contact our team through the official channels if issues persist

For feedback and feature requests, feel free to open an issue or submit a pull request!