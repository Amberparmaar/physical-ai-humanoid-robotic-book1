# Developer Guide - BookReader with Urdu Translation

## Table of Contents
1. [Project Setup](#project-setup)
2. [Development Workflow](#development-workflow)
3. [Folder Structure](#folder-structure)
4. [Component Architecture](#component-architecture)
5. [Styling Guidelines](#styling-guidelines)
6. [Testing](#testing)
7. [API Integration](#api-integration)
8. [Performance Optimization](#performance-optimization)
9. [Accessibility Guidelines](#accessibility-guidelines)
10. [Troubleshooting](#troubleshooting)

## Project Setup

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Git version control system
- Modern code editor (VS Code recommended)

### Installation Steps

1. **Clone the repository**
````
git clone <repository-url>
cd bookreader-app
````

2. **Install dependencies**
````
npm install
# or
yarn install
````

3. **Set up environment variables**
Create a `.env` file in the root directory:
````
REACT_APP_API_URL=https://api.bookreader.com
REACT_APP_TRANSLATION_API_KEY=your_translation_api_key_here
````

4. **Start development server**
````
npm run dev
# or
yarn dev
````

5. **Open your browser** to `http://localhost:3000`

### Development Scripts
- `npm start` - Start development server with hot reloading
- `npm run build` - Create production build
- `npm run test` - Run unit tests
- `npm run lint` - Run code linter
- `npm run format` - Format code with Prettier

## Development Workflow

### Creating a New Component
1. Identify the appropriate parent directory in `frontend/src/components/`
2. Create a new file with the `.tsx` extension
3. Follow the naming convention: `ComponentName.tsx`
4. Implement component using TypeScript interfaces
5. Export as default export
6. Write tests for the component
7. Update documentation if needed

### Component File Example
````
import React from 'react';

// Define interface for props
interface ComponentNameProps {
  property: string;
  optionalProperty?: boolean;
}

// Create component using props interface
const ComponentName: React.FC<ComponentNameProps> = ({ property, optionalProperty = false }) => {
  return (
    <div className="component-class">
      <h1>{property}</h1>
      {optionalProperty && <p>Optional content</p>}
    </div>
  );
};

export default ComponentName;
````

### Git Workflow
1. Create a feature branch: `git checkout -b feature/new-feature`
2. Make changes and test thoroughly
3. Commit with meaningful message: `git commit -m "feat: add new reading functionality"`
4. Push to remote: `git push origin feature/new-feature`
5. Create pull request for review

### Commit Message Format
````<type>(<scope>): <subject>````

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi colons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Other changes that don't modify src or test files

**Example**:
- `feat(reader): add floating translation button`
- `fix(translation): fix RTL text rendering bug`
- `refactor(components): improve ReadingArea performance`

## Folder Structure

````
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
````

## Component Architecture

### Component Guidelines
1. **Single Responsibility**: Each component should have one clear purpose
2. **Reusability**: Design components to be reusable where possible
3. **Props Validation**: Use TypeScript interfaces for all props
4. **Controlled Components**: Prefer controlled components over uncontrolled
5. **Performance**: Optimize components to prevent unnecessary renders

### Context API Usage
The application uses Context API for managing theme and user preferences:

````
import { useTheme } from '../UI/ThemeProvider';

const MyComponent = () => {
  const { theme, toggleTheme, fontSize } = useTheme();
  
  return (
    <div className={theme === 'dark' ? 'dark-theme' : 'light-theme'}>
      <p style={{ fontSize: `${fontSize}px` }}>Content</p>
      <button onClick={toggleTheme}>Switch Theme</button>
    </div>
  );
};
````

### Custom Hooks
Create custom hooks for reusable logic:

````
// hooks/useBookContent.ts
import { useState, useEffect } from 'react';
import { bookService } from '../services/bookService';

export const useBookContent = (bookId: string) => {
  const [book, setBook] = useState<Book | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchBookContent = async () => {
      try {
        setLoading(true);
        const fetchedBook = await bookService.getBookContent(bookId);
        setBook(fetchedBook);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch book content');
      } finally {
        setLoading(false);
      }
    };

    if (bookId) {
      fetchBookContent();
    }
  }, [bookId]);

  return { book, loading, error };
};
````

## Styling Guidelines

### CSS Methodology
- Use Tailwind CSS for utility-first styling
- For complex custom styles, create reusable CSS modules
- Follow BEM methodology for component-specific styles
- Use CSS variables for consistent theming

### Typography
- Use appropriate font stacks for English and Urdu text
- Ensure proper line height (1.6 for English, 1.8 for Urdu)
- Maintain readable font sizes (minimum 16px for body text)
- Use appropriate letter spacing

### Color System
- Use soft, muted colors for distraction-free reading
- Ensure accessibility contrast ratios (minimum 4.5:1)
- Implement dark mode with appropriate color adjustments
- Use consistent color palette throughout the application

### Responsive Design
- Use mobile-first approach
- Implement responsive breakpoints at 640px, 768px, 1024px, and 1280px
- Ensure touch targets are appropriately sized (minimum 44px)
- Optimize for both portrait and landscape orientations

## Testing

### Unit Testing
- Use Jest and React Testing Library for component testing
- Aim for minimum 80% code coverage
- Test all user interactions
- Mock external dependencies

### Testing Example
````
import { render, screen, fireEvent } from '@testing-library/react';
import TranslationPanel from '../components/Reader/TranslationPanel';

test('renders translation panel with original and translated text', () => {
  render(
    <TranslationPanel 
      isVisible={true} 
      originalText="English text" 
      translation="اردو ٹیکسٹ"
      displayMode="side_by_side" 
      onClose={() => {}} 
    />
  );
  
  expect(screen.getByText(/English text/)).toBeInTheDocument();
  expect(screen.getByText(/اردو ٹیکسٹ/)).toBeInTheDocument();
});

test('closes panel when close button is clicked', () => {
  const onCloseMock = jest.fn();
  
  render(
    <TranslationPanel 
      isVisible={true} 
      originalText="Test" 
      translation="ٹیسٹ"
      displayMode="toggle" 
      onClose={onCloseMock} 
    />
  );
  
  fireEvent.click(screen.getByText('Close'));
  expect(onCloseMock).toHaveBeenCalledTimes(1);
});
````

### Integration Testing
- Test API interactions with mock data
- Test state management flows
- Validate cross-component interactions
- Test user journey scenarios

## API Integration

### API Client
The application uses a centralized API client in `apiClient.ts`:

````
// services/apiClient.ts
class ApiClient {
  private headers: Headers;

  constructor() {
    this.headers = new Headers({
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.REACT_APP_API_KEY}` // if needed
    });
  }

  async request(endpoint: string, options: RequestInit = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      headers: this.headers,
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // HTTP method shortcuts
  get(endpoint: string) { /* ... */ }
  post(endpoint: string, data?: any) { /* ... */ }
  put(endpoint: string, data?: any) { /* ... */ }
  delete(endpoint: string) { /* ... */ }
}
````

### Service Implementation Pattern
````
// services/bookService.ts
import { Book } from '../models/Book';
import { apiClient } from './apiClient';

export const bookService = {
  async getBooks(): Promise<Book[]> {
    try {
      const response = await apiClient.get('/books');
      return response;
    } catch (error) {
      console.error('Error fetching books:', error);
      throw error;
    }
  },

  async getBookContent(bookId: string): Promise<Book> {
    try {
      const response = await apiClient.get(`/books/${bookId}`);
      return response;
    } catch (error) {
      console.error(`Error fetching book content for ID ${bookId}:`, error);
      throw error;
    }
  },

  async uploadBook(bookData: Omit<Book, 'id'>): Promise<Book> {
    try {
      const response = await apiClient.post('/books', bookData);
      return response;
    } catch (error) {
      console.error('Error uploading book:', error);
      throw error;
    }
  },
};
````

## Performance Optimization

### Virtual Scrolling
For handling long documents, implement virtual scrolling as demonstrated in `ReadingAreaVirtualized.tsx`:

````
// Use Intersection Observer API for performance
// Only render visible content
// Use appropriate buffer for content loading
````

### Image Optimization
- Use WebP format when possible
- Implement lazy loading for images
- Use appropriate image dimensions
- Consider SVG for icons

### Code Splitting
- Implement route-based code splitting
- Use React.lazy() for non-critical components
- Preload critical resources

### Caching Strategy
- LocalStorage for user preferences
- IndexedDB for book content and translations
- API response caching where appropriate
- Implement cache invalidation when needed

## Accessibility Guidelines

### Keyboard Navigation
- Ensure all interactive elements are keyboard accessible
- Implement proper focus management
- Use skip links for main content
- Test with keyboard-only navigation

### Screen Reader Support
- Use semantic HTML elements
- Implement proper ARIA labels and descriptions
- Ensure proper heading hierarchy
- Use `role` attributes when necessary

### Color and Contrast
- Maintain minimum 4.5:1 contrast ratio
- Don't rely solely on color to convey information
- Test with color blindness simulators
- Provide alternative visual indicators

### Focus Indicators
- Never remove default focus styles without replacement
- Use highly visible focus indicators
- Ensure focus order matches visual order
- Test focus flow through components

## Troubleshooting

### Common Issues

#### 1. API Requests Failing
- Check if API URL is correctly set in environment variables
- Verify API endpoints are accessible
- Check if authentication tokens are valid
- Look for CORS issues in browser console

#### 2. Translation Panel Not Showing
- Confirm selected text is being captured correctly
- Verify translation service is responding
- Check if component state is being updated properly
- Ensure proper event handling

#### 3. Performance Issues with Long Documents
- Implement virtual scrolling
- Use React.memo for components that render many items
- Optimize rendering of paragraph elements
- Consider pagination for very long content

#### 4. Dark Mode Not Applying Correctly
- Verify CSS classes are being applied
- Check if theme context is updating correctly
- Ensure Tailwind dark mode is configured properly
- Test with various components

#### 5. Font Size Changes Not Persisting
- Check if storage service is saving preferences correctly
- Verify updatePreferences function is being called
- Confirm localStorage/IndexedDB is available
- Test preference synchronization across components

### Browser Compatibility
- Test in modern browsers (Chrome, Firefox, Safari, Edge)
- Use feature detection for browser-specific features
- Provide fallbacks for older browsers
- Consider polyfills where necessary

### Debugging Tools
- React Developer Tools browser extension
- Browser's developer console
- Network tab for API request monitoring
- Component state inspection tools
- Performance profiling tools