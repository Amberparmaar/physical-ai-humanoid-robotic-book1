# BookReader UI with Urdu Translation - Documentation

## Overview

The BookReader UI with Urdu Translation is a modern and minimal web application that provides a premium reading experience with integrated Urdu translation capabilities. The application features clean typography, a distraction-free interface, and intuitive language switching options.

## Features

### Core Reading Experience
- Clean reading area with smooth typography optimized for long reading sessions
- Support for common book formats (plain text, PDF, EPUB)
- Responsive design that works across desktop and mobile devices

### Urdu Translation
- Floating translation button to access Urdu translations for selected paragraphs
- Side-by-side or toggle view to switch between English and Urdu text
- Translation available within 2 seconds as per performance requirements

### Customization Options
- Adjustable font size (12px to 36px range)
- Dark/light mode switching
- Language preference settings (English only, Urdu only, or both languages)

### User Experience
- Minimalist design with soft colors and rounded corners
- Consistent and predictable UI interactions
- Performance optimized for smooth scrolling and fast translations

## Architecture

### Technology Stack
- Frontend: React 18+ with TypeScript
- Styling: Tailwind CSS
- State Management: React Context API with custom hooks
- Build Tool: Vite
- Caching: LocalStorage for preferences, IndexedDB for content

### Project Structure
```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── Reader/          # Components for reading functionality
│   │   ├── Layout/          # Layout components
│   │   ├── Books/           # Book listing components
│   │   └── UI/              # Reusable UI components
│   ├── services/            # API and business logic services
│   ├── models/              # Data models
│   ├── hooks/               # Custom React hooks
│   ├── styles/              # CSS styling
│   ├── pages/               # Page components
│   ├── utils/               # Utility functions
│   ├── types/               # TypeScript type definitions
│   └── App.tsx              # Main application component
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### Key Components

#### Reader Components
- **ReadingArea**: Main component for displaying book content with paragraph selection
- **TranslationPanel**: Modal panel for displaying Urdu translations
- **FloatingTranslationButton**: Floating button to trigger translation functionality
- **LanguageToggle**: Component for switching between language display modes

#### Layout Components
- **SettingsPanel**: Modal for adjusting user preferences
- **ThemeProvider**: Context provider for theme management

#### Book Components
- **BookList**: List view for available books
- **BookCard**: Individual book display component

#### UI Components
- **Button**: Reusable button component
- **Slider**: Range slider for font size adjustment

## API Integration

The application integrates with the following API endpoints:

### Books API
- `GET /api/books` - Retrieve list of available books
- `GET /api/books/{bookId}` - Get content of a specific book
- `POST /api/books` - Upload a new book

### Translation API
- `POST /api/books/{bookId}/translate` - Get Urdu translation for selected text

### User Preferences API
- `GET /api/user/preferences` - Get current user preferences
- `PUT /api/user/preferences` - Update user preferences

### Reading Progress API
- `GET /api/books/{bookId}/progress` - Get reading progress
- `PUT /api/books/{bookId}/progress` - Update reading progress

## Data Models

### Book
````
interface Book {
  id: string;
  title: string;
  author: string;
  originalText: string;
  urduTranslation?: string;
  format: 'plain_text' | 'pdf' | 'epub';
  language: string;
  hasUrduTranslation: boolean;
  length: number;
  paragraphs: Paragraph[];
  metadata: BookMetadata;
}
````

### Paragraph
````
interface Paragraph {
  id: string;
  originalText: string;
  urduTranslation: string;
  position: number;
  bookId: string;
}
````

### UserPreferences
````
interface UserPreferences {
  id: string;
  userId: string;
  fontSize: number; // between 12 and 36
  theme: 'light' | 'dark';
  languagePreference: 'english' | 'urdu' | 'both';
  lastReadingPosition: {
    bookId: string;
    paragraphId: string;
  } | null;
  readingMode: 'side_by_side' | 'toggle';
}
````

## Implementation Details

### Translation Functionality
The translation feature works by:
1. User selects text in the ReadingArea component
2. FloatingTranslationButton appears for the selected text
3. On click, the text is sent to the translation API
4. TranslationPanel displays the original and translated text
5. Can be shown in side-by-side or toggle view

### Theme Management
The application uses a ThemeProvider component that:
- Manages light/dark mode state
- Persists theme preferences to local storage
- Applies theme classes to the body element
- Ensures consistent styling throughout the application

### Performance Optimizations
- Virtual scrolling for long documents
- Caching book content and translations in IndexedDB
- Lazy loading of content sections
- Memoization of expensive computations

## User Interface Guidelines

### Typography
- Uses Georgia or Times New Roman for serif fonts
- Urdu text uses Jameel Noori Nastaleeq font
- Responsive font sizes from 16px to 24px
- Appropriate line height and letter spacing for readability

### Color Palette
- Soft, muted colors following minimalist design principles
- High contrast for accessibility in both light and dark modes
- Consistent color scheme across all components

### Layout
- Clean, uncluttered interface with ample white space
- Consistent padding and margins
- Responsive grid layouts that adapt to different screen sizes

## Development

### Getting Started
1. Clone the repository
2. Install dependencies: `npm install`
3. Start development server: `npm run dev`
4. Visit `http://localhost:3000`

### Environment Variables
- `REACT_APP_API_URL` - Base URL for API endpoints
- `REACT_APP_TRANSLATION_API_KEY` - API key for translation service (if needed)

### Testing
- Unit tests using Jest and React Testing Library
- Integration tests for API interactions
- End-to-end tests with Cypress (planned)

## Deployment

The application can be built for production using:
```
npm run build
```

This creates an optimized build in the `dist/` directory that can be served by a web server.

## Accessibility

The application follows accessibility best practices:
- Sufficient color contrast ratios
- Proper heading hierarchy
- Keyboard navigation support
- Screen reader compatibility
- Adjustable font sizes
- Dark/light mode options for different user preferences