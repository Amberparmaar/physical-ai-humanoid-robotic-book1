# Component Documentation - BookReader with Urdu Translation

This document provides detailed information about the React components used in the BookReader UI with Urdu Translation feature.

## Component Hierarchy

```
App
├── ThemeProvider
│   ├── Header
│   ├── Main Content Area
│   │   ├── LibraryPage (when browsing books)
│   │   ├── ReaderPage (when reading)
│   │   └── SettingsPage (when viewing settings)
│   └── Footer
└── Global Modals (like TranslationPanel)
```

## Core Components

### App.tsx
**Location**: `frontend/src/App.tsx`

**Purpose**: Main application component that sets up routing and wraps the entire application with the ThemeProvider.

**Props**: None
**State**: None
**Contexts Used**: ThemeProvider

---

### ThemeProvider.tsx
**Location**: `frontend/src/components/UI/ThemeProvider.tsx`

**Purpose**: Provides theme context (light/dark mode, font size, user preferences) to all child components.

**Props**: 
- `children`: ReactNode - Child components to wrap

**State**:
- `theme`: 'light' | 'dark' - Current theme
- `currentPreferences`: UserPreferences | null - Current user preferences
- `fontSize`: number - Current font size

**Context Value**:
- `theme`: Current theme
- `toggleTheme`: Function to toggle theme
- `currentPreferences`: Current user preferences
- `updatePreferences`: Function to update preferences
- `fontSize`: Current font size
- `setFontSize`: Function to set font size

---

## Reader Components

### ReadingArea.tsx
**Location**: `frontend/src/components/Reader/ReadingArea.tsx`

**Purpose**: Main component for displaying book content with paragraph selection functionality.

**Props**:
- `content`: string - The book content to display
- `paragraphs`: Paragraph[] | undefined - Optional array of paragraphs if available
- `currentParagraphIndex`: number - Index of the currently selected paragraph
- `onParagraphSelect`: Function - Callback when a paragraph is selected

**State**:
- `selectedParagraph`: number | null - Currently selected paragraph index

**Features**:
- Displays content with appropriate typography
- Enables paragraph selection
- Supports language preference settings (English, Urdu, or both)
- Responsive to theme and font size changes
- Performance optimized for long content

---

### TranslationPanel.tsx
**Location**: `frontend/src/components/Reader/TranslationPanel.tsx`

**Purpose**: Modal panel that displays the Urdu translation of selected text.

**Props**:
- `isVisible`: boolean - Whether the panel should be displayed
- `originalText`: string - The original English text
- `translation`: string - The Urdu translation
- `displayMode`: 'side_by_side' | 'toggle' - How to display the languages
- `onClose`: Function - Callback when the panel is closed
- `onModeChange`: Function | undefined - Callback when display mode changes

**Features**:
- Modal display with overlay
- Side-by-side or toggle view for languages
- Right-to-left text rendering for Urdu
- Mode switching capability
- Responsive design

---

### FloatingTranslationButton.tsx
**Location**: `frontend/src/components/Reader/FloatingTranslationButton.tsx`

**Purpose**: Floating button that appears when text is selected, allowing users to translate the selected text.

**Props**:
- `isVisible`: boolean - Whether the button should be displayed
- `onClick`: Function - Callback when the button is clicked
- `selectedText`: string | undefined - The text currently selected

**Features**:
- Fixed positioning in the bottom-right corner
- Visible only when text is selected and panel is closed
- Accessible via keyboard navigation
- Smooth animations

---

### LanguageToggle.tsx
**Location**: `frontend/src/components/Reader/LanguageToggle.tsx`

**Purpose**: Component to toggle between English and Urdu views.

**Props**:
- `className`: string - Additional CSS classes

**Features**:
- Cycles through language display modes (English, Urdu, Both)
- Updates the current language preference
- Accessible button with appropriate labels

---

## Layout Components

### SettingsPanel.tsx
**Location**: `frontend/src/components/Layout/SettingsPanel.tsx`

**Purpose**: Modal panel for adjusting user preferences.

**Props**:
- `isVisible`: boolean - Whether the panel should be displayed
- `onClose`: Function - Callback when the panel is closed

**Features**:
- Theme selection (light/dark)
- Font size slider
- Language preference selection
- Form for user settings
- Responsive design

---

## Book Components

### BookList.tsx
**Location**: `frontend/src/components/Books/BookList.tsx`

**Purpose**: Lists available books with search and filtering capabilities.

**Props**:
- `onBookSelect`: Function - Callback when a book is selected

**State**:
- `books`: Book[] - List of available books
- `loading`: boolean - Whether books are loading
- `error`: string | null - Error message if loading failed

**Features**:
- Fetches books from the API
- Displays books in a responsive grid
- Handles loading and error states
- Calls back when a book is selected

---

### BookCard.tsx
**Location**: `frontend/src/components/Books/BookCard.tsx`

**Purpose**: Displays information about a single book in an attractive card format.

**Props**:
- `book`: Book - The book to display
- `onClick`: Function - Callback when the card is clicked

**Features**:
- Displays book title and author
- Shows translation availability indicator
- Character count information
- Hover effects
- Responsive design

---

## UI Components

### Button.tsx
**Location**: `frontend/src/components/UI/Button.tsx`

**Purpose**: Reusable button component with different variants and sizes.

**Props**:
- `children`: ReactNode - Button content
- `onClick`: Function - Click handler
- `variant`: 'primary' | 'secondary' | 'outline' - Button style variant
- `size`: 'sm' | 'md' | 'lg' - Button size
- `className`: string - Additional CSS classes
- `disabled`: boolean - Whether the button is disabled
- `type`: 'button' | 'submit' | 'reset' - Button type

**Features**:
- Multiple style variants
- Different size options
- Disabled state handling
- Accessible attributes

---

### Slider.tsx
**Location**: `frontend/src/components/UI/Slider.tsx`

**Purpose**: Reusable range slider component, primarily used for font size adjustment.

**Props**:
- `value`: number - Current value of the slider
- `min`: number - Minimum value
- `max`: number - Maximum value
- `step`: number - Step increment (default 1)
- `onChange`: Function - Callback when value changes
- `label`: string - Label for the slider
- `className`: string - Additional CSS classes

**Features**:
- Customizable min/max values
- Step controls
- Value display
- Accessible labels
- Responsive design

---

## Page Components

### ReaderPage.tsx
**Location**: `frontend/src/pages/ReaderPage.tsx`

**Purpose**: Main page for the reading experience, combines all reading components.

**Props**:
- `bookId`: string - ID of the book to display

**State**:
- `book`: Book | null - The loaded book
- `selectedText`: string - Currently selected text
- `showTranslationPanel`: boolean - Whether to show translation panel
- `displayMode`: 'side_by_side' | 'toggle' - Language display mode

**Features**:
- Loads book content by ID
- Manages translation panel state
- Handles paragraph selection
- Coordinates all reading components
- Implements translation functionality

---

### LibraryPage.tsx
**Location**: `frontend/src/pages/LibraryPage.tsx`

**Purpose**: Page for browsing available books.

**Props**:
- `onBookSelect`: Function - Callback when a book is selected

**Features**:
- Displays book list
- Handles book selection
- Provides navigation to reader page

---

### SettingsPage.tsx
**Location**: `frontend/src/pages/SettingsPage.tsx`

**Purpose**: Page for managing user settings.

**Features**:
- Access to settings panel
- Additional account settings
- About information
- Consistent with app layout

---

## Hooks

### useBookContent.ts
**Location**: `frontend/src/hooks/useBookContent.ts`

**Purpose**: Custom hook to manage loading and state of book content.

**Parameters**:
- `bookId`: string - ID of the book to load

**Returns**:
- `book`: Book | null - Loaded book
- `loading`: boolean - Loading state
- `error`: string | null - Error state

**Features**:
- Fetches book content from API
- Manages loading and error states
- Handles book caching
- Implements retry logic

---

### useTranslation.ts
**Location**: `frontend/src/hooks/useTranslation.ts`

**Purpose**: Custom hook to manage translation functionality.

**Returns**:
- `translation`: string - Translated text
- `loading`: boolean - Translation loading state
- `error`: string | null - Error state
- `translateText`: Function - Function to translate text
- `resetTranslation`: Function - Function to reset translation

**Features**:
- Translates text via API
- Manages translation states
- Provides error handling
- Implements caching

---

### useUserPreferences.ts
**Location**: `frontend/src/hooks/useUserPreferences.ts`

**Purpose**: Custom hook to manage user preferences.

**Returns**:
- `userPreferences`: UserPreferences | null - Current preferences
- `loading`: boolean - Loading state
- `error`: string | null - Error state
- `updatePreferences`: Function - Function to update preferences

**Features**:
- Loads preferences from storage/API
- Manages preference states
- Updates preferences and syncs to storage/API
- Provides error handling

---

## Services

### apiClient.ts
**Location**: `frontend/src/services/apiClient.ts`

**Purpose**: Generic API client for making HTTP requests.

**Features**:
- Configurable base URL
- Error handling
- Request/response interceptors
- Support for different HTTP methods

### bookService.ts
**Location**: `frontend/src/services/bookService.ts`

**Purpose**: Service for book-related API operations.

**Functions**:
- `getBooks()`: Get list of books
- `getBookContent(bookId)`: Get specific book content
- `uploadBook(bookData)`: Upload a new book
- `getUserPreferences()`: Get user preferences from API
- `updateUserPreferences(preferences)`: Update user preferences via API

### translationService.ts
**Location**: `frontend/src/services/translationService.ts`

**Purpose**: Service for translation-related operations.

**Functions**:
- `getTranslation(text, bookId)`: Get Urdu translation for text

### storageService.ts
**Location**: `frontend/src/services/storageService.ts`

**Purpose**: Service for local storage operations, with fallback to IndexedDB.

**Functions**:
- `saveUserPreferences(preferences)`: Save user preferences to storage
- `getUserPreferences()`: Get user preferences from storage
- `saveBookContent(bookId, content)`: Save book content to storage
- `getBookContent(bookId)`: Get book content from storage

---

## Models

### Book.ts
**Location**: `frontend/src/models/Book.ts`

**Purpose**: Type definitions for Book entities.

### Paragraph.ts
**Location**: `frontend/src/models/Paragraph.ts`

**Purpose**: Type definitions for Paragraph entities.

### UserPreferences.ts
**Location**: `frontend/src/models/UserPreferences.ts`

**Purpose**: Type definitions for UserPreferences entities.

### Translation.ts
**Location**: `frontend/src/models/Translation.ts`

**Purpose**: Type definitions for Translation entities.

---

## Styling

### Typography CSS
**Location**: `frontend/src/styles/typography.css`

**Purpose**: Typography styles optimized for reading experience.

**Features**:
- Appropriate font families for English and Urdu
- Line height and spacing for readability
- Responsive font sizing
- Dark mode considerations

### Global CSS
**Location**: `frontend/src/styles/globals.css`

**Purpose**: Global styles for the application.

**Features**:
- Reset/normalize styles
- Base element styles
- Scrollbar customization
- Dark mode classes
- Responsive utility classes

---

## Type Definitions

### index.ts
**Location**: `frontend/src/types/index.ts`

**Purpose**: Centralized type definitions used throughout the application.

**Types Defined**:
- BookFormat
- Theme
- LanguagePreference
- ReadingMode
- TranslationDisplayMode
- `ApiResponse<T>`
- ApiError