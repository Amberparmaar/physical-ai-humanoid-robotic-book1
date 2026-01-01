# Data Model: BookReader UI with Urdu Translation

## Entity: Reading Content

### Description
The book text being displayed in the reading area; includes both English original and Urdu translations

### Fields
- `id`: string (unique identifier for the content)
- `title`: string (title of the book/document)
- `author`: string (author of the book/document)
- `originalText`: string (the English text content)
- `urduTranslation`: string (Urdu translation of the original text)
- `format`: enum ['plain_text', 'pdf', 'epub'] (format of the original content)
- `length`: number (length of the content in characters)
- `paragraphs`: array of objects (structured paragraphs with corresponding translations)

### Validation Rules
- `title` and `author` are required
- `originalText` and `urduTranslation` must be provided together
- `format` must be one of the supported formats
- `paragraphs` must have corresponding translations for each paragraph

### Relationships
- Contains multiple `Paragraph` entities
- Associated with multiple `UserPreferences` (for different users)

## Entity: Paragraph

### Description
A paragraph of text with its Urdu translation, used for translation functionality

### Fields
- `id`: string (unique identifier for the paragraph)
- `originalText`: string (English paragraph text)
- `urduTranslation`: string (Urdu translation of the paragraph)
- `position`: number (position in the content)
- `bookId`: string (reference to the Reading Content)

### Validation Rules
- `originalText` and `urduTranslation` must be present
- `position` must be a positive integer
- `bookId` must reference a valid Reading Content

### Relationships
- Belongs to one `Reading Content`
- Connected to one `Translation Panel` (when active)

## Entity: User Preferences

### Description
Settings including font size, theme (dark/light mode), and language preference that persist across sessions

### Fields
- `id`: string (unique identifier for user preferences)
- `userId`: string (identifier for the user, 'anonymous' for non-registered users)
- `fontSize`: number (font size in pixels, default: 16)
- `theme`: enum ['light', 'dark'] (current theme setting, default: 'light')
- `languagePreference`: enum ['english', 'urdu', 'both'] (default: 'english')
- `lastReadingPosition`: object (track last reading position: {bookId, paragraphId})
- `readingMode`: enum ['side_by_side', 'toggle'] (how to display translations, default: 'toggle')

### Validation Rules
- `fontSize` must be between 12 and 36 pixels
- `theme` must be either 'light' or 'dark'
- `languagePreference` must be one of the allowed values

### Relationships
- Associated with one or more `Reading Content` entities

## Entity: Translation Panel

### Description
The UI component that displays the Urdu translation when activated, either as a sliding panel or popup

### Fields
- `id`: string (unique identifier for the panel instance)
- `paragraphId`: string (reference to the paragraph being translated)
- `isVisible`: boolean (whether the panel is currently showing)
- `displayMode`: enum ['popup', 'side_panel'] (how the translation is displayed)
- `userPreferencesId`: string (reference to user preferences affecting display)

### Validation Rules
- `paragraphId` must reference a valid paragraph
- `displayMode` must be one of the allowed values

### Relationships
- Connected to one `Paragraph` entity
- Associated with one `User Preferences` entity

## State Transitions

### Reading Content
- `loading` → `ready` when content is loaded
- `ready` → `reading` when user starts reading
- `reading` → `paused` when user pauses
- `reading` → `completed` when user finishes

### User Preferences
- `default` → `customized` when user changes any preference
- `customized` → `updated` when preferences are saved