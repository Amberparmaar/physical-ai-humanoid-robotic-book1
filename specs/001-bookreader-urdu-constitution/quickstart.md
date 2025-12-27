# Quickstart Guide: BookReader UI with Urdu Translation

## Overview
This guide will help you get started with the BookReader application featuring Urdu translation. The application provides a clean reading experience with the ability to translate selected text to Urdu.

## Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd bookreader-app
   ```

2. Install dependencies:
   ```
   npm install
   # or
   yarn install
   ```

3. Set up environment variables (create a `.env` file in the root):
   ```
   REACT_APP_TRANSLATION_API_KEY=your_translation_api_key
   REACT_APP_TRANSLATION_API_URL=your_translation_api_url
   ```

## Running the Application
1. Start the development server:
   ```
   npm start
   # or
   yarn start
   ```

2. Open your browser and navigate to `http://localhost:3000`

## Key Features
1. **Reading Experience**:
   - Clean, minimalist interface with smooth typography
   - Responsive design for all device sizes
   - Distraction-free reading environment

2. **Urdu Translation**:
   - Select any paragraph to translate to Urdu
   - Floating translation button to access Urdu translations
   - Toggle between English and Urdu or view both languages side-by-side

3. **Customization Options**:
   - Adjust font size with the font size control
   - Switch between light and dark themes
   - Choose language preference (English, Urdu, or both)

## How to Use
1. **Browse Books**: Navigate to the books section to view available titles
2. **Select a Book**: Click on a book to start reading
3. **Translate Text**: 
   - Highlight a paragraph
   - Click the floating translation button
   - View the Urdu translation in the slide-out panel
4. **Adjust Settings**:
   - Use the settings icon to adjust font size
   - Toggle between light and dark themes
   - Choose language display preferences
5. **Save Progress**: Your reading progress is automatically saved

## Development
To run the application in development mode with hot reloading:
```
npm run dev
# or
yarn dev
```

## Testing
Run the test suite:
```
npm test
# or
yarn test
```

## Building for Production
To create an optimized production build:
```
npm run build
# or
yarn build
```

## API Endpoints
Key API endpoints used by the application:
- `GET /api/books` - Get list of available books
- `GET /api/books/{bookId}` - Get content of a specific book
- `POST /api/books/{bookId}/translate` - Get Urdu translation for selected text
- `GET/PUT /api/user/preferences` - Manage user preferences
- `GET/PUT /api/books/{bookId}/progress` - Track reading progress

For complete API documentation, refer to the OpenAPI specification in the `contracts/openapi.yaml` file.