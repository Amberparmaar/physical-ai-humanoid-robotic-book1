# API Documentation - BookReader with Urdu Translation

## Base URL
All API requests should be made to the base URL followed by the specific endpoint.

## Authentication
Some endpoints may require authentication via an API token in the request header:
```
Authorization: Bearer <your-token>
```

## Content Types
- Request bodies should be in JSON format
- Set the appropriate content type header: `Content-Type: application/json`
- Responses will be in JSON format

---

## Books API

### Get List of Available Books
```
GET /api/books
```

**Description**: Retrieve a list of all available books in the system

**Response**:
```
Status: 200 OK
Content-Type: application/json
```
````
[
  {
    "id": "string",
    "title": "string",
    "author": "string",
    "format": "plain_text | pdf | epub",
    "language": "string",
    "hasUrduTranslation": "boolean",
    "length": "integer"
  }
]
````

**Example Request**:
````
curl -X GET "https://api.bookreader.com/api/books"
````

### Get Book Content
```
GET /api/books/{bookId}
```

**Description**: Retrieve the content of a specific book

**Path Parameters**:
- `bookId` (required): Unique identifier for the book

**Response**:
```
Status: 200 OK
Content-Type: application/json
```
````
{
  "id": "string",
  "title": "string",
  "author": "string",
  "paragraphs": [
    {
      "id": "string",
      "originalText": "string",
      "urduTranslation": "string",
      "position": "number",
      "bookId": "string"
    }
  ],
  "metadata": {
    "format": "plain_text | pdf | epub",
    "language": "string",
    "length": "integer",
    "hasUrduTranslation": "boolean"
  }
}
````

**Example Request**:
````
curl -X GET "https://api.bookreader.com/api/books/12345"
````

### Upload New Book
```
POST /api/books
```

**Description**: Add a new book to the system with English content and optional Urdu translation

**Request Body**:
````
{
  "title": "string",
  "author": "string",
  "format": "plain_text | pdf | epub",
  "originalText": "string",
  "urduTranslation": "string (optional)"
}
````

**Response**:
```
Status: 201 Created
Content-Type: application/json
```
````
{
  "id": "string",
  "title": "string",
  "author": "string",
  "originalText": "string",
  "urduTranslation": "string",
  "format": "plain_text | pdf | epub",
  "language": "string",
  "hasUrduTranslation": "boolean",
  "length": "integer",
  "paragraphs": [
    {
      "id": "string",
      "originalText": "string",
      "urduTranslation": "string",
      "position": "number",
      "bookId": "string"
    }
  ],
  "metadata": {
    "format": "plain_text | pdf | epub",
    "language": "string",
    "length": "integer",
    "hasUrduTranslation": "boolean"
  }
}
````

**Example Request**:
````
curl -X POST "https://api.bookreader.com/api/books"
  -H "Content-Type: application/json"
  -d '{
    "title": "Sample Book",
    "author": "Author Name",
    "format": "plain_text",
    "originalText": "This is the original English text.",
    "urduTranslation": "یہ اصل اردو ترجمہ ہے۔"
  }'
````

---

## Translation API

### Get Urdu Translation for Selected Text
```
POST /api/books/{bookId}/translate
```

**Description**: Get the Urdu translation of selected English text from a book

**Path Parameters**:
- `bookId` (required): Unique identifier for the book

**Request Body**:
````
{
  "text": "string (required)",
  "paragraphId": "string (optional)",
  "context": "string (optional)"
}
````

**Response**:
```
Status: 200 OK
Content-Type: application/json
```
````
{
  "originalText": "string",
  "urduTranslation": "string",
  "confidence": "number (0-1)"
}
````

**Example Request**:
````
curl -X POST "https://api.bookreader.com/api/books/12345/translate"
  -H "Content-Type: application/json"
  -d '{
    "text": "This is the text to translate"
  }'
````

---

## User Preferences API

### Get User Preferences
```
GET /api/user/preferences
```

**Description**: Retrieve current user preferences for font size, theme, and language

**Response**:
```
Status: 200 OK
Content-Type: application/json
```
````
{
  "fontSize": "integer (12-36)",
  "theme": "light | dark",
  "languagePreference": "english | urdu | both",
  "readingMode": "side_by_side | toggle"
}
````

**Example Request**:
````
curl -X GET "https://api.bookreader.com/api/user/preferences"
````

### Update User Preferences
```
PUT /api/user/preferences
```

**Description**: Update user preferences for font size, theme, and language

**Request Body**:
````
{
  "fontSize": "integer (12-36)",
  "theme": "light | dark",
  "languagePreference": "english | urdu | both",
  "readingMode": "side_by_side | toggle"
}
````

**Response**:
```
Status: 200 OK
Content-Type: application/json
```
````
{
  "fontSize": "integer",
  "theme": "light | dark",
  "languagePreference": "english | urdu | both",
  "readingMode": "side_by_side | toggle"
}
````

**Example Request**:
````
curl -X PUT "https://api.bookreader.com/api/user/preferences"
  -H "Content-Type: application/json"
  -d '{
    "fontSize": 18,
    "theme": "dark",
    "languagePreference": "both",
    "readingMode": "side_by_side"
  }'
````

---

## Reading Progress API

### Get Reading Progress
```
GET /api/books/{bookId}/progress
```

**Description**: Retrieve user's progress in a specific book

**Path Parameters**:
- `bookId` (required): Unique identifier for the book

**Response**:
```
Status: 200 OK
Content-Type: application/json
```
````
{
  "bookId": "string",
  "currentParagraphId": "string",
  "position": "integer (percentage)",
  "lastReadAt": "string (date-time)"
}
````

**Example Request**:
````
curl -X GET "https://api.bookreader.com/api/books/12345/progress"
````

### Update Reading Progress
```
PUT /api/books/{bookId}/progress
```

**Description**: Update user's progress in a specific book

**Path Parameters**:
- `bookId` (required): Unique identifier for the book

**Request Body**:
````
{
  "currentParagraphId": "string",
  "position": "integer (percentage)"
}
````

**Response**:
```
Status: 200 OK
Content-Type: application/json
```
````
{
  "bookId": "string",
  "currentParagraphId": "string",
  "position": "integer (percentage)",
  "lastReadAt": "string (date-time)"
}
````

**Example Request**:
````
curl -X PUT "https://api.bookreader.com/api/books/12345/progress"
  -H "Content-Type: application/json"
  -d '{
    "currentParagraphId": "paragraph-789",
    "position": 45
  }'
````

---

## Error Handling

### Common Error Responses

**400 Bad Request**:
````
{
  "success": false,
  "error": "string",
  "message": "string"
}
````

**401 Unauthorized**:
````
{
  "success": false,
  "error": "Unauthorized",
  "message": "Authentication token is missing or invalid"
}
````

**404 Not Found**:
````
{
  "success": false,
  "error": "Not Found",
  "message": "The requested resource was not found"
}
````

**500 Internal Server Error**:
````
{
  "success": false,
  "error": "Internal Server Error",
  "message": "An unexpected error occurred"
}
````

---

## Rate Limiting

To ensure fair usage of the API, rate limiting is applied to all endpoints:
- 1000 requests per hour per IP address
- 100 requests per hour per authenticated user

When rate limits are exceeded, the API will return:
```
Status: 429 Too Many Requests
X-RateLimit-Reset: timestamp (when the current rate limit window resets)
```