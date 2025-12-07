# Feature Specification: AI-native textbook: Physical AI & Humanoid Robotics

**Feature Branch**: `1-ai-native-textbook`
**Created**: 2025-12-07
**Status**: Draft
**Input**: User description: Produce a detailed specification document (Markdown) for "AI-native textbook: Physical AI & Humanoid Robotics" that implements the constitution. Output must include these sections (use exact headings): - Course details - Module list (ROS2, Gazebo & Unity, NVIDIA Isaac, VLA) with 1-paragraph module goals + 3 measurable learning outcomes each - Quarter overview & weekly plan (Weeks 1–13) as a markdown table - Assessments (deliverables, grading rubric) - Hardware requirements (table: name, role, min-spec, recommended, price-est.) - RAG chatbot design: architecture diagram (textual ASCII), data flows, FastAPI endpoints, Neon DB schema for users, Qdrant index schema (collection name, vector_dim, metadata fields) - Integration details: OpenAI model choices (embeddings + LLM), caching policy, rate limits - Bonus features mapping (explain precisely how to earn extra 50 points each for: Claude Code subagents, Better-Auth signup, chapter personalization button, Urdu translation button) - Deployment checklist for GitHub Pages (branch names, build matrix) - Acceptance criteria: list of tests (unit, integration, e2e) that must pass before submission Include concrete user stories for educators and students (at least 3 each). Keep output under 2,000 tokens. Integration of a RAG chatbot using: OpenAI Agents Context7 MCP + Context7 API (required for embeddings, memory, selected-text search, and agent actions) FastAPI Neon Postgres Qdrant

## User Scenarios & Testing *(mandatory)*

### Student User Stories:
- As a student, I want to interact with an AI tutor to receive personalized learning assistance for complex robotics concepts
- As a student, I want to access the textbook content in multiple languages to enhance my understanding of robotics topics
- As a student, I want to track my learning progress across different modules to identify areas for improvement

### Educator User Stories:
- As an educator, I want to customize textbook content to match my course curriculum and student needs
- As an educator, I want to monitor student engagement and comprehension through analytics
- As an educator, I want to integrate additional teaching materials that complement the core robotics textbook

### Independent Test:

For the AI-native textbook, independent testing can be conducted by verifying that students can successfully navigate the textbook interface, interact with the RAG chatbot, and complete module assessments. This delivers the specific value of demonstrating a functional AI-enhanced learning platform.

#### Acceptance Scenarios:

1. **Given** student accesses the AI-native textbook, **When** they open any module content, **Then** they can view well-formatted text, images, and interactive elements
2. **Given** student has a question about robotics concepts, **When** they interact with the RAG chatbot, **Then** they receive accurate and contextual responses based on textbook content
3. **Given** student wants to switch languages, **When** they select Urdu translation option, **Then** the content renders accurately in Urdu

---

### Edge Cases

- What happens when the RAG chatbot receives queries outside the textbook scope?
- How does the system handle simultaneous users during peak traffic periods?
- What occurs when hardware simulation fails during practical exercises?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide interactive textbook content covering Physical AI and Humanoid Robotics topics
- **FR-002**: System MUST integrate a RAG chatbot that responds to queries with context from textbook content
- **FR-003**: Students MUST be able to access content in both English and Urdu languages
- **FR-004**: System MUST track user learning progress across modules
- **FR-005**: System MUST provide hardware simulation capabilities through ROS2, Gazebo, NVIDIA Isaac, and VLA
- **FR-006**: System MUST support personalized chapter recommendations based on user progress
- **FR-007**: Educators MUST be able to customize content for their specific courses
- **FR-008**: System MUST implement secure user authentication via Better-Auth

### Key Entities

- **User**: Individual interacting with the textbook system (student, educator, admin) with profile, preferences and progress tracking
- **Module**: Educational unit containing content, learning objectives, assessments, and related resources
- **ChatSession**: Interactive conversation between user and RAG chatbot with history and context
- **LearningProgress**: Record of user's progress, scores and engagement across different modules

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Students can complete 90% of module assessments with passing grades (70%+) within 12 weeks
- **SC-002**: RAG chatbot provides accurate and helpful responses to 85% of student queries with 95% relevance
- **SC-003**: System supports at least 1000 concurrent users during peak usage times with minimal performance degradation
- **SC-004**: 95% of users report improved understanding of robotics concepts after using the AI tutor feature

# Course Details

**Course Title**: Physical AI & Humanoid Robotics: An AI-Native Textbook
**Course Duration**: 13 weeks (Quarter system)
**Target Audience**: Undergraduate/Graduate students in AI, Robotics, Computer Science, or Engineering programs
**Prerequisites**: Basic programming skills (Python), Linear Algebra, Calculus, introductory robotics concepts
**Course Level**: Advanced undergraduate / Introductory graduate
**Credit Hours**: 3 credit hours (3 lecture hours, 2 lab hours per week)

This AI-native textbook transforms traditional robotics education by integrating interactive AI tools, personalized learning paths, and multimodal content. Students engage with cutting-edge frameworks including ROS2, Gazebo, NVIDIA Isaac, and Vision Language Action (VLA) models while building humanoid robot applications. The course emphasizes hands-on learning through simulated environments and real-world case studies, enhanced by a RAG-powered chatbot for instant Q&A support.

# Module List

## Module 1: ROS2 (Robot Operating System 2)

This module introduces students to the fundamental concepts of ROS2, the latest generation of the Robot Operating System. Students will learn how to create nodes, manage topics and services, and implement distributed computing principles for robotic systems. The focus is on building robust, scalable robotic applications with real-time communication between different components.

1. Students will be able to create and configure ROS2 nodes, packages, and workspaces for humanoid robotics applications
2. Students will demonstrate proficiency in using ROS2 communication patterns including topics, services, and actions
3. Students will implement distributed robotic systems using ROS2 middleware for multi-robot coordination

## Module 2: Gazebo & Unity Simulation Environments

This module covers the use of simulation environments for humanoid robot development. Students will learn to create realistic 3D models of humanoid robots, set up physics-based simulation environments, and test robot behaviors in safe virtual spaces. The module emphasizes the importance of sim-to-real transfer and validation of robotic algorithms in controlled environments.

1. Students will create detailed 3D models of humanoid robots with accurate kinematic properties
2. Students will design and implement physics-based simulation scenarios that accurately reflect real-world conditions
3. Students will validate robotic algorithms in simulation with at least 80% correlation to real-world performance

## Module 3: NVIDIA Isaac Platform

This module focuses on the NVIDIA Isaac platform for AI-powered robotics. Students will explore GPU-accelerated computing for robotics, learn to develop perception and control algorithms using Isaac SDK, and implement AI models for navigation, manipulation, and decision-making in humanoid robots.

1. Students will develop perception algorithms using Isaac's perception libraries for object detection and scene understanding
2. Students will implement GPU-accelerated control systems for real-time humanoid robot operation
3. Students will deploy AI models on Isaac-based platforms with optimization for computational efficiency

## Module 4: Vision Language Action (VLA) Models

This module explores the integration of vision, language, and action models in humanoid robotics. Students will learn about multimodal AI systems that enable robots to perceive their environment, understand natural language commands, and execute complex tasks by combining perception and reasoning.

1. Students will implement vision-language models to enable robots to understand and execute natural language commands
2. Students will develop AI systems that combine visual perception with action planning for complex tasks
3. Students will evaluate the performance of VLA models in various humanoid robot scenarios with benchmark metrics

# Quarter Overview & Weekly Plan

| Week | Module | Topics | Learning Activities | Assessment |
|------|--------|--------|-------------------|------------|
| 1 | Course Introduction | Overview of Physical AI & Humanoid Robotics, textbook navigation, AI tutor introduction | Course orientation, textbook exploration | Pre-assessment quiz |
| 2 | Module 1 - ROS2 | ROS2 architecture, nodes, topics, services | Textbook reading, ROS2 workspace setup | ROS2 basic concepts quiz |
| 3 | Module 1 - ROS2 | Actions, launch files, parameters | Hands-on ROS2 tutorials in simulation | ROS2 package implementation |
| 4 | Module 1 - ROS2 | ROS2 middleware, real-time systems | Practical exercises, debugging techniques | ROS2 system design project |
| 5 | Module 2 - Gazebo & Unity | Physics simulation, 3D modeling | Simulation environment setup | 3D model creation assignment |
| 6 | Module 2 - Gazebo & Unity | Simulation physics, sensor modeling | Physics-based simulation tasks | Simulation validation report |
| 7 | Module 2 - Gazebo & Unity | Sim-to-real transfer, validation | Simulated robot control tasks | Performance correlation analysis |
| 8 | Module 3 - NVIDIA Isaac | Isaac platform architecture, perception | Isaac SDK setup, perception algorithms | Perception system implementation |
| 9 | Module 3 - NVIDIA Isaac | GPU-accelerated computing, control | Isaac control system implementation | GPU-accelerated algorithm project |
| 10 | Module 3 - NVIDIA Isaac | Isaac deployment, optimization | Optimization techniques, deployment | Isaac system optimization |
| 11 | Module 4 - VLA Models | Vision-language models, multimodal AI | VLA model implementation | VLA model design project |
| 12 | Module 4 - VLA Models | Action planning, execution | Complex task implementation | VLA integration project |
| 13 | Integration & Review | Course synthesis, final project | Final project development | Final project demonstration and report |

# Assessments

## Deliverables

### Module Projects (60% of final grade)
- **ROS2 Implementation Project**: Students will create a distributed robotic system with multiple nodes communicating via ROS2. (15%)
- **Simulation Validation Project**: Students will design and validate a humanoid robot in Gazebo/Unity with physics-based simulation. (15%)
- **Isaac AI Project**: Students will implement a GPU-accelerated perception and control system on the Isaac platform. (15%)
- **VLA Integration Project**: Students will create a multimodal AI system combining vision, language, and action for humanoid robotics. (15%)

### Weekly Assessments (25% of final grade)
- **Quizzes**: Short assessments at the end of each week to evaluate understanding of key concepts. (10%)
- **Practical Exercises**: Hands-on tasks that apply concepts learned in each module. (10%)
- **Peer Reviews**: Evaluation of fellow students' projects and contributions to class discussions. (5%)

### Final Project (15% of final grade)
- **Integrated Robotics System**: Students will create a complete humanoid robotics application that integrates all four modules, demonstrating proficiency in ROS2, simulation, Isaac platform, and VLA models.

## Grading Rubric

| Assessment Type | Criteria | Excellent (A) | Proficient (B) | Developing (C) | Beginning (D) |
|----------------|----------|---------------|----------------|----------------|---------------|
| Project Implementation | Technical Execution | Implementation exceeds requirements with advanced features | Implementation meets all requirements effectively | Implementation meets basic requirements with minor issues | Implementation has significant technical deficiencies |
| Project Implementation | Documentation & Presentation | Exceptional documentation with clear explanations and professional presentation | Good documentation with clear explanations | Basic documentation with adequate explanations | Minimal documentation with unclear explanations |
| Project Implementation | Innovation & Problem Solving | Creative solutions with evidence of deep understanding | Effective problem-solving with solid understanding | Adequate solutions with basic understanding | Limited problem-solving with minimal understanding |
| Weekly Quizzes | Conceptual Understanding | Demonstrates comprehensive understanding of concepts | Demonstrates strong understanding of concepts | Demonstrates adequate understanding of concepts | Demonstrates limited understanding of concepts |
| Practical Exercises | Application of Skills | Skillfully applies learned concepts with independence | Applies concepts with occasional guidance | Applies concepts with significant guidance | Struggles to apply concepts even with guidance |
| Final Project | Integration | Seamlessly integrates all four modules with advanced functionality | Successfully integrates all four modules | Integrates most modules with minor gaps | Struggles to integrate modules effectively |

# Hardware Requirements

| Name | Role | Min-Spec | Recommended | Price-Est. |
|------|------|----------|-------------|------------|
| GPU | AI/ML Processing for Isaac & VLA Models | NVIDIA RTX 3060 (12GB VRAM) | NVIDIA RTX 4080 (16GB VRAM) | $300-$1,200 |
| CPU | General Processing & Simulation | Intel i5-10400 / AMD Ryzen 5 3600 | Intel i7-12700K / AMD Ryzen 7 5800X | $200-$400 |
| RAM | Memory for Simulation & AI Models | 16 GB DDR4 | 32 GB DDR4 | $70-$150 |
| Storage | OS, Software & Project Files | 500GB SSD | 1TB NVMe SSD | $50-$100 |
| Motherboard | System Platform | B450 / B560 chipset | B550 / Z690 chipset | $80-$200 |
| PSU | Power Supply | 650W 80+ Bronze | 750W 80+ Gold | $60-$120 |
| Robot Platform | Physical Testing Platform | Basic wheeled robot | Humanoid robot (Trossen REEM-C, PAL Robotics TALOS) | $2,000-$15,000 |
| Sensors | Perception System | Basic IMU, Camera | LIDAR, RGB-D Camera, Force Sensors | $200-$1,000 |
| Networking | High-speed communication | Gigabit Ethernet | 10GbE + WiFi 6 | $50-$150 |
| Display | Development & Simulation | 1080p, 60Hz | 1440p, 144Hz | $200-$400 |

# RAG Chatbot Design

## Architecture Diagram (Textual ASCII)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Textbook      │    │   Context7       │    │   Vector DB     │
│   Content       │───▶│   MCP + API      │───▶│   (Qdrant)      │
│                 │    │                  │    │                 │
│ (Chapters,      │    │ - Embeddings     │    │ - Embedding     │
│  Lessons,       │    │ - Memory Mgmt    │    │   Index         │
│  Examples)      │    │ - Selected-text  │    │ - Metadata      │
│                 │    │   Search         │    │   Storage       │
└─────────────────┘    │ - Agent Actions  │    └─────────────────┘
                       └──────────────────┘            │
                              │                       │
                              ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   Neon DB       │
│   (Docusaurus)  │◀──▶│   Backend        │◀──▶│   (User Data)   │
│                 │    │                  │    │                 │
│ - Textbook UI   │    │ - Authentication │    │ - User profiles │
│ - Chat Interface│    │ - Chat Endpoints │    │ - Preferences   │
│ - User Profiles │    │ - Content APIs   │    │ - Learning      │
└─────────────────┘    │ - Rate Limiting  │    │   Progress      │
                       └──────────────────┘    └─────────────────┘
```

## Data Flows

1. **Content Ingestion Flow**:
   - Textbook content is extracted and processed
   - Content is chunked into semantic segments
   - Embeddings are generated using OpenAI's text-embedding model
   - Vector representations with metadata are stored in Qdrant

2. **Query Processing Flow**:
   - User query comes through the frontend
   - Query is embedded using the same embedding model
   - Vector search retrieves relevant context from Qdrant
   - Context is combined with original query to form RAG prompt
   - GPT model generates response based on retrieved context

3. **User Interaction Flow**:
   - User authentication managed by Better-Auth
   - User preferences and learning progress stored in Neon DB
   - Chat history and conversation context maintained in system
   - Response logged for analytics and improvement

## FastAPI Endpoints

```
POST /api/chat
- Input: {query: str, user_id: str, context?: str}
- Output: {response: str, sources: List[str], timestamp: str}
- Function: Process user query using RAG system

GET /api/user/{user_id}/progress
- Output: {modules_completed: List[str], scores: Dict[str, float], time_spent: int}
- Function: Retrieve user learning progress

POST /api/user/{user_id}/preferences
- Input: {lang_pref: str, personalization_enabled: bool, ...}
- Output: {status: str}
- Function: Update user preferences

GET /api/content/search
- Input: {query: str, filters?: Dict[str, str]}
- Output: {results: List[Dict[str, Any]]}
- Function: Semantic search across textbook content

POST /api/auth/login
- Input: {email: str, password: str}
- Output: {token: str, user_id: str}
- Function: User authentication with Better-Auth
```

## Neon DB Schema for Users

```
Table: users
- id: UUID (primary key)
- email: VARCHAR(255) (unique, not null)
- name: VARCHAR(255)
- password_hash: VARCHAR(255) (not null)
- created_at: TIMESTAMP (default: now())
- updated_at: TIMESTAMP
- role: VARCHAR(50) (default: 'student')
- preferences: JSONB (default: {})
- last_login: TIMESTAMP

Table: learning_progress
- id: UUID (primary key)
- user_id: UUID (foreign key: users.id)
- module_id: VARCHAR(100) (not null)
- content_completed: INTEGER (default: 0)
- score: DECIMAL(5,2)
- time_spent: INTEGER (in seconds)
- last_accessed: TIMESTAMP
- created_at: TIMESTAMP (default: now())
- updated_at: TIMESTAMP

Table: chat_history
- id: UUID (primary key)
- user_id: UUID (foreign key: users.id)
- session_id: UUID
- query: TEXT (not null)
- response: TEXT (not null)
- timestamp: TIMESTAMP (default: now())
- sources: JSONB
```

## Qdrant Index Schema

```
Collection: textbook_content
- Collection name: textbook_content
- Vector dimension: 1536 (for OpenAI embeddings)
- Metadata fields:
  - content_id: keyword (unique identifier for each content piece)
  - module: keyword (module name: "ROS2", "Gazebo", "Isaac", "VLA")
  - section: keyword (specific section within module)
  - page_number: integer (original textbook page)
  - topic: keyword (specific topic covered)
  - difficulty_level: keyword (beginner, intermediate, advanced)
  - content_type: keyword (text, image_caption, code_snippet, example)
  - language: keyword (en, ur)
  - created_at: integer (timestamp)
```

# Integration Details

## OpenAI Model Choices

- **Embeddings Model**: OpenAI text-embedding-3-large (1536 dimensions)
  - Used for converting textbook content and user queries to vector representations
  - Provides high-quality semantic embeddings for accurate retrieval

- **LLM Model**: OpenAI GPT-4 Turbo
  - Used for generating responses based on retrieved context
  - Balances performance and cost for educational applications
  - Supports 128K context window, important for complex robotics discussions

## Caching Policy

- **Query Results Cache**: Redis-based caching for frequently asked questions
  - TTL: 4 hours for general questions, 24 hours for fundamental concepts
  - Stores pre-computed answers to common queries to reduce response time
  - Cache invalidation when underlying textbook content is updated

- **Embedding Cache**: Local cache for recently computed embeddings
  - TTL: 1 hour for query embeddings
  - Reduces API calls for repeated or similar queries

- **Session Context Cache**: In-memory cache for conversation history
  - TTL: 30 minutes of inactivity
  - Maintains context for multi-turn conversations
  - Cleared when users log out

## Rate Limits

- **Per-User Limits**:
  - 100 chat requests per hour for free tier users
  - 500 chat requests per hour for premium users
  - 10 requests per minute burst limit (sliding window)

- **System-Wide Limits**:
  - 1000 embedding requests per minute (sliding window)
  - 500 LLM generation requests per minute (sliding window)
  - 2000 content search requests per minute (sliding window)

- **Content Processing Limits**:
  - Max 100 pages per content chunking request
  - Max 50 files per batch upload for content ingestion

# Bonus Features Mapping

Each of the following bonus features is worth an additional 50 points when successfully implemented and demonstrated:

## Claude Code Subagents (50 bonus points)
Students can earn 50 bonus points by implementing autonomous AI subagents using Claude Code to solve complex robotics problems. This involves:
- Creating specialized agents for different robotics tasks (navigation, manipulation, perception)
- Implementing agent communication protocols using shared memory or message queues
- Demonstrating successful task completion through multi-agent collaboration
- Providing a detailed analysis of the subagents' performance and limitations

## Better-Auth Signup (50 bonus points)
Students can earn 50 bonus points by implementing advanced authentication and user management. This includes:
- Setting up secure OAuth2/OpenID Connect with multiple providers (Google, GitHub, etc.)
- Implementing role-based access control with different permissions for students, educators, and administrators
- Adding multi-factor authentication for enhanced security
- Creating user profile customization with preferences for personalized learning experiences

## Chapter Personalization Button (50 bonus points)
Students can earn 50 bonus points by developing an AI-driven content personalization system with:
- Machine learning algorithms that adapt content based on individual learning progress
- User preference settings that customize content difficulty and presentation style
- Dynamic assessment generation based on user performance patterns
- Progress tracking that adjusts learning paths in real-time based on user interactions

## Urdu Translation Button (50 bonus points)
Students can earn 50 bonus points by implementing comprehensive multilingual support with:
- Real-time translation between English and Urdu for all textbook content
- Preservation of technical terminology accuracy in both languages
- Cultural adaptation of examples and use cases for Urdu-speaking audience
- Voice synthesis in Urdu for accessibility features
- Quality assurance process to ensure translation accuracy and consistency

# Deployment Checklist for GitHub Pages

## Branch Names
- `main` - Production branch (GitHub Pages source)
- `develop` - Development branch
- `feature/*` - Feature-specific branches
- `release/*` - Release candidate branches
- `hotfix/*` - Urgent fixes to production

## Build Matrix Configuration

### GitHub Actions Workflow (`/.github/workflows/deploy.yml`)
- Node.js versions: 18.x, 20.x
- Operating systems: Ubuntu 22.04, Windows Server 2022
- Build environment: Production, Staging

### Required Environment Variables
- `OPENAI_API_KEY` - For embedding and LLM services
- `QDRANT_URL` - Vector database connection
- `NEON_DB_URL` - Postgres database connection
- `REACT_APP_CONTEXT7_API_KEY` - Context7 API access
- `GITHUB_TOKEN` - For GitHub Pages deployment

### Build Steps
- [ ] Install dependencies: `npm install`
- [ ] Build static assets: `npm run build`
- [ ] Run unit tests: `npm test`
- [ ] Verify API connectivity to external services
- [ ] Run integration tests
- [ ] Deploy to GitHub Pages

### Post-Deployment Verification
- [ ] Verify all pages load without errors
- [ ] Test RAG chatbot functionality
- [ ] Confirm user authentication works
- [ ] Validate content search functionality
- [ ] Confirm all modules content displays properly
- [ ] Verify assessment submission functionality
- [ ] Test multilingual support (English/Urdu)
- [ ] Validate personalization features

# Acceptance Criteria

## Unit Tests
- [ ] All textbook content modules load without errors (95% success rate)
- [ ] RAG chatbot correctly retrieves relevant information (85% accuracy in retrieval)
- [ ] User authentication functions correctly with 99.9% uptime
- [ ] Content search returns relevant results within 2 seconds (90% of queries)
- [ ] Assessment submission and grading system works correctly
- [ ] Progress tracking updates in real-time (within 5 seconds)

## Integration Tests
- [ ] End-to-end user registration and login flow completes successfully
- [ ] RAG system connects to all external APIs without errors
- [ ] Textbook content displays properly with all images and code snippets
- [ ] User preferences persist across sessions and devices
- [ ] Chatbot responses include appropriate source citations
- [ ] Assessment results sync with learning progress tracking

## End-to-End Tests
- [ ] New user can complete course registration and access first module within 5 minutes
- [ ] Student can navigate through all 4 modules and complete assessments
- [ ] Educator can customize content for their course requirements
- [ ] Student can interact with RAG chatbot and receive helpful responses
- [ ] Multilingual support switches between English and Urdu without errors
- [ ] Personalization features adapt content based on user progress
- [ ] All hardware simulation modules function correctly in browser
- [ ] Final project submission and evaluation process works end-to-end

# User Stories for Educators and Students

## Student User Stories:

1. As a student, I want to interact with an AI tutor to receive personalized learning assistance for complex robotics concepts, so that I can better understand challenging topics at my own pace.

2. As a student, I want to access the textbook content in multiple languages to enhance my understanding of robotics topics, so that language is not a barrier to learning advanced concepts.

3. As a student, I want to track my learning progress across different modules to identify areas for improvement, so that I can focus my efforts on the topics that need more attention.

## Educator User Stories:

1. As an educator, I want to customize textbook content to match my course curriculum and student needs, so that I can align the material with my specific learning objectives.

2. As an educator, I want to monitor student engagement and comprehension through analytics, so that I can identify struggling students and adjust my teaching approach accordingly.

3. As an educator, I want to integrate additional teaching materials that complement the core robotics textbook, so that I can enrich the learning experience with specialized content.