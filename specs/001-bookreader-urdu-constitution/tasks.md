---
description: "Task list for BookReader UI with Urdu translation feature"
---

# Tasks: BookReader UI with Urdu Translation

**Input**: Design documents from `/specs/001-bookreader-urdu-constitution/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The feature specification did not explicitly request tests, so test tasks are not included in this implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `frontend/src/` for the application

<!--
  ============================================================================
  IMPORTANT: The tasks below are generated based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/
  - Technology decisions from research.md

  Tasks are organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  Each story includes: Models → Services → Components → API Integration
  ============================================================================

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create frontend project structure with React, TypeScript and Tailwind CSS
- [x] T002 Initialize package.json with necessary dependencies (react, react-dom, typescript, tailwindcss)
- [x] T003 [P] Configure build tools (vite.config.ts, tsconfig.json, postcss.config.js)
- [x] T004 Create environment configuration files (.env.example)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Foundational tasks:

- [x] T005 [P] Create base models: Book.ts in frontend/src/models/Book.ts
- [x] T006 [P] Create base models: Paragraph.ts in frontend/src/models/Paragraph.ts
- [x] T007 [P] Create base models: UserPreferences.ts in frontend/src/models/UserPreferences.ts
- [x] T008 [P] Create base models: Translation.ts in frontend/src/models/Translation.ts
- [x] T009 Create API client service in frontend/src/services/apiClient.ts
- [x] T010 [P] Implement bookService in frontend/src/services/bookService.ts
- [x] T011 [P] Implement translationService in frontend/src/services/translationService.ts
- [x] T012 [P] Implement storageService in frontend/src/services/storageService.ts
- [x] T013 Set up basic routing in frontend/src/App.tsx
- [x] T014 Create type definitions in frontend/src/types/index.ts
- [x] T015 Create base styling in frontend/src/styles/globals.css
- [x] T016 Create theme provider in frontend/src/components/UI/ThemeProvider.tsx

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Reading with Urdu Translation (Priority: P1) 🎯 MVP

**Goal**: Enable users to read content with Urdu translation available via a floating button

**Independent Test**: User can open a book, see the clean reading area with smooth typography, and access the Urdu translation for any selected paragraph via a floating translation button.

### Implementation for User Story 1

- [x] T017 [P] [US1] Create ReadingArea component in frontend/src/components/Reader/ReadingArea.tsx
- [x] T018 [P] [US1] Create TranslationPanel component in frontend/src/components/Reader/TranslationPanel.tsx
- [x] T019 [P] [US1] Create FloatingTranslationButton component in frontend/src/components/Reader/FloatingTranslationButton.tsx
- [x] T020 [P] [US1] Create hooks: useBookContent in frontend/src/hooks/useBookContent.ts
- [x] T021 [P] [US1] Create hooks: useTranslation in frontend/src/hooks/useTranslation.ts
- [x] T022 [US1] Integrate TranslationPanel with ReadingArea for display
- [x] T023 [US1] Implement functionality to select text and show translation
- [x] T024 [US1] Add smooth typography with CSS in frontend/src/styles/typography.css
- [x] T025 [US1] Implement side-by-side and toggle view for languages
- [x] T026 [US1] Style components with Tailwind to follow minimalist design principles
- [x] T027 [US1] Connect bookService to fetch book content from API
- [x] T028 [US1] Connect translationService to fetch translations from API

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Adjusting Reading Preferences (Priority: P2)

**Goal**: Allow users to adjust font size and theme (dark/light mode) for comfortable reading

**Independent Test**: User can increase/decrease font size and switch between dark/light modes to customize their reading environment.

### Implementation for User Story 2

- [x] T029 [P] [US2] Create SettingsPanel component in frontend/src/components/Layout/SettingsPanel.tsx
- [x] T030 [P] [US2] Create hooks: useUserPreferences in frontend/src/hooks/useUserPreferences.ts
- [x] T031 [P] [US2] Create Button component in frontend/src/components/UI/Button.tsx
- [x] T032 [P] [US2] Create Slider component in frontend/src/components/UI/Slider.tsx
- [x] T033 [US2] Integrate ThemeProvider with App.tsx for theme switching
- [x] T034 [US2] Implement font size adjustment functionality
- [x] T035 [US2] Implement theme switching (dark/light mode)
- [x] T036 [US2] Add functionality to save user preferences to localStorage
- [x] T037 [US2] Connect storageService to persist preferences across sessions
- [x] T038 [US2] Update ReadingArea to respond to theme and font size changes

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Language Switching (Priority: P3)

**Goal**: Enable users to easily switch between English and Urdu views

**Independent Test**: User can switch between English and Urdu text display, which provides the multilingual functionality that differentiates this application.

### Implementation for User Story 3

- [x] T039 [P] [US3] Create LanguageToggle component in frontend/src/components/Reader/LanguageToggle.tsx
- [x] T040 [US3] Update ReadingArea to support language switching
- [x] T041 [US3] Update TranslationPanel to work with language switching
- [x] T042 [US3] Implement language preference setting in UserPreferences
- [x] T043 [US3] Connect language preferences to API for user settings
- [x] T044 [US3] Add UI controls for switching between English ↔ Urdu

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T045 [P] Create BookList component in frontend/src/components/Books/BookList.tsx
- [x] T046 [P] Create BookCard component in frontend/src/components/Books/BookCard.tsx
- [x] T047 [P] Create ReaderPage in frontend/src/pages/ReaderPage.tsx
- [x] T048 [P] Create LibraryPage in frontend/src/pages/LibraryPage.tsx
- [x] T049 [P] Create SettingsPage in frontend/src/pages/SettingsPage.tsx
- [x] T050 [P] Documentation updates in docs/
- [x] T051 Code cleanup and refactoring
- [x] T052 Performance optimization across all stories
- [x] T053 Add caching to IndexedDB for book content and translations
- [x] T054 Implement virtual scrolling for long documents
- [x] T055 Security hardening
- [x] T056 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Models before services
- Services before components
- Components before API integration
- Core functionality before UI enhancements
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel
- Components within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all components for User Story 1 together:
Task: "Create ReadingArea component in frontend/src/components/Reader/ReadingArea.tsx"
Task: "Create TranslationPanel component in frontend/src/components/Reader/TranslationPanel.tsx"
Task: "Create FloatingTranslationButton component in frontend/src/components/Reader/FloatingTranslationButton.tsx"
Task: "Create hooks: useBookContent in frontend/src/hooks/useBookContent.ts"
Task: "Create hooks: useTranslation in frontend/src/hooks/useTranslation.ts"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence