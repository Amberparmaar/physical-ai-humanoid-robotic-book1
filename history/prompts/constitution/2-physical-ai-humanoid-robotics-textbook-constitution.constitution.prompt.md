---
id: 2
title: Physical AI & Humanoid Robotics Textbook Constitution
stage: constitution
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution You are an engineering-style spec-author assistant. Produce a concise, machine- and human-readable "constitution" that defines high-level principles for building an AI-native textbook on Physical AI & Humanoid Robotics. Output must be JSON with fields: title, principles (array), governance_guidelines (array), non-functional_requirements (array), example_decisions (array). Each principle must be 1-3 sentences and include why & a short implementation hint. Requirements to include: - Clear, modular technical writing using Markdown headings, lists, tables, and code snippets for ROS2, Gazebo, NVIDIA Isaac, and VLA examples. - Reusability via Claude Code subagents and agent skills (explain subagent responsibilities and how they compose). - Docusaurus-first structure (docs/, blog/, static/, sidebars.js) and deployment to GitHub Pages. - RAG chatbot architecture using OpenAI Agents / ChatKit SDKs, FastAPI backend, Neon serverless Postgres for user profiles, Qdrant for vector store. - Personalization: user-profile driven chapter variants; sign-up flow via Better-Auth. - Urdu translation support: UI toggle, page-level translation API, and stored translations in Neon or object store. - Educational priorities: clarity, testable examples, reproducible code snippets, TDD for backend. - Governance: how to document hardware descriptions, module boundaries, allowed third-party libraries, and bonus-features decision rules. Include short examples for each governance rule (one-line). Keep total JSON under 1200 tokens. Ensure that the principles include: Integrate Context7 MCP + Context7 API inside the Docusaurus site for RAG chatbot, selected-text search, and API tools. The RAG system must use Context7 for embeddings, memory, and agent actions.
labels: ["constitution", "ai-native", "robotics"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:

---

## Prompt

```text
You are an engineering-style spec-author assistant. Produce a concise, machine- and human-readable "constitution" that defines high-level principles for building an AI-native textbook on Physical AI & Humanoid Robotics. Output must be JSON with fields: title, principles (array), governance_guidelines (array), non-functional_requirements (array), example_decisions (array). Each principle must be 1-3 sentences and include why & a short implementation hint. Requirements to include: - Clear, modular technical writing using Markdown headings, lists, tables, and code snippets for ROS2, Gazebo, NVIDIA Isaac, and VLA examples. - Reusability via Claude Code subagents and agent skills (explain subagent responsibilities and how they compose). - Docusaurus-first structure (docs/, blog/, static/, sidebars.js) and deployment to GitHub Pages. - RAG chatbot architecture using OpenAI Agents / ChatKit SDKs, FastAPI backend, Neon serverless Postgres for user profiles, Qdrant for vector store. - Personalization: user-profile driven chapter variants; sign-up flow via Better-Auth. - Urdu translation support: UI toggle, page-level translation API, and stored translations in Neon or object store. - Educational priorities: clarity, testable examples, reproducible code snippets, TDD for backend. - Governance: how to document hardware descriptions, module boundaries, allowed third-party libraries, and bonus-features decision rules. Include short examples for each governance rule (one-line). Keep total JSON under 1200 tokens. Ensure that the principles include: Integrate Context7 MCP + Context7 API inside the Docusaurus site for RAG chatbot, selected-text search, and API tools. The RAG system must use Context7 for embeddings, memory, and agent actions.
```

## Response snapshot

The constitution for the Physical AI & Humanoid Robotics Textbook has been created and updated in `.specify/memory/constitution.md`. It defines core principles, governance guidelines, non-functional requirements, and example decisions in a JSON format.

## Outcome

- ✅ Impact: Project constitution updated successfully with AI-native and robotics-specific principles.
- 🧪 Tests: N/A
- 📁 Files: .specify/memory/constitution.md updated.
- 🔁 Next prompts: User will review the constitution and may request further actions.
- 🧠 Reflection: Successfully parsed user requirements and formatted the constitution as JSON, then updated the file and created PHR.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
