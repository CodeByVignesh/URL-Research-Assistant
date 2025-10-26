# URL Research Assistant

## Description:
#### The URL Research Assistant is an AI-powered web application built with Streamlit and LangChain that allows users to query multiple URLs in natural language and get precise answers along with source references. The system leverages OpenAI models for question answering and a FAISS vector store for retrieving relevant content from web pages, enabling a seamless multi-turn conversational experience.

### Key Features:

#### Multi-URL Input: Users can provide one or more URLs to research.

#### Document Processing: Automatically extracts and splits content from web pages into manageable chunks.

#### Embeddings & Retrieval: Uses OpenAI embeddings and FAISS vector store for semantic search.

#### Multi-Turn Chat: Maintains conversation context across multiple questions using Streamlit session state.

#### Source-Cited Answers: Responses include the source URL for transparency and reference.

#### Interactive UI: Streamlit chat interface with user and assistant messages displayed in order.

### Tech Stack:

#### Frontend: Streamlit

#### Backend & LLM: LangChain, OpenAI Chat Models (gpt-5-chat-latest / gpt-5-mini)

#### Vector Search: FAISS

#### Document Loading: UnstructuredURLLoader for web content extraction

#### Environment Management: Python, .env for API keys

### Usage:

#### Enter one or more URLs in the sidebar.

#### Click “Start Research” to process the pages and create the embeddings.

#### Ask questions in the chat input and get instant answers with source citations.

#### Continue the conversation for multi-turn Q&A.

### Potential Applications:

#### Market research from multiple websites

#### Academic research and literature surveys

#### Customer support knowledge retrieval

#### Quick summarization of web content