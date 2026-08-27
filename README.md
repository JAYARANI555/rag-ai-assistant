# rag-ai-assistant
RAG-based AI assistant combining vector semantic search with LLM-powered responses for intelligent learning systems.

# RAG-Based AI Assistant for Intelligent Learning Systems

A Retrieval-Augmented Generation (RAG) AI assistant that combines vector-based semantic search with LLM-powered responses to deliver accurate, context-aware answers.

## Overview

This project implements a RAG pipeline for intelligent learning systems — ingesting documents, converting them into vector embeddings, and using similarity search combined with LLM prompting to generate accurate, grounded responses.

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines semantic search with LLM-powered response generation
- **Document ingestion pipeline**: Processes raw content into structured, embedded vector representations
- **FAISS-based similarity search**: Fast and efficient vector similarity search for relevant context retrieval
- **Prompt engineering**: Optimized prompts to improve response accuracy and relevance
- **Backend API design**: Structured API layer connecting retrieval and generation components

## Tech Stack

- **Language**: Python
- **Framework**: LangChain
- **LLM Integration**: LLM APIs
- **Vector Database**: FAISS
- **Core Concepts**: Vector embeddings, semantic search, prompt engineering

## Architecture

1. **Document Ingestion** — Raw content (documents/text) is collected and preprocessed
2. **Embedding Generation** — Content is converted into vector embeddings
3. **Vector Storage** — Embeddings are stored and indexed in a FAISS vector database
4. **Retrieval** — On query, FAISS performs similarity search to fetch the most relevant chunks
5. **Generation** — Retrieved context is passed to an LLM via engineered prompts to generate the final response
6. **API Layer** — Backend API exposes the assistant's functionality for integration

## Installation

```bash
git clone <your-repo-url>
cd rag-ai-assistant
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with your LLM API credentials:

```
LLM_API_KEY=your_api_key_here
```

## Usage

```bash
python app.py
```

Ingest your documents into the vector database, then query the assistant to get retrieval-grounded, LLM-generated answers.

## Requirements

```
langchain
faiss-cpu
openai
python-dotenv
```

## Author

**Jaya Rani**
ranijaya0292@gmail.com

## License

This project is open source and available for educational use.
