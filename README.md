# AI-Career-Copilot-2

## Live Demo

https://ai-career-copilot-2-ycsq2hvsy5zmctwbmwbtux.streamlit.app/

## 🚀 AI Career Copilot

An end-to-end AI-powered career assistant built with Streamlit, SQLite, FAISS, Sentence Transformers, and Groq LLMs.

The application helps users:

Upload and analyze resumes

Chat with resumes using RAG

Match resumes against job descriptions

Identify skill gaps

Generate personalized learning roadmaps

Create mock interview questions

Calculate recruiter-style fit scores

Designed as a production-style MVP using only free and open-source technologies.

## ✨ Features

### 📄 Resume Upload & Processing

Upload PDF resumes

Parse resumes using PyMuPDF

Semantic chunking with LangChain

Embedding generation using Sentence Transformers

Vector indexing using FAISS

### 💬 Resume Q&A (RAG Chatbot)

Ask questions about your resume such as:

“What are my strongest technical skills?”

“What projects relate to machine learning?”

“Summarize my experience”

The system retrieves relevant chunks from the vector database and generates contextual answers using Groq LLMs.

### 🎯 Job Description Matching

Compare your resume with a target job description and get:

Matching skills

Missing skills

Compatibility analysis

Improvement suggestions

### 📊 Skill-Gap Analysis

Identify:

Existing matching skills

Missing requirements

Partial competencies

Priority learning areas

### 🗺️ Personalized Learning Roadmap

Generate structured learning plans for target roles such as:

ML Engineer

Data Scientist

Backend Developer

AI Engineer

Roadmaps include:

Topics to learn

Free learning resources

Project ideas

Progress milestones

### 🎤 Mock Interview Generator

AI-generated:

Behavioral questions

Technical questions

Situational questions

Culture-fit questions

Each question includes:

Difficulty level

Interviewer expectations

Answering tips

### 📈 Recruiter Fit Score

Get recruiter-style evaluation scores:

Skills Match

Experience Relevance

Education Fit

Overall Impression

Includes:

Hire/No-Hire recommendation

Strengths analysis

Concerns analysis

### 🎮 Quick Demo Mode

Instantly test the application using a preloaded demo resume and demo job description.

No upload required.

## 🏗️ Tech Stack

Frontend

Streamlit

Backend

Python 3.10

Database

SQLite

Vector Database

FAISS

Embeddings

Sentence Transformers (all-MiniLM-L6-v2)

LLM

Groq API

Llama 3.1 8B Instant

NLP / RAG

LangChain

PDF Parsing

PyMuPDF

## 🏗️ System Architecture

```text
                +----------------------+
                |     Streamlit UI     |
                |  Interactive Frontend|
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    Business Logic    |
                |    RAG Pipelines     |
                | Resume Intelligence  |
                +----------+-----------+
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
+-------------------+              +----------------------+
|    SQLite DB      |              |   FAISS Vector DB   |
|-------------------|              |----------------------|
| Users             |              | Resume Embeddings    |
| Resumes           |              | Semantic Search      |
| Chat History      |              | Similarity Retrieval |
| Analyses          |              +----------------------+
+-------------------+
                           |
                           v
                +----------------------+
                |      Groq LLM        |
                |  AI Response Engine  |
                +----------------------+
```
## 🧠 RAG Pipeline

```text
Resume Q&A Flow

PDF Resume
    ↓
Text Extraction
    ↓
Semantic Chunking
    ↓
Embedding Generation
    ↓
FAISS Indexing
    ↓
User Query
    ↓
Similarity Search
    ↓
Relevant Chunks Retrieved
    ↓
Prompt Construction
    ↓
Groq LLM Response
```

# 📸 Application Screenshots

## 🔐 Register Page

![Register Page](register.png)

---

## 🔑 Login Page

![Login Page](login.png)

## Resume Upload


![Resume Upload Page](upload.png)


## Resume Question Answer


![Resume Q&A Page 1](Q&A1.png)

![Resume Q&A Page 2](Q&A2.png)


## Job Matching

![Job Matching 0](jobmatching0.png)

![Job Matching 1](jobmatching1.png)

![Job Matching 2](jobmatching2.png)

![Job Matching 3](jobmatching3.png)



## Skill Gap Analysis

![Skill Gap Analysis 0](skillgapanalysis0.png)

![Skill Gap Analysis 1](skillgapanalysis1.png)

![Skill Gap Analysis 2](skillgapanalysis2.png)

![Skill Gap Analysis 3](skillgapanalysis3.png)


## Learning Roadmap

![Learning Roadmap 1](Learningroadmap1.png)

![Learning Roadmap 2](Learningroadmap2.png)

![Learning Roadmap 3](Learningroadmap3.png)

![Learning Roadmap 4](Learningroadmap4.png)

![Learning Roadmap 5](Learningroadmap5.png)

![Learning Roadmap 6](Learningroadmap6.png)

![Learning Roadmap 7](Learningroadmap7.png)

## Mock interview for desired role

![Mock Interview 1](Mockinterview1.png)

![Mock Interview 2](Mockinterview2.png)

![Mock Interview 3](Mockinterview3.png)


## Fit Score

![Fit Score 1](Fitscore1.png)

![Fit Score 2](Fitscore2.png)



