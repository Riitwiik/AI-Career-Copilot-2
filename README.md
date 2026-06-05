# AI-Career-Copilot-2
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

![Register Page](images/register.png)

![Login Page](images/login.png)

PDF Parsing
PyMuPDF
