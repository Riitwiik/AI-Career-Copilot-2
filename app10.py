


import os
import sys
import json
import logging
import sqlite3
import hashlib
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- Env / Config ---
from dotenv import load_dotenv

# --- Data ---
import numpy as np

# --- PDF Parsing ---
import fitz  # PyMuPDF

# --- Embeddings & Vector Store ---
from sentence_transformers import SentenceTransformer
import faiss

# --- LangChain (Community Edition) ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from groq import Groq

# --- Auth ---
import jwt

# --- Streamlit ---
import streamlit as st
st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "career_copilot.db"
FAISS_DIR = DATA_DIR / "faiss_index"
FAISS_DIR.mkdir(exist_ok=True)

# --- Demo asset paths (beside app.py, NOT in data/) ---
DEMO_RESUME_PDF = BASE_DIR / "demo_resume.pdf"
DEMO_RESUME_INDEX = BASE_DIR / "demo_resume.index"
DEMO_RESUME_METADATA = BASE_DIR / "demo_resume_metadata.json"
DEMO_JD_TEXT = BASE_DIR / "demo_job_description.txt"
DEMO_JD_METADATA = BASE_DIR / "demo_job_description_metadata.json"

# --- Shared demo resume identifier (NOT a user account) ---
DEMO_RESUME_ID = "global_demo_resume"

# --- Max upload size (10 MB) ---
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# --- App Settings ---
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY", "")
)
JWT_SECRET = os.getenv("JWT_SECRET") or st.secrets.get("JWT_SECRET")
JWT_ALGORITHM = (
    os.getenv("JWT_ALGORITHM")
    or st.secrets.get("JWT_ALGORITHM", "HS256")
)
JWT_EXPIRY_HOURS = int(
    os.getenv("JWT_EXPIRY_HOURS")
    or st.secrets.get("JWT_EXPIRY_HOURS", 24)
)
EMBEDDING_MODEL_NAME = (
    os.getenv("EMBEDDING_MODEL_NAME")
    or st.secrets.get(
        "EMBEDDING_MODEL_NAME",
        "all-MiniLM-L6-v2"
    )
)
GROQ_MODEL = (
    os.getenv("GROQ_MODEL")
    or st.secrets.get(
        "GROQ_MODEL",
        "llama-3.1-8b-instant"
    )
)

# --- Retry settings ---
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2  # seconds

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "app.log", mode="a"),
    ],
)
logger = logging.getLogger("career_copilot")



class AppError(Exception):
    """Base application error with user-friendly message and status code."""

    def __init__(self, message: str, status_code: int = 500, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or message
        logger.error(f"AppError({status_code}): {message} | {detail}")


def handle_error(func):
    """Decorator that wraps functions with centralized error handling."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AppError:
            raise
        except Exception as exc:
            logger.exception(f"Unhandled error in {func.__name__}")
            raise AppError(
                message="An unexpected error occurred. Please try again.",
                status_code=500,
                detail=str(exc),
            )

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def safe_error(e: Exception) -> str:
    """Return a user-friendly error message, never exposing tracebacks."""
    if isinstance(e, AppError):
        return e.message
    logger.exception("Unexpected error converted to user message")
    return "An unexpected error occurred. Please try again."



@contextmanager
def get_db():
    """Context manager that yields a SQLite connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist, apply PRAGMA settings and indexes."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Performance PRAGMAs
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Resumes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                chunks_json TEXT,
                file_hash TEXT,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Job descriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                resume_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Analyses table (skill-gap, roadmaps, interviews, scores)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                resume_id INTEGER NOT NULL,
                job_id INTEGER,
                analysis_type TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_resumes_user ON resumes(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_user ON analyses(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_resumes_hash ON resumes(file_hash)",
        ]
        for idx_sql in indexes:
            cursor.execute(idx_sql)

        conn.commit()
    logger.info("Database initialized successfully")



def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against the stored salt:hash."""
    salt, hashed = stored_hash.split(":")
    computed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return secrets.compare_digest(computed, hashed)


def create_token(user_id: int, username: str) -> str:
    """Generate a JWT token for the given user."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token. Raises AppError on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AppError("Token has expired. Please log in again.", 401)
    except jwt.InvalidTokenError:
        raise AppError("Invalid token.", 401)


@handle_error
def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """Register a new user. Returns token on success."""
    with get_db() as conn:
        try:
            pw_hash = hash_password(password)
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, pw_hash),
            )
            conn.commit()
            user_id = cursor.lastrowid
            token = create_token(user_id, username)
            logger.info(f"User registered: {username}")
            return {"user_id": user_id, "username": username, "token": token}
        except sqlite3.IntegrityError:
            raise AppError("Username or email already exists.", 409)


@handle_error
def login_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticate a user. Returns token on success."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            raise AppError("Invalid username or password.", 401)
        token = create_token(row["id"], row["username"])
        logger.info(f"User logged in: {username}")
        return {"user_id": row["id"], "username": row["username"], "token": token}



@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (lazy, loaded once)."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully")
    return model


def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for a list of text chunks in batches."""
    model = get_embedding_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(np.array(batch_embeddings, dtype=np.float32))
    if len(all_embeddings) == 1:
        return all_embeddings[0]
    return np.vstack(all_embeddings)


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS inner-product index from embeddings."""
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, resume_id: int):
    """Persist FAISS index and its chunk metadata to disk."""
    index_path = FAISS_DIR / f"resume_{resume_id}.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved for resume_id={resume_id}")


def load_faiss_index(resume_id: int) -> Optional[faiss.Index]:
    """Load a FAISS index from disk. Returns None if not found."""
    index_path = FAISS_DIR / f"resume_{resume_id}.index"
    if index_path.exists():
        try:
            return faiss.read_index(str(index_path))
        except Exception as e:
            logger.error(f"Failed to load FAISS index for resume_id={resume_id}: {e}")
            return None
    return None


def search_faiss(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 5):
    """Search the FAISS index for the most similar chunks."""
    query_embedding = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    return scores[0], indices[0]



@handle_error
def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        doc = None
        if not text.strip():
            raise AppError(
                "Could not extract text from PDF. It may be image-based or corrupted.",
                400,
            )
        logger.info(f"PDF parsed: {len(text)} characters extracted")
        return text
    except AppError:
        raise
    except Exception as e:
        raise AppError(f"Failed to parse PDF. The file may be corrupted.", 400, detail=str(e))
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def semantic_chunk(
    text: str, chunk_size: int = 0, chunk_overlap: int = 100
) -> List[str]:
    """
    Split text into semantic chunks using RecursiveCharacterTextSplitter.
    Dynamic chunking: short resumes get larger chunks, long resumes get smaller chunks.
    """
    text_length = len(text)
    if chunk_size <= 0:
        if text_length < 2000:
            chunk_size = 800
        elif text_length < 5000:
            chunk_size = 500
        else:
            chunk_size = 350

    computed_overlap = min(chunk_overlap, chunk_size // 5)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=computed_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info(
        f"Text chunked into {len(chunks)} segments "
        f"(chunk_size={chunk_size}, text_length={text_length})"
    )
    return chunks



@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        raise AppError(
            "GROQ_API_KEY not set. Add it to your .env file.",
            500,
            "Missing Groq API key",
        )
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized")
    return client


def ask_llm(prompt: str) -> str:
    client = get_groq_client()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)

    raise AppError("AI service is currently unavailable.", 502, detail=str(last_error))


def _compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def _find_existing_resume(user_id: int, file_hash: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, filename FROM resumes WHERE user_id = ? AND file_hash = ? LIMIT 1",
            (user_id, file_hash),
        ).fetchone()
        if row:
            return dict(row)
    return None


@handle_error
def process_resume(
    user_id: int, filename: str, file_bytes: bytes, progress_callback=None
) -> Dict[str, Any]:
    """Process a USER-UPLOADED resume: parse -> chunk -> embed -> index.

    This pipeline is ONLY used for user uploads. The demo resume bypasses
    it entirely by loading pre-built assets from disk.
    """
    if not file_bytes:
        raise AppError("Uploaded file is empty.", 400)
    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise AppError(f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB.", 400)
    if not filename.lower().endswith(".pdf"):
        raise AppError("Only PDF files are supported.", 400)

    file_hash = _compute_file_hash(file_bytes)
    existing = _find_existing_resume(user_id, file_hash)
    if existing:
        logger.info(f"Duplicate resume detected (hash={file_hash}), reusing resume_id={existing['id']}")
        index = load_faiss_index(existing["id"])
        if index is not None:
            return {"resume_id": existing["id"], "filename": existing["filename"], "char_count": 0, "chunk_count": 0, "reused": True, "message": "This resume was already uploaded. Reusing existing data."}

    if progress_callback:
        progress_callback(0.1, "Parsing PDF...")
    raw_text = parse_pdf(file_bytes)

    if progress_callback:
        progress_callback(0.3, "Chunking text...")
    chunks = semantic_chunk(raw_text)

    if progress_callback:
        progress_callback(0.5, "Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    if progress_callback:
        progress_callback(0.8, "Building search index...")
    index = create_faiss_index(embeddings)

    with get_db() as conn:
        if existing:
            conn.execute("UPDATE resumes SET raw_text = ?, chunks_json = ? WHERE id = ?", (raw_text, json.dumps(chunks), existing["id"]))
            conn.commit()
            resume_id = existing["id"]
        else:
            cursor = conn.execute("INSERT INTO resumes (user_id, filename, raw_text, chunks_json, file_hash) VALUES (?, ?, ?, ?, ?)", (user_id, filename, raw_text, json.dumps(chunks), file_hash))
            conn.commit()
            resume_id = cursor.lastrowid

    save_faiss_index(index, resume_id)
    del embeddings
    del index

    if progress_callback:
        progress_callback(1.0, "Done!")

    logger.info(f"Resume processed: id={resume_id}, chunks={len(chunks)}")
    return {"resume_id": resume_id, "filename": filename, "char_count": len(raw_text), "chunk_count": len(chunks), "reused": False}



def _is_demo_resume(resume_id: int) -> bool:
    """Check if the given resume_id corresponds to the shared demo resume."""
    demo_meta = load_demo_metadata()
    if demo_meta is None:
        return False
    with get_db() as conn:
        row = conn.execute(
            "SELECT file_hash FROM resumes WHERE id = ?",
            (resume_id,),
        ).fetchone()
    if row and row["file_hash"] == "demo_resume_prebuilt":
        return True
    return False


def _get_resume_data(resume_id: int, user_id: int) -> Dict[str, Any]:
    """Get resume data (raw_text, chunks_json, FAISS index) for any resume.

    For user-owned resumes: validates ownership (user_id match).
    For the shared demo resume: skips ownership validation.
    Returns dict with keys: raw_text, chunks_json, index
    """
    is_demo = _is_demo_resume(resume_id)

    with get_db() as conn:
        if is_demo:
            row = conn.execute(
                "SELECT raw_text, chunks_json FROM resumes WHERE id = ?",
                (resume_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT raw_text, chunks_json FROM resumes WHERE id = ? AND user_id = ?",
                (resume_id, user_id),
            ).fetchone()

    if not row:
        raise AppError("Resume not found.", 404)

    # Load FAISS index — for demo resume, try DATA_DIR first, then BASE_DIR
    index = load_faiss_index(resume_id)
    if index is None and is_demo:
        # Fallback: load from the pre-built demo index beside app.py
        index = load_demo_faiss_index()
    if index is None:
        raise AppError("Resume index not found. Please re-upload your resume.", 404)

    return {
        "raw_text": row["raw_text"],
        "chunks_json": row["chunks_json"],
        "index": index,
    }



@handle_error
def resume_qa(user_id: int, resume_id: int, question: str) -> Dict[str, Any]:
    data = _get_resume_data(resume_id, user_id)
    chunks = json.loads(data["chunks_json"])
    index = data["index"]

    query_embedding = generate_embeddings([question])[0]
    scores, indices = search_faiss(index, query_embedding, top_k=5)

    context_parts = []
    for idx in indices:
        if 0 <= idx < len(chunks):
            context_parts.append(chunks[idx])
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an AI Career Copilot. Answer the user's question based on the resume context below.
If the answer is not found in the context, say so clearly.

RESUME CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""

    answer = ask_llm(prompt)

    with get_db() as conn:
        conn.execute("INSERT INTO chat_history (user_id, resume_id, question, answer) VALUES (?, ?, ?, ?)", (user_id, resume_id, question, answer))
        conn.commit()

    return {"question": question, "answer": answer, "context_chunks_used": len(context_parts)}


@handle_error
def match_job(user_id: int, resume_id: int, job_description: str, job_title: str = "Untitled") -> Dict[str, Any]:
    data = _get_resume_data(resume_id, user_id)
    chunks = json.loads(data["chunks_json"])
    index = data["index"]

    job_embedding = generate_embeddings([job_description])[0]
    scores, indices = search_faiss(index, job_embedding, top_k=8)

    context_parts = [chunks[i] for i in indices if 0 <= i < len(chunks)]
    context = "\n\n---\n\n".join(context_parts)
    avg_score = float(np.mean(scores[scores > 0])) if np.any(scores > 0) else 0.0

    with get_db() as conn:
        cursor = conn.execute("INSERT INTO job_descriptions (user_id, title, description) VALUES (?, ?, ?)", (user_id, job_title, job_description))
        conn.commit()
        job_id = cursor.lastrowid

    prompt = f"""You are an expert career advisor. Compare the resume sections below with the job description.
Provide:
1. Key matching skills and experiences
2. Missing skills or experiences
3. Overall compatibility assessment
4. Specific suggestions for improvement

JOB DESCRIPTION:
{job_description}

RESUME SECTIONS:
{context}

DETAILED MATCH ANALYSIS:"""

    analysis = ask_llm(prompt)

    with get_db() as conn:
        conn.execute("INSERT INTO analyses (user_id, resume_id, job_id, analysis_type, result_json) VALUES (?, ?, ?, ?, ?)", (user_id, resume_id, job_id, "job_match", json.dumps({"analysis": analysis, "similarity_score": avg_score})))
        conn.commit()

    return {"job_id": job_id, "similarity_score": round(avg_score, 4), "analysis": analysis}


@handle_error
def skill_gap_analysis(user_id: int, resume_id: int, job_description: str) -> Dict[str, Any]:
    data = _get_resume_data(resume_id, user_id)
    resume_text = data["raw_text"][:3000]

    prompt = f"""You are an expert technical recruiter. Perform a detailed skill-gap analysis.

RESUME (excerpt):
{resume_text}

TARGET JOB DESCRIPTION:
{job_description}

Provide your analysis in this format:
1. SKILLS YOU HAVE (that match the job)
2. SKILLS YOU'RE MISSING (required by the job but not in your resume)
3. PARTIAL SKILLS (skills you have some experience with but need deeper expertise)
4. PRIORITY RECOMMENDATIONS (which gaps to close first and why)

DETAILED SKILL-GAP ANALYSIS:"""

    analysis = ask_llm(prompt)
    with get_db() as conn:
        conn.execute("INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)", (user_id, resume_id, "skill_gap", json.dumps({"analysis": analysis})))
        conn.commit()
    return {"analysis": analysis}


@handle_error
def generate_roadmap(user_id: int, resume_id: int, target_role: str) -> Dict[str, Any]:
    data = _get_resume_data(resume_id, user_id)
    resume_text = data["raw_text"][:3000]

    prompt = f"""You are a career mentor and learning strategist. Based on the person's current resume and their target role, create a practical learning roadmap.

CURRENT RESUME (excerpt):
{resume_text}

TARGET ROLE: {target_role}

Create a structured roadmap with:
1. PHASE 1 - Foundation (0-2 months): Core skills to build
2. PHASE 2 - Intermediate (2-4 months): Building depth
3. PHASE 3 - Advanced (4-6 months): Specialization
4. PHASE 4 - Portfolio & Networking (6-8 months): Real-world application

For each phase, include:
- Specific topics/concepts to learn
- Recommended free resources (courses, books, YouTube channels)
- Mini-projects to build
- Metrics to track progress

LEARNING ROADMAP:"""

    roadmap = ask_llm(prompt)
    with get_db() as conn:
        conn.execute("INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)", (user_id, resume_id, "roadmap", json.dumps({"target_role": target_role, "roadmap": roadmap})))
        conn.commit()
    return {"target_role": target_role, "roadmap": roadmap}


@handle_error
def generate_interview_questions(user_id: int, resume_id: int, target_role: str, num_questions: int = 10) -> Dict[str, Any]:
    data = _get_resume_data(resume_id, user_id)
    resume_text = data["raw_text"][:3000]

    prompt = f"""You are a technical interview coach. Generate {num_questions} mock interview questions based on this resume and target role.

CURRENT RESUME (excerpt):
{resume_text}

TARGET ROLE: {target_role}

Generate a mix of:
- Behavioral questions (STAR method)
- Technical questions (based on resume skills)
- Situational questions (real-world scenarios)
- Culture-fit questions

For each question, provide:
1. The question
2. What the interviewer is looking for
3. A brief tip on how to answer well
4. Difficulty level (Easy/Medium/Hard)

MOCK INTERVIEW QUESTIONS:"""

    questions = ask_llm(prompt)
    with get_db() as conn:
        conn.execute("INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)", (user_id, resume_id, "interview", json.dumps({"target_role": target_role, "questions": questions})))
        conn.commit()
    return {"target_role": target_role, "questions": questions}


@handle_error
def recruiter_fit_score(user_id: int, resume_id: int, job_description: str) -> Dict[str, Any]:
    data = _get_resume_data(resume_id, user_id)
    chunks = json.loads(data["chunks_json"])
    index = data["index"]

    job_embedding = generate_embeddings([job_description])[0]
    scores, _ = search_faiss(index, job_embedding, top_k=10)
    semantic_score = float(np.mean(scores[scores > 0])) * 100 if np.any(scores > 0) else 0.0
    semantic_score = min(semantic_score, 100)

    resume_text = data["raw_text"][:3000]

    prompt = f"""You are a senior technical recruiter at a top company. Evaluate this resume against the job description.

RESUME (excerpt):
{resume_text}

JOB DESCRIPTION:
{job_description}

Provide scores (0-100) for each category:
1. SKILLS MATCH: How well do the candidate's skills match the job requirements?
2. EXPERIENCE RELEVANCE: How relevant is their experience?
3. EDUCATION FIT: How well does their education align?
4. OVERALL IMPRESSION: General recruiter impression

Also provide:
- A one-paragraph summary of the candidate's strengths
- A one-paragraph summary of concerns
- A HIRE/NO-HIRE recommendation with confidence level

Format your response clearly with labeled scores and sections.

RECRUITER EVALUATION:"""

    evaluation = ask_llm(prompt)

    result = {"semantic_similarity_score": round(semantic_score, 1), "evaluation": evaluation}
    with get_db() as conn:
        conn.execute("INSERT INTO analyses (user_id, resume_id, analysis_type, result_json) VALUES (?, ?, ?, ?)", (user_id, resume_id, "fit_score", json.dumps(result)))
        conn.commit()
    return result


@st.cache_data(ttl=60)
def list_resumes(user_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT id, filename, length(raw_text) as chars, uploaded_at FROM resumes WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,)).fetchall()
        return [dict(r) for r in rows]


@handle_error
def get_chat_history(user_id: int, resume_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    with get_db() as conn:
        if resume_id:
            rows = conn.execute("SELECT question, answer, created_at FROM chat_history WHERE user_id = ? AND resume_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, resume_id, limit)).fetchall()
        else:
            rows = conn.execute("SELECT question, answer, resume_id, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]


DEMO_USER_ID = 0


@st.cache_resource
def load_demo_faiss_index() -> Optional[faiss.Index]:
    """Load the pre-built demo FAISS index from BASE_DIR.

    This is separate from load_faiss_index() which reads from DATA_DIR/faiss_index/.
    The demo index lives beside app.py and is never rebuilt.
    """
    if DEMO_RESUME_INDEX.exists():
        try:
            index = faiss.read_index(str(DEMO_RESUME_INDEX))
            logger.info(f"Demo FAISS index loaded: {index.ntotal} vectors, dim={index.d}")
            return index
        except Exception as e:
            logger.error(f"Failed to load demo FAISS index: {e}")
            return None
    logger.warning(f"Demo FAISS index not found at {DEMO_RESUME_INDEX}")
    return None


@st.cache_data
def load_demo_metadata() -> Optional[Dict[str, Any]]:
    """Load the pre-built demo resume metadata from BASE_DIR.

    Returns the metadata dict with chunks, or None on failure.
    """
    if DEMO_RESUME_METADATA.exists():
        try:
            with open(DEMO_RESUME_METADATA, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Demo metadata loaded: {metadata.get('chunk_count', 0)} chunks")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load demo metadata: {e}")
            return None
    logger.warning(f"Demo metadata not found at {DEMO_RESUME_METADATA}")
    return None


@st.cache_data
def load_demo_jd_text() -> Optional[str]:
    """Load the demo job description text from BASE_DIR.

    Simply reads the file and returns its contents.
    No API calls, no regeneration, no embeddings.
    """
    if DEMO_JD_TEXT.exists():
        try:
            with open(DEMO_JD_TEXT, "r", encoding="utf-8") as f:
                text = f.read().strip()
            logger.info(f"Demo JD loaded: {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"Failed to load demo JD: {e}")
            return None
    logger.warning(f"Demo JD text not found at {DEMO_JD_TEXT}")
    return None


def _get_demo_resume_db_id() -> Optional[int]:
    """Return the shared demo resume's database row ID, or None.

    The demo resume is stored with user_id=0 (sentinel) so it does not
    belong to any real user account.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM resumes WHERE user_id = ? AND file_hash = ? "
            "ORDER BY uploaded_at DESC LIMIT 1",
            (DEMO_USER_ID, "demo_resume_prebuilt"),
        ).fetchone()
        return row["id"] if row else None


def _activate_demo_for_user(user_id: int) -> Optional[int]:
    """Register the shared demo resume for a specific user if not already done.

    This creates a copy of the demo resume row in SQLite under the requesting
    user's account (with a special file_hash) and copies the FAISS index to
    the user's expected path. This ensures all existing feature functions
    (resume_qa, match_job, etc.) work without modification.

    Returns the resume_id for the user's copy of the demo resume, or None on failure.
    """
    # Check if user already has a demo resume copy
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM resumes WHERE user_id = ? AND file_hash = ? "
            "ORDER BY uploaded_at DESC LIMIT 1",
            (user_id, "demo_resume_prebuilt"),
        ).fetchone()
        if row:
            resume_id = row["id"]
            # Ensure the FAISS index exists for this resume
            index_path = FAISS_DIR / f"resume_{resume_id}.index"
            if index_path.exists():
                return resume_id
            # Copy from pre-built demo index
            demo_index = load_demo_faiss_index()
            if demo_index is not None:
                faiss.write_index(demo_index, str(index_path))
                logger.info(f"Copied demo FAISS index to {index_path}")
                return resume_id

    # Need to create a new row for this user
    demo_metadata = load_demo_metadata()
    demo_index = load_demo_faiss_index()

    if demo_metadata is None or demo_index is None:
        logger.error("Cannot activate demo resume — pre-built assets not available")
        return None

    chunks = demo_metadata.get("chunks", [])
    raw_text = "\n".join(chunks)

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO resumes (user_id, filename, raw_text, chunks_json, file_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, "demo_resume.pdf", raw_text, json.dumps(chunks), "demo_resume_prebuilt"),
        )
        conn.commit()
        resume_id = cursor.lastrowid

    # Copy the pre-built FAISS index to the user-index location
    index_path = FAISS_DIR / f"resume_{resume_id}.index"
    faiss.write_index(demo_index, str(index_path))

    logger.info(
        f"Demo resume activated for user_id={user_id} "
        f"(resume_id={resume_id}, {len(chunks)} chunks) — NO re-embedding"
    )
    # Clear the resume list cache so the new resume appears
    list_resumes.clear()
    return resume_id


@st.cache_resource
def init_demo_data():
    """Initialize the shared demo resume in the database using PRE-BUILT assets.

    Startup flow:
      1. Check if demo resume already registered in DB (user_id=0 sentinel)
      2. If YES: done — instant return
      3. If NO: load pre-built demo_resume.index + demo_resume_metadata.json
         and register them in SQLite under user_id=0. NO re-embedding,
         NO PDF parsing, NO demo user account.
      4. Fallback: if pre-built files are missing, process demo_resume.pdf
         once (only needed in development, never in production).

    This function is cached by @st.cache_resource — runs only once per app
    lifecycle.
    """
    logger.info("Initializing shared demo data (no demo user)...")

    # ── Step 1: Check if demo resume already registered under sentinel user_id ──
    existing_id = _get_demo_resume_db_id()
    if existing_id is not None:
        # Also verify the FAISS index file exists in DATA_DIR
        index_path = FAISS_DIR / f"resume_{existing_id}.index"
        if index_path.exists():
            logger.info(f"Shared demo resume already registered (id={existing_id}) — instant return")
            return True
        # DB row exists but FAISS missing — copy from pre-built
        demo_index = load_demo_faiss_index()
        if demo_index is not None:
            faiss.write_index(demo_index, str(index_path))
            logger.info(f"Copied pre-built demo index to {index_path}")
            return True

    # ── Step 2: Load from pre-built assets (FAST PATH) ──
    demo_metadata = load_demo_metadata()
    demo_index = load_demo_faiss_index()

    if demo_metadata is not None and demo_index is not None:
        chunks = demo_metadata.get("chunks", [])
        raw_text = "\n".join(chunks)

        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO resumes (user_id, filename, raw_text, chunks_json, file_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (DEMO_USER_ID, "demo_resume.pdf", raw_text, json.dumps(chunks), "demo_resume_prebuilt"),
            )
            conn.commit()
            resume_id = cursor.lastrowid

        # Copy the pre-built FAISS index to the user-index location
        index_path = FAISS_DIR / f"resume_{resume_id}.index"
        faiss.write_index(demo_index, str(index_path))

        logger.info(f"Shared demo resume loaded from pre-built assets (id={resume_id}, {len(chunks)} chunks) — NO re-embedding")
        return True

    # ── Step 3: Fallback — process demo PDF (DEVELOPMENT ONLY) ──
    if DEMO_RESUME_PDF.exists():
        logger.warning("Pre-built demo assets not found — falling back to PDF processing (development mode)")
        try:
            pdf_bytes = DEMO_RESUME_PDF.read_bytes()
            raw_text = parse_pdf(pdf_bytes)
            chunks = semantic_chunk(raw_text)
            embeddings = generate_embeddings(chunks)
            index = create_faiss_index(embeddings)

            with get_db() as conn:
                cursor = conn.execute(
                    "INSERT INTO resumes (user_id, filename, raw_text, chunks_json, file_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (DEMO_USER_ID, "demo_resume.pdf", raw_text, json.dumps(chunks), "demo_resume_prebuilt"),
                )
                conn.commit()
                resume_id = cursor.lastrowid

            save_faiss_index(index, resume_id)
            del embeddings
            del index

            logger.info(f"Shared demo resume processed via fallback: id={resume_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to process demo resume via fallback: {e}")
            return False

    logger.error("No demo resume available — neither pre-built assets nor PDF found")
    return False



def init_session_state():
    defaults = {
        "token": None, "username": None, "user_id": None,
        "current_resume_id": None, "chat_history": [],
        "uploaded_resume": None, "resume_processed": False,
        "uploaded_filename": None, "upload_file_hash": None,
        "_demo_jd_loaded": False, "_demo_sg_jd_loaded": False,
        "_demo_fit_jd_loaded": False,
        "demo_resume_activated": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def run_streamlit():
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; }
        .sub-header { font-size: 1.3rem; font-weight: 600; color: #424242; }
        .success-box { padding: 1rem; border-radius: 0.5rem; background: #E8F5E9; border-left: 4px solid #43A047; }
        .info-box { padding: 1rem; border-radius: 0.5rem; background: #E3F2FD; border-left: 4px solid #1E88E5; }
        .warning-box { padding: 1rem; border-radius: 0.5rem; background: #FFF8E1; border-left: 4px solid #FFA000; }
        .analysis-card { padding: 1.5rem; border-radius: 0.75rem; background: #FAFAFA; border: 1px solid #E0E0E0; margin: 1rem 0; }
        .score-circle { font-size: 3rem; font-weight: 700; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()
    init_db()
    init_demo_data()

    if not st.session_state.token:
        st.markdown('<p class="main-header">🚀 AI Career Copilot</p>', unsafe_allow_html=True)
        st.markdown("Your intelligent career companion — powered by AI")

        tab_reg, tab_login = st.tabs(["Register", "Login"])

        with tab_reg:
            with st.form("register_form"):
                reg_user = st.text_input("Username", key="reg_username")
                reg_email = st.text_input("Email", key="reg_email")
                reg_pass = st.text_input("Password (6+ chars)", type="password", key="reg_password")
                reg_submit = st.form_submit_button("Create Account")
                if reg_submit:
                    try:
                        result = register_user(reg_user, reg_email, reg_pass)
                        st.session_state.token = result["token"]
                        st.session_state.username = result["username"]
                        st.session_state.user_id = result["user_id"]
                        st.success("Account created! Welcome aboard 🎉")
                        st.rerun()
                    except AppError as e:
                        st.error(e.message)

        with tab_login:
            with st.form("login_form"):
                log_user = st.text_input("Username", key="log_username")
                log_pass = st.text_input("Password", type="password", key="log_password")
                log_submit = st.form_submit_button("Login")
                if log_submit:
                    try:
                        result = login_user(log_user, log_pass)
                        st.session_state.token = result["token"]
                        st.session_state.username = result["username"]
                        st.session_state.user_id = result["user_id"]
                        st.success(f"Welcome back, {result['username']}! 🎉")
                        st.rerun()
                    except AppError as e:
                        st.error(e.message)
        return

    st.sidebar.markdown(f"👤 **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.divider()

    # ── Quick Demo (TOP PRIORITY — first thing users see after login) ──
    with st.sidebar.container():
        st.subheader("🎮 Quick Demo")
        st.caption(
            "Try the complete application instantly using a preloaded demo resume."
            " No upload required."
        )
        if st.sidebar.button(
            "🚀 Use Demo Resume",
            use_container_width=True,
            type="primary",
            help="Instantly load a shared demo resume to test all features",
        ):
            demo_rid = _activate_demo_for_user(st.session_state.user_id)
            if demo_rid:
                st.session_state.current_resume_id = demo_rid
                st.session_state.demo_resume_activated = True
                list_resumes.clear()
                st.sidebar.success("✅ Demo Resume Active")
                st.rerun()
            else:
                st.sidebar.error("Demo resume not available. Pre-built assets may be missing.")

        # Show current active resume status
        if st.session_state.current_resume_id:
            is_demo = _is_demo_resume(st.session_state.current_resume_id)
            if is_demo:
                st.sidebar.success("✅ Demo Resume Active")

    st.sidebar.divider()

    # ── Resume Selection ──
    st.sidebar.subheader("📂 Resume Selection")
    try:
        resumes = list_resumes(st.session_state.user_id)
        if resumes:
            resume_options = {f"ID:{r['id']} - {r['filename']}": r["id"] for r in resumes}
            selected = st.sidebar.selectbox("Select a resume", list(resume_options.keys()))
            st.session_state.current_resume_id = resume_options[selected]
        else:
            st.sidebar.info("Upload a resume or use the Demo Resume!")
    except Exception:
        st.sidebar.warning("Could not load resumes")

    st.sidebar.divider()

    # ── Navigation ──
    page = st.sidebar.radio("Navigate", ["📄 Resume Upload", "💬 Resume Q&A", "🎯 Job Matching", "📊 Skill-Gap Analysis", "🗺️ Learning Roadmap", "🎤 Mock Interview", "📈 Fit Score"])

    if page == "📄 Resume Upload":
        st.markdown('<p class="main-header">📄 Resume Upload</p>', unsafe_allow_html=True)
        st.markdown("Upload your resume PDF and we'll parse, chunk, embed, and index it for AI analysis.")

        # Show demo resume availability
        demo_available = load_demo_metadata() is not None and load_demo_faiss_index() is not None
        if demo_available:
            st.markdown(
                '<div class="info-box">🎮 <b>Quick Start:</b> Click '
                '<b>🚀 Use Demo Resume</b> in the sidebar to instantly load a '
                'shared demo resume and try all features without uploading!</div>',
                unsafe_allow_html=True,
            )

        uploaded = st.file_uploader("Choose a PDF resume", type=["pdf"])
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            current_hash = hashlib.md5(file_bytes).hexdigest()
            if st.session_state.upload_file_hash != current_hash:
                st.session_state.uploaded_resume = file_bytes
                st.session_state.uploaded_filename = uploaded.name
                st.session_state.upload_file_hash = current_hash
                st.session_state.resume_processed = False

            if st.button("🔍 Process Resume", type="primary"):
                progress_bar = st.progress(0, text="Preparing...")
                try:
                    def progress_cb(pct, msg): progress_bar.progress(int(pct * 100), text=msg)
                    result = process_resume(st.session_state.user_id, st.session_state.uploaded_filename, st.session_state.uploaded_resume, progress_callback=progress_cb)
                    st.session_state.current_resume_id = result["resume_id"]
                    st.session_state.resume_processed = True
                    list_resumes.clear()
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    if result.get("reused"): st.success("Resume already exists — reusing existing data!")
                    else: st.success("Resume processed successfully!")
                    st.json(result)
                    st.markdown('</div>', unsafe_allow_html=True)
                except AppError as e: st.error(f"Error: {e.message}")
                except Exception as e: st.error(f"Error: {safe_error(e)}")

        st.divider()
        st.subheader("Your Resumes")
        try:
            resumes = list_resumes(st.session_state.user_id)
            if resumes:
                for r in resumes:
                    with st.expander(f"📄 {r['filename']} (uploaded {r['uploaded_at']})"):
                        st.write(f"**Resume ID:** {r['id']}")
                        st.write(f"**Characters:** {r['chars']}")
            else: st.info("No resumes uploaded yet. Upload your first one above or use the Demo Resume!")
        except Exception as e: st.error(f"Could not load resumes: {safe_error(e)}")

    elif page == "💬 Resume Q&A":
        st.markdown('<p class="main-header">💬 Resume Q&A Chat</p>', unsafe_allow_html=True)
        st.markdown("Ask anything about your resume — the AI uses RAG to give context-aware answers.")
        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first, or click 🚀 Use Demo Resume in the sidebar!")
            return
        st.info(f"Chatting about Resume ID: {st.session_state.current_resume_id}")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input("Ask about your resume..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = resume_qa(st.session_state.user_id, st.session_state.current_resume_id, prompt)
                        answer = result["answer"]
                        st.markdown(answer)
                        st.caption(f"Retrieved {result['context_chunks_used']} relevant chunks")
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except AppError as e: st.error(f"Error: {e.message}")
                    except Exception as e: st.error(f"Error: {safe_error(e)}")

    elif page == "🎯 Job Matching":
        st.markdown('<p class="main-header">🎯 Job Description Matching</p>', unsafe_allow_html=True)
        st.markdown("Paste a job description and see how well your resume matches.")
        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first, or click 🚀 Use Demo Resume in the sidebar!")
            return
        job_title = st.text_input("Job Title", placeholder="e.g., Senior ML Engineer at Sarvam AI")
        job_desc = st.text_area("Job Description", height=250, placeholder="Paste the full job description here...")
        col_jd1, col_jd2 = st.columns([1, 3])
        with col_jd1:
            if st.button("📋 Load Demo JD", help="Fill with a sample Senior ML Engineer job description"):
                st.session_state._demo_jd_loaded = True
                st.rerun()
        with col_jd2:
            if st.session_state.get("_demo_jd_loaded"):
                jd_text = load_demo_jd_text()
                if jd_text:
                    job_desc = jd_text
                    st.info("✅ Demo job description loaded!")
        if st.button("🔍 Analyze Match", type="primary"):
            if not job_desc.strip(): st.error("Please enter a job description.")
            else:
                with st.spinner("Analyzing match..."):
                    try:
                        result = match_job(st.session_state.user_id, st.session_state.current_resume_id, job_desc, job_title or "Untitled")
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.metric("Semantic Similarity", f"{result['similarity_score']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["analysis"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e: st.error(f"Error: {e.message}")
                    except Exception as e: st.error(f"Error: {safe_error(e)}")

    elif page == "📊 Skill-Gap Analysis":
        st.markdown('<p class="main-header">📊 Skill-Gap Analysis</p>', unsafe_allow_html=True)
        st.markdown("Identify exactly what skills you need to develop for your target job.")
        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first, or click 🚀 Use Demo Resume in the sidebar!")
            return
        sg_jd_key = "_demo_sg_jd_loaded"
        job_desc = st.text_area("Target Job Description", height=250, placeholder="Paste the job description you're targeting...")
        col_sg1, col_sg2 = st.columns([1, 3])
        with col_sg1:
            if st.button("📋 Load Demo JD", key="sg_load_demo_jd", help="Fill with a sample Senior ML Engineer job description"):
                st.session_state[sg_jd_key] = True
                st.rerun()
        with col_sg2:
            if st.session_state.get(sg_jd_key):
                jd_text = load_demo_jd_text()
                if jd_text:
                    job_desc = jd_text
                    st.info("✅ Demo job description loaded!")
        if st.button("📊 Analyze Skill Gaps", type="primary"):
            if not job_desc.strip(): st.error("Please enter a job description.")
            else:
                with st.spinner("Analyzing skill gaps..."):
                    try:
                        result = skill_gap_analysis(st.session_state.user_id, st.session_state.current_resume_id, job_desc)
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["analysis"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e: st.error(f"Error: {e.message}")
                    except Exception as e: st.error(f"Error: {safe_error(e)}")

    elif page == "🗺️ Learning Roadmap":
        st.markdown('<p class="main-header">🗺️ Learning Roadmap</p>', unsafe_allow_html=True)
        st.markdown("Get a personalized, phased learning plan for your target role.")
        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first, or click 🚀 Use Demo Resume in the sidebar!")
            return
        target_role = st.text_input("Target Role", placeholder="e.g., ML Engineer, Data Scientist, Backend Developer")
        if st.button("🗺️ Generate Roadmap", type="primary"):
            if not target_role.strip(): st.error("Please enter a target role.")
            else:
                with st.spinner("Generating your personalized roadmap..."):
                    try:
                        result = generate_roadmap(st.session_state.user_id, st.session_state.current_resume_id, target_role)
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["roadmap"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e: st.error(f"Error: {e.message}")
                    except Exception as e: st.error(f"Error: {safe_error(e)}")

    elif page == "🎤 Mock Interview":
        st.markdown('<p class="main-header">🎤 Mock Interview Questions</p>', unsafe_allow_html=True)
        st.markdown("Practice with AI-generated interview questions tailored to your resume and target role.")
        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first, or click 🚀 Use Demo Resume in the sidebar!")
            return
        target_role = st.text_input("Target Role", placeholder="e.g., ML Engineer at Sarvam AI", key="int_role")
        num_q = st.slider("Number of questions", 5, 20, 10)
        if st.button("🎤 Generate Questions", type="primary"):
            if not target_role.strip(): st.error("Please enter a target role.")
            else:
                with st.spinner("Generating interview questions..."):
                    try:
                        result = generate_interview_questions(st.session_state.user_id, st.session_state.current_resume_id, target_role, num_q)
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["questions"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e: st.error(f"Error: {e.message}")
                    except Exception as e: st.error(f"Error: {safe_error(e)}")

    elif page == "📈 Fit Score":
        st.markdown('<p class="main-header">📈 Recruiter Fit Score</p>', unsafe_allow_html=True)
        st.markdown("Get a comprehensive recruiter-style evaluation of your resume against a job description.")
        if not st.session_state.current_resume_id:
            st.warning("Please upload and select a resume first, or click 🚀 Use Demo Resume in the sidebar!")
            return
        fit_jd_key = "_demo_fit_jd_loaded"
        job_desc = st.text_area("Job Description", height=250, placeholder="Paste the full job description here...", key="fit_jd")
        col_fit1, col_fit2 = st.columns([1, 3])
        with col_fit1:
            if st.button("📋 Load Demo JD", key="fit_load_demo_jd", help="Fill with a sample Senior ML Engineer job description"):
                st.session_state[fit_jd_key] = True
                st.rerun()
        with col_fit2:
            if st.session_state.get(fit_jd_key):
                jd_text = load_demo_jd_text()
                if jd_text:
                    job_desc = jd_text
                    st.info("✅ Demo job description loaded!")
        if st.button("📈 Calculate Fit Score", type="primary"):
            if not job_desc.strip(): st.error("Please enter a job description.")
            else:
                with st.spinner("Calculating recruiter fit score..."):
                    try:
                        result = recruiter_fit_score(st.session_state.user_id, st.session_state.current_resume_id, job_desc)
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.metric("Semantic Similarity Score", f"{result['semantic_similarity_score']:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown(result["evaluation"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    except AppError as e: st.error(f"Error: {e.message}")
                    except Exception as e: st.error(f"Error: {safe_error(e)}")

if __name__ == "__main__":
    run_streamlit()
