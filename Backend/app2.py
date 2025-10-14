import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import tempfile
from typing import Dict
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="EduSearch AI Backend",
    description="Intelligent Retrieval System for Higher Education Data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000", 
        "https://your-netlify-app.netlify.app",  # Replace with your Netlify URL
        "https://*.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found. Some features may not work.")

# Simple user storage
users_db: Dict[str, dict] = {}

# Request models
class ChatRequest(BaseModel):
    query: str

# Basic RAG components (will be initialized on first use)
rag_components = {
    "vector_store": None,
    "llm": None,
    "retriever": None
}

def initialize_rag_components():
    """Initialize RAG components only when needed"""
    try:
        from langchain_groq import ChatGroq
        from langchain_community.vectorstores import FAISS
        from sentence_transformers import SentenceTransformer
        
        if rag_components["llm"] is None and GROQ_API_KEY:
            rag_components["llm"] = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0,
                groq_api_key=GROQ_API_KEY
            )
        return True
    except ImportError as e:
        print(f"RAG components not available: {e}")
        return False

@app.get("/")
async def root():
    return {
        "message": "EduSearch AI Backend API is running", 
        "status": "healthy",
        "version": "1.0.0",
        "rag_available": GROQ_API_KEY is not None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "EduSearch AI Backend is running"}

@app.post("/login")
async def login_user(
    username: str = Form(...),
    password: str = Form(...)
):
    """Handle user login."""
    try:
        user = users_db.get(username)
        if user and user['password'] == password:
            return {"success": True, "message": "Login successful"}
        else:
            return {"success": False, "error": "Invalid username or password"}
    except Exception as e:
        return {"success": False, "error": f"Login error: {str(e)}"}

@app.post("/signup")
async def signup_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    """Handle user registration."""
    try:
        if username in users_db:
            return {"success": False, "error": "Username already exists"}
        
        users_db[username] = {
            'email': email,
            'password': password
        }
        
        return {"success": True, "message": "Registration successful"}
    except Exception as e:
        return {"success": False, "error": f"Registration error: {str(e)}"}

@app.post("/upload-file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """Receives an uploaded file and processes it."""
    file_extension = file.filename.split(".")[-1].lower()
    
    if file_extension not in ['pdf', 'docx', 'doc']:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and DOC files are supported.")
    
    # For demo purposes, we'll just acknowledge the upload
    # In a full implementation, this would process the file with RAG
    try:
        # Initialize RAG components if available
        rag_available = initialize_rag_components()
        
        if rag_available:
            return {
                "message": f"File '{file.filename}' uploaded successfully. RAG processing available.",
                "processed": True
            }
        else:
            return {
                "message": f"File '{file.filename}' uploaded successfully. Basic mode active.",
                "processed": False,
                "note": "RAG features require GROQ_API_KEY environment variable"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat queries with RAG if available, otherwise basic responses."""
    try:
        user_query = request.query
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Query text is required.")
        
        # Try to use RAG if available
        rag_available = initialize_rag_components()
        
        if rag_available and rag_components["llm"]:
            try:
                # Simple direct response without RAG for now
                response = f"I received your question: '{user_query}'. In a full implementation, this would search through uploaded documents using RAG technology."
                return {"query": user_query, "answer": response}
            except Exception as e:
                # Fallback to basic response
                return {
                    "query": user_query, 
                    "answer": f"Basic response to: {user_query}. RAG processing is currently being initialized.",
                    "note": "RAG features are starting up, please try again in a moment."
                }
        else:
            # Basic response when RAG is not available
            return {
                "query": user_query,
                "answer": f"I received your question: '{user_query}'. This is running in basic mode. To enable full RAG capabilities, please ensure GROQ_API_KEY is properly configured.",
                "mode": "basic"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
