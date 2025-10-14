from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import Dict

app = FastAPI(
    title="EduSearch AI Backend",
    description="Intelligent Retrieval System for Higher Education Data",
    version="1.0.0"
)

# CORS middleware - allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
users_db: Dict[str, dict] = {}
uploaded_files = []

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {
        "message": "EduSearch AI Backend API is running!", 
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/login")
async def login_user(
    username: str = Form(...),
    password: str = Form(...)
):
    """Handle user login - accepts any credentials for demo"""
    try:
        # For demo purposes, accept any login
        if username and password:
            return {
                "success": True, 
                "message": "Login successful",
                "user": username
            }
        else:
            return {"success": False, "error": "Username and password required"}
    except Exception as e:
        return {"success": False, "error": f"Login error: {str(e)}"}

@app.post("/signup")
async def signup_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    """Handle user registration - always succeeds for demo"""
    try:
        if username in users_db:
            return {"success": False, "error": "Username already exists"}
        
        users_db[username] = {
            'email': email,
            'password': password
        }
        
        return {
            "success": True, 
            "message": "Registration successful",
            "user": username
        }
    except Exception as e:
        return {"success": False, "error": f"Registration error: {str(e)}"}

@app.post("/upload-file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """Handle file uploads - accepts any file for demo"""
    try:
        print(f"Received file upload: {file.filename}, Content-Type: {file.content_type}")
        
        # Check file type
        allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']
        if file.content_type not in allowed_types:
            return {
                "success": False,
                "error": f"File type {file.content_type} not supported. Please upload PDF, DOCX, or DOC files."
            }
        
        # Read file content (for demo, we just get the size)
        content = await file.read()
        file_size = len(content)
        
        # Store file info
        uploaded_files.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file_size,
            "uploaded_at": "now"
        })
        
        print(f"File processed successfully: {file.filename}, Size: {file_size} bytes")
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded and processed successfully!",
            "file_info": {
                "filename": file.filename,
                "size": file_size,
                "type": file.content_type
            }
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return {
            "success": False,
            "error": f"Error processing file: {str(e)}"
        }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat queries"""
    try:
        user_query = request.query
        
        if not user_query:
            return {"success": False, "error": "Query is required"}
        
        # Demo responses
        if "scholarship" in user_query.lower():
            response = "Based on available documents, scholarship eligibility typically requires: 1) Academic performance above 75%, 2) Family income below specified limits, 3) Admission to recognized institutions. Please check specific scholarship guidelines for detailed criteria."
        elif "admission" in user_query.lower():
            response = "Admission processes vary by institution. Common requirements include: entrance exam scores, academic transcripts, and application forms. Refer to individual university guidelines for specific admission procedures."
        elif "regulation" in user_query.lower():
            response = "Higher education regulations cover areas like curriculum standards, faculty qualifications, infrastructure requirements, and student welfare. Specific regulations depend on the governing bodies and institution types."
        else:
            response = f"I understand you're asking about: '{user_query}'. In a full implementation, this would search through uploaded higher education documents using RAG technology. Currently running in demo mode."
        
        return {
            "success": True,
            "query": user_query,
            "answer": response,
            "mode": "demo"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Chat error: {str(e)}"}

@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "users_count": len(users_db),
        "files_uploaded": len(uploaded_files),
        "uploaded_files": uploaded_files,
        "status": "operational",
        "mode": "demo"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
