import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="EduSearch AI Backend",
    description="Intelligent Retrieval System for Higher Education Data",
    version="1.0.0"
)

# CORS middleware - allow all for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
users_db = {}
uploaded_files = []

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {
        "message": "EduSearch AI Backend API is running successfully!", 
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
    """Simple login - accepts any credentials for demo"""
    try:
        # For demo, accept any login
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
    """Simple signup - always succeeds for demo"""
    try:
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
    """Accept file upload - store filename for demo"""
    try:
        # Just store the filename for demo
        uploaded_files.append({
            "filename": file.filename,
            "size": "demo",
            "uploaded_at": "now"
        })
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully (demo mode)",
            "files_count": len(uploaded_files)
        }
    except Exception as e:
        return {"success": False, "error": f"Upload error: {str(e)}"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Simple chat response for demo"""
    try:
        user_query = request.query
        
        if not user_query:
            return {"success": False, "error": "Query is required"}
        
        # Demo responses based on query
        if "scholarship" in user_query.lower():
            response = "Based on available documents, scholarship eligibility typically requires: 1) Academic performance above 75%, 2) Family income below specified limits, 3) Admission to recognized institutions. Please check specific scholarship guidelines for detailed criteria."
        elif "admission" in user_query.lower():
            response = "Admission processes vary by institution. Common requirements include: entrance exam scores, academic transcripts, and application forms. Refer to individual university guidelines for specific admission procedures."
        elif "regulation" in user_query.lower():
            response = "Higher education regulations cover areas like curriculum standards, faculty qualifications, infrastructure requirements, and student welfare. Specific regulations depend on the governing bodies and institution types."
        else:
            response = f"I understand you're asking about: '{user_query}'. In a full implementation, this would search through uploaded higher education documents using AI technology. Currently running in demo mode."
        
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
        "status": "operational",
        "mode": "demo"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
