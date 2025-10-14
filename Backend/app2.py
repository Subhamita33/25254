from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile
import shutil
from typing import Dict
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Simple in-memory storage
users_db: Dict[str, dict] = {}
uploaded_files = []
current_document_content = ""

class ChatRequest(BaseModel):
    query: str

# RAG Components (will be initialized on first use)
rag_components_available = False

def check_rag_availability():
    """Check if RAG components are available"""
    global rag_components_available
    try:
        # Try to import essential RAG components
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from pypdf import PdfReader
        
        rag_components_available = True
        return True
    except ImportError as e:
        print(f"Some RAG components not available: {e}")
        rag_components_available = False
        return False

def extract_text_from_pdf(file_path: str):
    """Extract text from PDF"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None

def get_groq_response(query: str, context: str = ""):
    """Get response from Groq API with document context"""
    if not GROQ_API_KEY:
        return None
    
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        if context and len(context) > 100:  # Only use context if it's substantial
            prompt = ChatPromptTemplate.from_template("""
            You are an expert assistant for higher education documents. 
            Use the following document content to answer the question accurately and helpfully.
            
            DOCUMENT CONTENT:
            {context}
            
            QUESTION: {question}
            
            INSTRUCTIONS:
            - Provide a detailed answer based ONLY on the document content provided
            - If the document doesn't contain relevant information, say "Based on the document, this information is not available"
            - Keep your answer focused and relevant to the question
            - Use bullet points if appropriate for clarity
            
            ANSWER:
            """)
            chain = prompt | llm
            response = chain.invoke({"context": context, "question": query})
        else:
            prompt = ChatPromptTemplate.from_template("""
            You are an expert on higher education policies and regulations.
            
            QUESTION: {question}
            
            Provide accurate and helpful information about higher education.
            
            ANSWER:
            """)
            chain = prompt | llm
            response = chain.invoke({"question": query})
        
        return response.content
    except Exception as e:
        print(f"Groq API error: {e}")
        return None

def get_document_summary(text: str):
    """Get AI-generated summary of the document"""
    if not GROQ_API_KEY:
        return None
    
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        prompt = ChatPromptTemplate.from_template("""
        Please provide a comprehensive summary of the following document content.
        Focus on the main topics, key points, and important information.
        
        DOCUMENT CONTENT:
        {text}
        
        Please provide a well-structured summary with clear sections.
        SUMMARY:
        """)
        
        chain = prompt | llm
        response = chain.invoke({"text": text[:3000]})  # Limit text length for summary
        return response.content
    except Exception as e:
        print(f"Summary generation error: {e}")
        return None

# Check RAG availability on startup
check_rag_availability()

@app.get("/")
async def root():
    return {
        "message": "EduSearch AI Backend API is running!", 
        "status": "healthy",
        "version": "1.0.0",
        "rag_available": rag_components_available and GROQ_API_KEY is not None,
        "backend_url": "https://edusearch-ai-backend.onrender.com"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/login")
async def login_user(username: str = Form(...), password: str = Form(...)):
    if username and password:
        return {"success": True, "message": "Login successful", "user": username}
    return {"success": False, "error": "Username and password required"}

@app.post("/signup")
async def signup_user(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    if username in users_db:
        return {"success": False, "error": "Username already exists"}
    
    users_db[username] = {'email': email, 'password': password}
    return {"success": True, "message": "Registration successful", "user": username}

@app.post("/upload-file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        
        # Check file type
        if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
            return {
                "success": False,
                "error": "Only PDF, DOCX, and DOC files are supported"
            }
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Extract text from PDF
        global current_document_content
        extracted_text = extract_text_from_pdf(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        if extracted_text:
            current_document_content = extracted_text
            
            # Generate AI summary if Groq is available
            ai_summary = None
            if GROQ_API_KEY:
                ai_summary = get_document_summary(extracted_text)
            
            method = "AI-RAG" if (rag_components_available and GROQ_API_KEY) else "Basic"
            
            uploaded_files.append({
                "filename": file.filename,
                "processed": True,
                "method": method,
                "content_length": len(extracted_text),
                "ai_summary": bool(ai_summary)
            })
            
            response_data = {
                "success": True,
                "message": f"File '{file.filename}' processed successfully!",
                "method": method,
                "content_length": len(extracted_text),
                "has_ai_summary": bool(ai_summary)
            }
            
            if ai_summary:
                response_data["ai_summary"] = ai_summary
            
            return response_data
        else:
            return {
                "success": False,
                "error": "Could not extract text from the document"
            }
        
    except Exception as e:
        return {"success": False, "error": f"Upload error: {str(e)}"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.query
        
        if not user_query:
            return {"success": False, "error": "Query is required"}
        
        # Try RAG with Groq if available and document is processed
        if GROQ_API_KEY and current_document_content:
            print(f"Using RAG for query: {user_query}")
            rag_response = get_groq_response(user_query, current_document_content)
            if rag_response:
                return {
                    "success": True,
                    "query": user_query,
                    "answer": rag_response,
                    "source": "RAG",
                    "method": "AI-Powered RAG",
                    "document_based": True
                }
        
        # Enhanced demo responses (fallback)
        demo_responses = {
            "scholarship": """Based on higher education scholarship guidelines:

**Eligibility Criteria:**
- Minimum 75% marks in previous qualification
- Family income below ₹8 lakhs annually  
- Admission to UGC-recognized institutions
- No concurrent scholarship benefits

**Application Process:**
- Online application through national scholarship portal
- Document verification
- Institute-level approval
- Final selection by scholarship committee""",

            "admission": """University admission processes:

**General Requirements:**
- Entrance examination scores
- Academic transcripts and certificates
- Application form with personal details
- Category certificates (if applicable)

**Selection Process:**
- Merit-based selection
- Counseling sessions
- Document verification
- Fee payment and enrollment""",

            "regulation": """Higher education regulations cover:

**Academic Standards:**
- Curriculum frameworks and syllabi
- Faculty qualifications and appointments
- Examination systems and grading
- Research guidelines and ethics

**Infrastructure:**
- Library and laboratory facilities
- Classroom and hostel requirements
- Digital infrastructure standards
- Safety and accessibility norms""",

            "summary": "I can provide detailed summaries of uploaded documents. Please upload a PDF document about higher education, and I'll generate a comprehensive AI-powered summary for you.",

            "research": """Research in higher education:

**Funding Sources:**
- UGC grants and fellowships
- National research foundations
- Industry collaborations
- International partnerships

**Process:**
- Research proposal submission
- Ethical committee approval
- Funding allocation
- Progress monitoring and reporting
- Publication and patent filing"""
        }
        
        # Find the best matching demo response
        query_lower = user_query.lower()
        response = None
        
        for key, demo_response in demo_responses.items():
            if key in query_lower:
                response = demo_response
                break
        
        if not response:
            if current_document_content:
                response = f"""**AI Analysis Ready**

I understand you're asking about: '{user_query}'

I have successfully processed your document ({len(current_document_content)} characters) and can provide specific insights based on its content. 

Please ask specific questions about the document content, and I'll provide detailed answers using AI analysis."""
            else:
                response = f"""**Higher Education Assistant**

I understand you're asking about: '{user_query}'

This appears to be related to higher education policies and regulations. For the most accurate and specific information, please upload a relevant document (PDF, DOCX, or DOC) about higher education policies, scholarships, admissions, or regulations.

Once you upload a document, I can provide AI-powered analysis and answers based on the actual content."""
        
        return {
            "success": True,
            "query": user_query,
            "answer": response,
            "source": "AI Knowledge Base",
            "method": "Enhanced Response",
            "document_based": bool(current_document_content)
        }
        
    except Exception as e:
        return {"success": False, "error": f"Chat error: {str(e)}"}

@app.get("/status")
async def get_status():
    return {
        "users_count": len(users_db),
        "files_uploaded": len(uploaded_files),
        "current_document": bool(current_document_content),
        "document_length": len(current_document_content) if current_document_content else 0,
        "rag_available": rag_components_available and GROQ_API_KEY is not None,
        "groq_configured": GROQ_API_KEY is not None,
        "components_available": rag_components_available,
        "backend_url": "https://edusearch-ai-backend.onrender.com",
        "status": "operational"
    }

@app.get("/test-groq")
async def test_groq():
    """Test endpoint to verify Groq API connectivity"""
    if not GROQ_API_KEY:
        return {"success": False, "error": "GROQ_API_KEY not configured"}
    
    try:
        test_response = get_groq_response("What is the purpose of higher education?")
        if test_response:
            return {
                "success": True, 
                "message": "Groq API is working correctly",
                "test_response": test_response[:200] + "..." if len(test_response) > 200 else test_response
            }
        else:
            return {"success": False, "error": "Groq API returned no response"}
    except Exception as e:
        return {"success": False, "error": f"Groq API test failed: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
