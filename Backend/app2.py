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
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found. Running in demo mode.")

# Simple in-memory storage
users_db: Dict[str, dict] = {}
uploaded_files = []
current_document = None

class ChatRequest(BaseModel):
    query: str

# RAG Components
def initialize_rag_components():
    """Initialize RAG components only when needed"""
    try:
        from langchain_groq import ChatGroq
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        
        return {
            'ChatGroq': ChatGroq,
            'PyPDFLoader': PyPDFLoader,
            'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter,
            'FAISS': FAISS,
            'HuggingFaceEmbeddings': HuggingFaceEmbeddings,
            'ChatPromptTemplate': ChatPromptTemplate,
            'create_stuff_documents_chain': create_stuff_documents_chain,
            'create_retrieval_chain': create_retrieval_chain
        }
    except ImportError as e:
        print(f"RAG components not available: {e}")
        return None

rag_components = initialize_rag_components()
rag_chain = None

def process_document_with_rag(file_path: str):
    """Process document using RAG pipeline"""
    global rag_chain
    
    if not rag_components or not GROQ_API_KEY:
        return {"success": False, "error": "RAG components not available"}
    
    try:
        # Load document
        loader = rag_components['PyPDFLoader'](file_path)
        documents = loader.load()
        
        if not documents:
            return {"success": False, "error": "Could not extract content from document"}
        
        # Split documents
        text_splitter = rag_components['RecursiveCharacterTextSplitter'](
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = rag_components['HuggingFaceEmbeddings'](
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vectorstore = rag_components['FAISS'].from_documents(texts, embeddings)
        
        # Create LLM
        llm = rag_components['ChatGroq'](
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        # Create retrieval chain
        prompt = rag_components['ChatPromptTemplate'].from_template("""
        You are an expert assistant for higher education documents. Use the following context to answer the question.
        If you don't know the answer based on the context, say you don't know. Don't make up information.
        
        Context: {context}
        
        Question: {input}
        
        Answer:
        """)
        
        document_chain = rag_components['create_stuff_documents_chain'](llm, prompt)
        retriever = vectorstore.as_retriever()
        rag_chain = rag_components['create_retrieval_chain'](retriever, document_chain)
        
        return {"success": True, "message": "Document processed with RAG"}
        
    except Exception as e:
        return {"success": False, "error": f"RAG processing failed: {str(e)}"}

def get_rag_response(query: str):
    """Get response from RAG chain"""
    global rag_chain
    
    if not rag_chain:
        return {"success": False, "error": "No document has been processed yet"}
    
    try:
        response = rag_chain.invoke({"input": query})
        return {
            "success": True,
            "answer": response["answer"],
            "source": "RAG"
        }
    except Exception as e:
        return {"success": False, "error": f"RAG response failed: {str(e)}"}

@app.get("/")
async def root():
    return {
        "message": "EduSearch AI Backend API is running!", 
        "status": "healthy",
        "version": "1.0.0",
        "rag_available": rag_components is not None and GROQ_API_KEY is not None
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
        
        # Process with RAG if available
        if rag_components and GROQ_API_KEY:
            result = process_document_with_rag(tmp_path)
            if result["success"]:
                uploaded_files.append({
                    "filename": file.filename,
                    "processed": True,
                    "method": "RAG"
                })
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                return {
                    "success": True,
                    "message": f"File '{file.filename}' processed with RAG successfully!",
                    "method": "RAG"
                }
            else:
                # If RAG fails, fall back to demo mode
                print(f"RAG processing failed: {result['error']}")
        
        # Fallback to demo mode
        uploaded_files.append({
            "filename": file.filename,
            "processed": True,
            "method": "Demo"
        })
        
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully (Demo mode)",
            "method": "Demo"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Upload error: {str(e)}"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.query
        
        if not user_query:
            return {"success": False, "error": "Query is required"}
        
        # Try RAG first if available and document is processed
        if rag_chain:
            rag_response = get_rag_response(user_query)
            if rag_response["success"]:
                return {
                    "success": True,
                    "query": user_query,
                    "answer": rag_response["answer"],
                    "source": "RAG",
                    "method": "AI-Powered RAG"
                }
        
        # Fallback to demo responses
        if "scholarship" in user_query.lower():
            response = "Based on higher education documents, scholarship eligibility typically requires: 1) Academic performance above 75%, 2) Family income below ₹8 lakhs per annum, 3) Admission to recognized institutions, 4) No other concurrent scholarship benefits."
        elif "admission" in user_query.lower():
            response = "Admission processes include: entrance exam scores, academic transcripts, application forms, and sometimes interviews. Specific requirements vary by institution and program type."
        elif "regulation" in user_query.lower():
            response = "Higher education regulations cover curriculum standards, faculty qualifications, infrastructure requirements, research guidelines, and student welfare policies as per UGC and AICTE guidelines."
        elif "summary" in user_query.lower() or "summarize" in user_query.lower():
            response = "I would provide a comprehensive summary of the uploaded document here. In demo mode, I can give general information about higher education topics."
        else:
            response = f"I understand you're asking about: '{user_query}'. Based on the uploaded document, I would provide specific information. Currently running in demo mode with general higher education knowledge."
        
        return {
            "success": True,
            "query": user_query,
            "answer": response,
            "source": "Demo",
            "method": "Demo Mode"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Chat error: {str(e)}"}

@app.get("/status")
async def get_status():
    return {
        "users_count": len(users_db),
        "files_uploaded": len(uploaded_files),
        "rag_available": rag_components is not None and GROQ_API_KEY is not None,
        "groq_configured": GROQ_API_KEY is not None,
        "status": "operational"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
