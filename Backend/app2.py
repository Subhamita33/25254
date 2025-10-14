import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import shutil
import tempfile
from typing import Dict
import uvicorn

# --- Configuration & Initialization ---

load_dotenv()

app = FastAPI(title="EduSearch AI Backend", description="Intelligent Retrieval System for Higher Education Data")

# Add CORS middleware - UPDATE WITH YOUR NETLIFY URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://your-actual-netlify-app.netlify.app",  # REPLACE WITH YOUR ACTUAL NETLIFY URL
        "https://*.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for GROQ API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Simple user storage (in production, use a database)
users_db: Dict[str, dict] = {}
# Global state to hold the RAG chain
rag_pipeline = {"chain": None, "llm_model": "llama-3.1-8b-instant"}

# --- Core RAG Functions ---

def process_and_index_file(file_path: str, file_type: str):
    """Loads, chunks, and indexes a single file."""
    all_documents = []

    try:
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type in ['docx', 'doc']:
            # Use PyPDFLoader as fallback or implement docx parsing
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        all_documents.extend(loader.load())

        if not all_documents:
            raise HTTPException(status_code=500, detail="Could not load any content from the file.")

        # Split the Document into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(all_documents)

        # Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create the FAISS vector store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Create the RAG Chain
        llm = ChatGroq(model_name=rag_pipeline["llm_model"], temperature=0)

        system_prompt = (
            "You are an expert Q&A assistant for the provided documents. "
            "Use the following retrieved context to answer the user's question. "
            "If you don't know the answer, just say that you don't know, and acknowledge that the answer is not in the document."
            "\n\nContext: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        rag_chain = create_retrieval_chain(retriever, document_chain)

        return rag_chain

    except Exception as e:
        print(f"Error in process_and_index_file: {str(e)}")
        raise

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    """Root endpoint showing API status"""
    return {
        "message": "EduSearch AI Backend API is running", 
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "EduSearch AI Backend is running"}

@app.post("/login")
async def login_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    """Handle user login."""
    try:
        # Simple authentication (in production, use proper hashing and database)
        user = users_db.get(username)
        if user and user['password'] == password:
            return {"success": True, "message": "Login successful"}
        else:
            return {"success": False, "error": "Invalid username or password"}
    except Exception as e:
        return {"success": False, "error": f"Login error: {str(e)}"}

@app.post("/signup")
async def signup_user(
    request: Request,
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
            'password': password  # In production, hash this password
        }
        
        return {"success": True, "message": "Registration successful"}
    except Exception as e:
        return {"success": False, "error": f"Registration error: {str(e)}"}

@app.post("/upload-file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """Receives an uploaded file, saves it, indexes it, and updates the RAG chain."""
    
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ['pdf', 'docx', 'doc']:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and DOC files are supported.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        new_rag_chain = process_and_index_file(tmp_file_path, file_extension)
        rag_pipeline["chain"] = new_rag_chain
        
        return {"message": f"File '{file.filename}' uploaded and RAG index successfully created."}

    except Exception as e:
        print(f"Indexing Error: {e}")
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file and create RAG index: {str(e)}")

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/chat")
async def chat_endpoint(query: dict):
    """Receives a text query and returns an answer using the current RAG chain."""
    try:
        user_query = query.get("query")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query text is required.")

        if rag_pipeline["chain"] is None:
            raise HTTPException(status_code=400, detail="No document has been uploaded and indexed yet. Please upload a file first.")

        response = rag_pipeline["chain"].invoke({"input": user_query})
        return {"query": user_query, "answer": response["answer"]}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
