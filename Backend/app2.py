from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="EduSearch AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

users_db = {}
uploaded_files = []

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "EduSearch AI Backend is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    return {"success": True, "message": "Login successful"}

@app.post("/signup")
async def signup(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    users_db[username] = {"email": email, "password": password}
    return {"success": True, "message": "Signup successful"}

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    uploaded_files.append(file.filename)
    return {"success": True, "message": f"File {file.filename} uploaded"}

@app.post("/chat")
async def chat(request: ChatRequest):
    response = f"Demo response to: {request.query}"
    return {"success": True, "query": request.query, "answer": response}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
