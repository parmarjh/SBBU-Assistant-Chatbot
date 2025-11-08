# 📦 SBBU Chatbot - Complete Repository Download Guide

## 🚀 Quick Download Options

### Option 1: Download as ZIP (Recommended for Quick Start)

Since this is an artifact-based project, here's how to set it up:

1. **Create Project Directory**
```bash
mkdir sbbu-chatbot
cd sbbu-chatbot
```

2. **Create Folder Structure**
```bash
mkdir -p frontend/src/components
mkdir -p frontend/src/utils
mkdir -p backend/models
mkdir -p backend/services
mkdir -p backend/utils
mkdir -p docs
mkdir -p tests
```

---

## 📁 Complete File Structure

```
sbbu-chatbot/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── Chatbot.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── tailwind.config.js
├── backend/
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── document.py
│   │   └── rag.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── anthropic_service.py
│   │   ├── openai_service.py
│   │   ├── vector_db.py
│   │   └── search.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── chunking.py
│   │   └── cache.py
│   └── requirements.txt
├── docs/
│   ├── API.md
│   ├── ALGORITHMS.md
│   └── DEPLOYMENT.md
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   └── test_api.py
├── .env.example
├── .gitignore
├── README.md
├── LICENSE
└── docker-compose.yml
```

---

## 📝 File Contents

### 1. **frontend/package.json**
```json
{
  "name": "sbbu-chatbot-frontend",
  "version": "1.0.0",
  "description": "SBBU Assistant Chatbot Frontend",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "lucide-react": "^0.263.1"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.3.6",
    "vite": "^5.0.8"
  }
}
```

### 2. **frontend/vite.config.js**
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
```

### 3. **frontend/tailwind.config.js**
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### 4. **frontend/index.html**
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>SBBU Assistant Chatbot</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

### 5. **frontend/src/main.jsx**
```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

### 6. **frontend/src/App.jsx**
```jsx
import React from 'react'
import Chatbot from './components/Chatbot'

function App() {
  return (
    <div className="App">
      <Chatbot />
    </div>
  )
}

export default App
```

### 7. **frontend/src/index.css**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

### 8. **backend/requirements.txt**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
langchain==0.1.0
langchain-anthropic==0.1.0
langchain-openai==0.0.5
langchain-google-genai==0.0.5
chromadb==0.4.18
python-dotenv==1.0.0
sqlalchemy==2.0.23
pandas==2.1.3
pypdf2==3.0.1
python-docx==1.1.0
python-multipart==0.0.6
beautifulsoup4==4.12.2
tavily-python==0.3.0
pydantic==2.5.0
httpx==0.25.2
```

### 9. **backend/main.py**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SBBU Assistant Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str
    provider: str

class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str

# Routes
@app.get("/")
async def root():
    return {"message": "SBBU Assistant Chatbot API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Import provider services
        if request.provider == "anthropic":
            from services.anthropic_service import get_anthropic_response
            response = await get_anthropic_response(request.messages, request.model)
        elif request.provider == "openai":
            from services.openai_service import get_openai_response
            response = await get_openai_response(request.messages, request.model)
        else:
            response = f"Response from {request.provider} - {request.model}"
        
        return ChatResponse(
            content=response,
            model=request.model,
            provider=request.provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        contents = await file.read()
        file_path = f"uploads/{file.filename}"
        
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Process and index document
        from services.vector_db import index_document
        await index_document(file_path)
        
        return {"filename": file.filename, "status": "uploaded and indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    try:
        file_path = f"uploads/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "deleted"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 10. **backend/services/anthropic_service.py**
```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

async def get_anthropic_response(messages, model):
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": m.role, "content": m.content} for m in messages]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"
```

### 11. **backend/services/openai_service.py**
```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_openai_response(messages, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
```

### 12. **backend/services/vector_db.py**
```python
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = client.get_or_create_collection(name="sbbu_documents")

async def index_document(file_path: str):
    """Index document into vector database"""
    # Read and chunk document
    from utils.chunking import chunk_document
    chunks = chunk_document(file_path)
    
    # Generate embeddings and store
    collection.add(
        documents=chunks,
        ids=[f"{file_path}_{i}" for i in range(len(chunks))]
    )

async def search_documents(query: str, n_results: int = 5):
    """Search for relevant documents"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results
```

### 13. **backend/utils/chunking.py**
```python
def chunk_document(file_path: str, chunk_size: int = 500, overlap: int = 50):
    """Chunk document into smaller pieces"""
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks
```

### 14. **backend/utils/cache.py**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Global cache instance
response_cache = LRUCache(capacity=100)
```

### 15. **.env.example**
```env
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Database
DATABASE_URL=sqlite:///./sbbu_chatbot.db

# Server
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_URL=http://localhost:3000
```

### 16. **.gitignore**
```
# Dependencies
node_modules/
venv/
__pycache__/
*.py[cod]
*$py.class

# Environment
.env
.env.local

# Build
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Database
*.db
*.sqlite
chroma_db/

# Uploads
uploads/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
```

### 17. **LICENSE**
```
MIT License

Copyright (c) 2024 SBBU - Shaheed Benazir Bhutto University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 18. **docker-compose.yml**
```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./uploads:/app/uploads
    environment:
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 19. **backend/Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 20. **frontend/Dockerfile**
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"]
```

---

## 🚀 Installation Steps

### Step 1: Clone/Download Files

Copy the React component from the artifact above and save as:
- `frontend/src/components/Chatbot.jsx`

### Step 2: Install Frontend

```bash
cd frontend
npm install
```

### Step 3: Install Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Step 5: Run Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Step 6: Access Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🐳 Docker Deployment (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Stop containers
docker-compose down
```

---

## 📦 Create Downloadable ZIP

```bash
# Create ZIP archive
zip -r sbbu-chatbot.zip sbbu-chatbot/ \
  -x "*/node_modules/*" \
  -x "*/venv/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x ".git/*"
```

---

## 🔗 GitHub Repository Setup

```bash
# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: SBBU Assistant Chatbot"

# Add remote
git remote add origin https://github.com/yourusername/sbbu-chatbot.git

# Push
git push -u origin main
```

---

## 📞 Support

For issues or questions:
- Email: support@sbbu.edu.pk
- GitHub Issues: https://github.com/sbbu/chatbot/issues

---

**Happy Coding! 🚀**