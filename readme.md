# SBBU Assistant Chatbot 🤖

## Shaheed Benazir Bhutto University - Sanghar Campus

An intelligent AI-powered chatbot that assists SBBU students and faculty by answering queries related to university academics, syllabus, faculty details, and more using AI-driven retrieval from uploaded documents.

---

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Algorithms & Data Structures](#algorithms--data-structures)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Team Members](#team-members)

---

## ✨ Features

- **Agentic AI System**: Built with LangGraph to control tool routing and reasoning flow
- **Retrieval Augmented Generation (RAG)**: Fetches accurate answers from uploaded PDFs, DOCX, or HTML documents
- **Context-Aware Chat**: Uses SQLite to store and recall chat sessions
- **Document Upload & Deletion**: Dynamically index or remove files in Chroma Vector DB
- **Fast Reasoning**: Powered by multiple LLM models (Claude, GPT, Gemini, etc.)
- **Web Search Integration**: Uses Tavily Search API for up-to-date online information
- **Voice Interaction**: Speech-to-text input and text-to-speech output
- **Multi-Model Support**: Switch between 7+ AI providers and 15+ models
- **Interactive Frontend**: Streamlit/React-based user interface

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│              (React + Tailwind CSS + Lucide Icons)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing Layer                    │
│         • Text Input Handler                                 │
│         • Speech Recognition (Web Speech API)                │
│         • Document Upload Handler                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Orchestration Layer                    │
│         • Model Selection Router                             │
│         • Context Management (Message History)               │
│         • RAG Pipeline Controller                            │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   AI Models  │  │  Vector DB   │  │ Web Search   │
│   • Claude   │  │  • Chroma    │  │  • Tavily    │
│   • GPT      │  │  • Embeddings│  │  API         │
│   • Gemini   │  │  • Indexing  │  │              │
│   • Groq     │  │              │  │              │
│   • DeepSeek │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Response Generation                       │
│         • Text Formatting                                    │
│         • Speech Synthesis (TTS)                             │
│         • Citation Management                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage & Persistence                     │
│         • Chat History (In-Memory State)                     │
│         • Document Metadata (Array Storage)                  │
│         • Session Management                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧮 Algorithms & Data Structures

### 1. **Message Management - Queue (FIFO)**

**Data Structure**: Array-based Queue

```javascript
// Message queue implementation
messages = [
  { role: 'assistant', content: '...' },
  { role: 'user', content: '...' },
  { role: 'assistant', content: '...' }
]

// Add new message (Enqueue)
setMessages(prev => [...prev, newMessage])  // O(n) time, O(1) amortized
```

**Time Complexity**:
- Add message: O(1) amortized
- Retrieve all messages: O(n)
- Display messages: O(n)

**Use Case**: Maintains conversation history in chronological order

---

### 2. **Document Storage - Dynamic Array**

**Data Structure**: Array with Dynamic Resizing

```javascript
documents = [
  { id: 123456, name: 'syllabus.pdf', size: '2.5 MB', type: 'application/pdf' },
  { id: 123457, name: 'notes.docx', size: '1.2 MB', type: 'application/docx' }
]

// Add document - O(1) amortized
setDocuments([...documents, newDoc])

// Remove document - O(n)
setDocuments(documents.filter(doc => doc.id !== targetId))

// Search document - O(n)
const doc = documents.find(doc => doc.id === targetId)
```

**Time Complexity**:
- Insert: O(1) amortized
- Delete: O(n)
- Search: O(n)

**Space Complexity**: O(n) where n = number of documents

---

### 3. **Vector Similarity Search (RAG) - k-NN Algorithm**

**Algorithm**: k-Nearest Neighbors with Cosine Similarity

```python
# Pseudo-code for RAG retrieval
def retrieve_relevant_chunks(query, k=5):
    # 1. Vectorize query
    query_embedding = embed(query)  # O(d) where d = embedding dimension
    
    # 2. Calculate cosine similarity
    similarities = []
    for doc_chunk in vector_db:  # O(n)
        similarity = cosine_similarity(query_embedding, doc_chunk.embedding)
        similarities.append((doc_chunk, similarity))
    
    # 3. Sort and get top-k
    similarities.sort(key=lambda x: x[1], reverse=True)  # O(n log n)
    top_k = similarities[:k]  # O(k)
    
    return [chunk for chunk, score in top_k]
```

**Time Complexity**:
- Embedding generation: O(d) where d = embedding dimension
- Similarity calculation: O(n × d) where n = number of chunks
- Sorting: O(n log n)
- **Total**: O(n log n + n×d)

**Space Complexity**: O(n × d)

**Optimization**: Using approximate nearest neighbor (ANN) algorithms like HNSW can reduce to O(log n)

---

### 4. **LLM Context Window Management - Sliding Window**

**Algorithm**: Sliding Window for Context Truncation

```javascript
function manageContextWindow(messages, maxTokens = 4096) {
    let tokens = 0;
    let contextMessages = [];
    
    // Start from most recent (LIFO approach)
    for (let i = messages.length - 1; i >= 0; i--) {
        const messageTokens = estimateTokens(messages[i].content);
        
        if (tokens + messageTokens <= maxTokens) {
            contextMessages.unshift(messages[i]);  // Add to front
            tokens += messageTokens;
        } else {
            break;  // Context window full
        }
    }
    
    return contextMessages;
}
```

**Time Complexity**: O(n) where n = number of messages
**Space Complexity**: O(k) where k = messages within token limit

---

### 5. **Model Router - Strategy Pattern**

**Design Pattern**: Strategy Pattern with Hash Map

```javascript
const modelStrategies = {
    'anthropic': async (messages, model) => {
        return await callAnthropicAPI(messages, model);
    },
    'openai': async (messages, model) => {
        return await callOpenAIAPI(messages, model);
    },
    'groq': async (messages, model) => {
        return await callGroqAPI(messages, model);
    },
    // ... other providers
};

// O(1) lookup and execution
const response = await modelStrategies[apiProvider](messages, model);
```

**Time Complexity**: O(1) for provider selection
**Space Complexity**: O(p) where p = number of providers

---

### 6. **Speech Recognition - Finite State Machine (FSM)**

**Algorithm**: State Machine for Voice Input

```
States:
┌─────────┐  start()   ┌─────────────┐  onresult  ┌─────────────┐
│  IDLE   │ ────────▶  │  LISTENING  │ ────────▶  │  PROCESSING │
└─────────┘            └─────────────┘            └─────────────┘
     ▲                        │                           │
     │                  stop() │                           │
     │                        ▼                           │
     │                 ┌─────────────┐                    │
     └─────────────────│   STOPPED   │◀───────────────────┘
                       └─────────────┘
```

**Time Complexity**: O(1) for state transitions
**Space Complexity**: O(1)

---

### 7. **Document Indexing - Inverted Index**

**Data Structure**: Hash Map (Inverted Index)

```javascript
// Inverted index for document search
invertedIndex = {
    'machine': [doc1, doc3, doc5],
    'learning': [doc1, doc2, doc3],
    'algorithm': [doc2, doc4, doc5],
    'neural': [doc3, doc5]
}

// Search query: "machine learning"
function searchDocuments(query) {
    const terms = tokenize(query);  // O(m) where m = query length
    const docSets = terms.map(term => invertedIndex[term] || []);  // O(k)
    
    // Find intersection of document sets
    return intersection(...docSets);  // O(n × k)
}
```

**Time Complexity**:
- Indexing: O(n × m) where n = docs, m = avg terms per doc
- Search: O(k × n) where k = query terms, n = matching docs

---

### 8. **Caching - LRU Cache for API Responses**

**Data Structure**: Doubly Linked List + Hash Map

```javascript
class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();  // Hash map for O(1) access
        this.dll = new DoublyLinkedList();  // For LRU ordering
    }
    
    get(key) {  // O(1)
        if (!this.cache.has(key)) return null;
        
        // Move to front (most recently used)
        this.dll.moveToFront(key);
        return this.cache.get(key);
    }
    
    put(key, value) {  // O(1)
        if (this.cache.size >= this.capacity) {
            // Evict least recently used
            const lru = this.dll.removeLast();
            this.cache.delete(lru);
        }
        
        this.cache.set(key, value);
        this.dll.addToFront(key);
    }
}
```

**Time Complexity**: O(1) for both get and put
**Space Complexity**: O(n) where n = cache capacity

---

### 9. **Rate Limiting - Token Bucket Algorithm**

**Algorithm**: Token Bucket for API Rate Limiting

```javascript
class TokenBucket {
    constructor(capacity, refillRate) {
        this.capacity = capacity;  // Max tokens
        this.tokens = capacity;
        this.refillRate = refillRate;  // Tokens per second
        this.lastRefill = Date.now();
    }
    
    allowRequest() {  // O(1)
        this.refill();
        
        if (this.tokens >= 1) {
            this.tokens -= 1;
            return true;
        }
        return false;
    }
    
    refill() {  // O(1)
        const now = Date.now();
        const elapsed = (now - this.lastRefill) / 1000;
        const tokensToAdd = elapsed * this.refillRate;
        
        this.tokens = Math.min(this.capacity, this.tokens + tokensToAdd);
        this.lastRefill = now;
    }
}
```

**Time Complexity**: O(1)
**Space Complexity**: O(1)

---

### 10. **Text Chunking - Sliding Window with Overlap**

**Algorithm**: Overlapping Sliding Window for Document Chunking

```python
def chunk_document(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move window with overlap
        start += (chunk_size - overlap)
    
    return chunks
```

**Time Complexity**: O(n) where n = text length
**Space Complexity**: O(n)

---

## 🛠️ Technologies Used

### **Frontend**
- **React** 18.2.0 - UI library
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

### **AI/ML**
- **Python** 3.11+
- **LangChain** - LLM orchestration
- **LangGraph** - Agent workflow
- **ChromaDB** - Vector database
- **OpenAI Embeddings** - Text vectorization

### **APIs**
- **Anthropic Claude API** - Claude models
- **OpenAI API** - GPT models
- **Groq API** - Fast inference
- **Google Gemini API** - Gemini models
- **DeepSeek API** - DeepSeek models
- **Tavily Search API** - Web search

### **Backend**
- **FastAPI** - REST API server
- **SQLite** - Chat history storage
- **Streamlit** - Interactive dashboard

### **Voice**
- **Web Speech API** - Speech recognition
- **Speech Synthesis API** - Text-to-speech

---

## 📦 Installation

### Prerequisites
```bash
Node.js >= 18.0.0
Python >= 3.11
npm or yarn
```

### Frontend Setup

```bash
# Clone the repository
git clone https://github.com/sbbu/chatbot.git
cd chatbot

# Install dependencies
npm install

# Install required packages
npm install lucide-react react react-dom

# Start development server
npm run dev
```

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
GROQ_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_api_key_here
DEEPSEEK_API_KEY=your_api_key_here
TAVILY_API_KEY=your_api_key_here
EOF

# Run FastAPI server
uvicorn main:app --reload
```

### requirements.txt
```txt
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-anthropic==0.1.0
langchain-openai==0.0.2
chromadb==0.4.18
python-dotenv==1.0.0
sqlalchemy==2.0.23
pandas==2.1.3
pypdf2==3.0.1
python-docx==1.1.0
beautifulsoup4==4.12.2
tavily-python==0.3.0
```

---

## 🚀 Usage

### Starting the Application

1. **Start Backend Server**:
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

2. **Start Frontend**:
```bash
cd frontend
npm run dev
```

3. **Access Application**:
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

### Basic Usage

1. **Upload Documents**:
   - Click upload icon in sidebar
   - Select PDF, DOCX, or HTML files
   - Documents are automatically indexed

2. **Ask Questions**:
   - Type or speak your question
   - Press Enter or click Send
   - Get AI-powered responses

3. **Switch Models**:
   - Select AI Provider from dropdown
   - Choose specific model
   - Continue conversation

4. **Voice Interaction**:
   - Enable "Voice Response" in settings
   - Click microphone to speak
   - Hear responses automatically

---

## 🔌 API Integration

### Anthropic Claude
```javascript
const response = await fetch('https://api.anthropic.com/v1/messages', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': ANTHROPIC_API_KEY
  },
  body: JSON.stringify({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1000,
    messages: messages
  })
});
```

### OpenAI GPT
```javascript
const response = await fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${OPENAI_API_KEY}`
  },
  body: JSON.stringify({
    model: 'gpt-4',
    messages: messages
  })
});
```

---

## 📁 Project Structure

```
sbbu-chatbot/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chatbot.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   ├── MessageList.jsx
│   │   │   └── InputArea.jsx
│   │   ├── utils/
│   │   │   ├── api.js
│   │   │   └── speech.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── models/
│   │   ├── chat.py          # Chat models
│   │   ├── document.py      # Document models
│   │   └── rag.py           # RAG pipeline
│   ├── services/
│   │   ├── anthropic.py     # Claude integration
│   │   ├── openai.py        # GPT integration
│   │   ├── vector_db.py     # ChromaDB operations
│   │   └── search.py        # Tavily search
│   ├── utils/
│   │   ├── embeddings.py    # Text embeddings
│   │   ├── chunking.py      # Document chunking
│   │   └── cache.py         # LRU cache
│   └── requirements.txt
├── docs/
│   ├── API.md
│   ├── ALGORITHMS.md
│   └── DEPLOYMENT.md
├── tests/
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_algorithms.py
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🧪 Testing

```bash
# Run frontend tests
npm test

# Run backend tests
pytest tests/

# Test specific algorithm
pytest tests/test_algorithms.py::test_lru_cache

# Test API endpoints
pytest tests/test_api.py -v
```

---

## 🔧 Configuration

### Model Configuration
```javascript
// config.js
export const MODEL_CONFIG = {
  anthropic: {
    models: ['claude-sonnet-4', 'claude-opus-4'],
    maxTokens: 4096,
    temperature: 0.7
  },
  openai: {
    models: ['gpt-4', 'gpt-4-turbo', 'gpt-5'],
    maxTokens: 8192,
    temperature: 0.7
  }
  // ... other providers
};
```

### Vector Database Configuration
```python
# vector_db_config.py
CHROMA_CONFIG = {
    'collection_name': 'sbbu_documents',
    'embedding_model': 'text-embedding-ada-002',
    'chunk_size': 500,
    'chunk_overlap': 50,
    'similarity_threshold': 0.7
}
```

---

## 📊 Performance Metrics

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Message Storage | O(1) | O(n) |
| Document Search | O(n) | O(n) |
| Vector Search (RAG) | O(n log n) | O(n × d) |
| Context Window | O(n) | O(k) |
| Model Routing | O(1) | O(p) |
| Speech Recognition | O(1) | O(1) |
| LRU Cache | O(1) | O(c) |

**Legend**: n = items, d = dimensions, k = window size, p = providers, c = cache capacity

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 👥 Team Members

- **Jahanzeb** - 23BS(IT)07 - Backend Developer
- **Sheshant** - 23BS(IT)05 - Frontend Developer

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Shaheed Benazir Bhutto University, Sanghar Campus
- Anthropic for Claude API
- OpenAI for GPT API
- LangChain community
- All open-source contributors

---

## 📞 Contact

**Email**: support@sbbu.edu.pk  
**Website**: https://sbbu.edu.pk  
**GitHub**: https://github.com/sbbu/chatbot

---

## 🔮 Future Enhancements

- [ ] Multi-language support (Urdu, Sindhi)
- [ ] Mobile application (React Native)
- [ ] Advanced analytics dashboard
- [ ] Fine-tuned models for SBBU-specific queries
- [ ] Integration with university LMS
- [ ] Real-time collaboration features
- [ ] Enhanced RAG with graph databases
- [ ] Improved caching strategies

---

**Made with ❤️ for SBBU Students and Faculty**
