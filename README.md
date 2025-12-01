# BotTube Backend - Complete RAG Implementation

A production-ready Django backend that enables AI-powered conversations with YouTube videos using Retrieval-Augmented Generation (RAG).

## 🌟 Features

- ✅ **RAG Implementation**: Intelligent question-answering using vector embeddings
- ✅ **YouTube Integration**: Automatic transcript fetching
- ✅ **Smart Caching**: Fast responses for repeated queries
- ✅ **Production Ready**: Comprehensive error handling and logging
- ✅ **Educational**: Extensively documented code with explanations
- ✅ **Type Safe**: Full type hints for better code quality
- ✅ **REST API**: Clean, well-structured API endpoints

## 📚 Documentation

This project includes comprehensive documentation:

1. **[QUICK_START.md](./QUICK_START.md)** - Get running in 5 minutes
2. **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** - Detailed installation and configuration
3. **[RAG_EXPLAINED.md](./RAG_EXPLAINED.md)** - Learn RAG from scratch (50+ pages)
4. **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Visual system architecture diagrams

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Configure Django
# Add configurations from settings_config.py to server/settings.py

# 4. Run server
python manage.py runserver

# 5. Test
curl http://localhost:8000/api/health/
```

## 📁 What's Included

```
api/
├── views.py              # API endpoints (chat, health check)
├── serializers.py        # Request/response validation
├── urls.py               # URL routing configuration
└── services/
    └── rag_service.py    # Core RAG implementation

docs/
├── QUICK_START.md        # 5-minute setup guide
├── SETUP_GUIDE.md        # Complete installation guide
├── RAG_EXPLAINED.md      # Educational deep-dive (50+ pages)
└── ARCHITECTURE.md       # Visual architecture diagrams

config/
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── settings_config.py    # Django settings snippet
```

## 🎯 API Endpoints

### Health Check
```bash
GET /api/health/

Response: 200 OK
{
  "status": "ok",
  "message": "API is running"
}
```

### Chat with Video
```bash
POST /api/chat/
Content-Type: application/json

{
  "video_id": "dQw4w9WgXcQ",
  "question": "What is this video about?"
}

Response: 200 OK
{
  "answer": "This video is about...",
  "video_id": "dQw4w9WgXcQ"
}
```

## 🧠 How RAG Works

```
1. User asks question about a video
   ↓
2. System fetches video transcript (cached if available)
   ↓
3. Transcript is split into manageable chunks
   ↓
4. Chunks and question are converted to embeddings (vectors)
   ↓
5. Most relevant chunks are found using cosine similarity
   ↓
6. Relevant chunks are sent to AI with the question
   ↓
7. AI generates accurate answer based on video content
   ↓
8. Answer is returned to user
```

**Why RAG?**
- More accurate than general AI (uses actual video content)
- More efficient than sending entire transcript
- Scalable to any video length

## 🛠️ Technologies

- **Django 5.0** - Web framework
- **Django REST Framework 3.14** - API framework
- **Google Gemini** - LLM for embeddings and generation
- **YouTube Transcript API** - Fetch video transcripts
- **NumPy** - Mathematical operations for similarity
- **Python 3.10+** - Programming language

## 📖 Learning Resources

### For Beginners
Start with **RAG_EXPLAINED.md** which covers:
- What is RAG and why we need it
- Every Python concept used (Dict, List, typing, etc.)
- Step-by-step code walkthrough
- How everything connects
- Common questions answered

### Key Concepts Explained
- **Embeddings**: Converting text to numbers that capture meaning
- **Chunking**: Breaking large texts into manageable pieces
- **Cosine Similarity**: Measuring how similar two pieces of text are
- **Type Hints**: Python's way of documenting expected types
- **Caching**: Storing data for faster repeated access

## 🔧 Configuration

### Environment Variables (.env)
```bash
GEMINI_API_KEY=your_api_key_here
SECRET_KEY=your_django_secret_key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

### Django Settings
Add to `server/settings.py`:
- REST Framework configuration
- CORS headers
- Cache settings
- Logging configuration

See `settings_config.py` for complete configuration.

## 🎨 Customization

### Change Chunk Size
```python
# In api/services/rag_service.py
self.chunk_size = 1000  # Change this value
self.chunk_overlap = 200  # And this
```

### Use Different Model
```python
# In api/services/rag_service.py
self.llm_model = genai.GenerativeModel('gemini-pro')  # Change model
```

### Modify Number of Retrieved Chunks
```python
# In api/services/rag_service.py
self.top_k_chunks = 3  # Retrieve more or fewer chunks
```

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8000/api/health/

# Test chat endpoint
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ",
    "question": "What is this video about?"
  }'
```

## 📊 Performance

- **First query**: ~2-3 seconds (fetches transcript)
- **Subsequent queries**: ~0.5-1 second (uses cache)
- **Cache duration**: 1 hour (configurable)
- **Supports**: Videos of any length with available transcripts

## 🐛 Troubleshooting

### "Failed to fetch transcript"
- Video may not have captions/subtitles
- Check if video is available in your region
- Some videos restrict transcript access

### "Invalid API key"
- Verify GEMINI_API_KEY in .env file
- Restart Django server after updating .env
- Get a new key from https://makersuite.google.com/app/apikey

### "CORS policy blocked"
- Add your frontend URL to CORS_ALLOWED_ORIGINS
- Restart Django server

See **SETUP_GUIDE.md** for more troubleshooting tips.

## 🚀 Production Deployment

1. **Set DEBUG=False** in settings
2. **Use Redis** for caching (instead of in-memory)
3. **Use PostgreSQL** for database (instead of SQLite)
4. **Add rate limiting** to prevent abuse
5. **Use Gunicorn** as WSGI server
6. **Set up HTTPS** with SSL certificate
7. **Configure proper logging**

## 📈 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement conversation history
- [ ] Add user authentication
- [ ] Support for video timestamps in answers
- [ ] Cache embeddings (not just transcripts)
- [ ] Add vector database (Pinecone, Weaviate)
- [ ] Support for private videos (with auth)
- [ ] Add streaming responses

## 🤝 Contributing

This is an educational project. Feel free to:
- Modify for your needs
- Add new features
- Optimize performance
- Improve documentation

## 📝 License

This project is provided as-is for educational purposes.

## 🆘 Support

1. Check the documentation files
2. Review inline code comments (very detailed)
3. Read RAG_EXPLAINED.md for concept clarification
4. Check Django/DRF official documentation

## 🎓 What You'll Learn

By studying this project, you'll understand:
- How RAG systems work
- Django REST API development
- Working with LLMs and embeddings
- Vector similarity search
- Caching strategies
- Error handling patterns
- Type hints and documentation
- Production-ready code practices

## 💡 Key Takeaways

1. **RAG is powerful**: Combines retrieval precision with AI generation
2. **Documentation matters**: Understand before you code
3. **Type hints help**: Makes code maintainable
4. **Caching is important**: Dramatically improves performance
5. **Error handling is crucial**: Provides good user experience

## 🎉 Success Metrics

If everything is set up correctly:
- ✅ Health endpoint returns OK
- ✅ Chat endpoint returns relevant answers
- ✅ No CORS errors in frontend
- ✅ Responses are fast (< 3s first time, < 1s after)
- ✅ Errors are handled gracefully

## 📞 API Response Examples

### Successful Response
```json
{
  "answer": "This video discusses the history of music in the 1980s, focusing on the evolution of pop culture and the impact of MTV on the music industry.",
  "video_id": "dQw4w9WgXcQ"
}
```

### Error Response
```json
{
  "error": "Failed to process your question. Please ensure the video has captions available.",
  "detail": "TranscriptsDisabled: Subtitles are disabled for this video"
}
```

## 🎯 Best Practices Implemented

- ✅ Separation of concerns (views, serializers, services)
- ✅ Type hints throughout
- ✅ Comprehensive error handling
- ✅ Detailed logging
- ✅ Caching for performance
- ✅ Input validation
- ✅ Clean code structure
- ✅ Extensive documentation