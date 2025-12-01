# BotTube Backend - Quick Start Guide

## 🎯 What You Have

A complete Django backend with RAG (Retrieval-Augmented Generation) functionality for chatting with YouTube videos.

## 📦 Files Included

1. **api/** - Complete Django app
   - `views.py` - API endpoints
   - `serializers.py` - Data validation
   - `urls.py` - URL routing
   - `services/rag_service.py` - RAG implementation

2. **RAG_EXPLAINED.md** - Comprehensive educational document
   - Explains every line of code
   - Covers all Python concepts
   - Step-by-step RAG pipeline
   - Perfect for learning

3. **SETUP_GUIDE.md** - Installation and configuration
   - Step-by-step setup
   - Troubleshooting tips
   - Testing instructions

4. **requirements.txt** - Python dependencies

5. **.env.example** - Environment variables template

6. **settings_config.py** - Django settings configuration

## ⚡ Quick Setup (5 Minutes)

```bash
# 1. Copy files to your Django project
cp -r api/ /path/to/your/server/
cp requirements.txt /path/to/your/server/
cp .env.example /path/to/your/server/.env

# 2. Install dependencies
cd /path/to/your/server
pip install -r requirements.txt

# 3. Edit .env and add your Gemini API key
# Get key from: https://makersuite.google.com/app/apikey
nano .env

# 4. Update server/settings.py
# Add configurations from settings_config.py

# 5. Update server/urls.py
# Add: path('api/', include('api.urls')),

# 6. Run migrations
python manage.py migrate

# 7. Start server
python manage.py runserver
```

## 🧪 Test It!

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

## 📚 Learning Path

1. **Start Here:** Read `SETUP_GUIDE.md` to get everything running

2. **Understand RAG:** Read `RAG_EXPLAINED.md` - it explains:
   - What RAG is and why we use it
   - Every Python concept (Dict, List, typing, etc.)
   - Step-by-step code walkthrough
   - How everything connects

3. **Customize:** Experiment with the code:
   - Change chunk sizes
   - Try different models
   - Add new features

## 🎓 Key Concepts Explained

### What is RAG?
Think of it as giving the AI a "cheat sheet" - we find relevant parts of the video transcript and give only those to the AI, making answers more accurate and efficient.

### The Flow:
```
User Question → Get Transcript → Split into Chunks → 
Create Embeddings → Find Similar Chunks → 
Generate Answer with AI → Return to User
```

### Key Technologies:
- **Django REST Framework**: API endpoints
- **YouTube Transcript API**: Get video transcripts
- **Google Gemini**: AI for embeddings and answers
- **NumPy**: Math operations for similarity

## 🔥 Cool Features

✅ Smart chunking with overlap (maintains context)
✅ Caching (faster repeated queries)
✅ Cosine similarity (finds most relevant content)
✅ Comprehensive error handling
✅ Type hints (better code quality)
✅ Detailed logging (easy debugging)

## 🤔 Common Questions

**Q: Do I need a paid API key?**
A: Gemini has a free tier that's generous for development.

**Q: Can I use this with other LLMs?**
A: Yes! The code is modular. Just change the model in `rag_service.py`.

**Q: How accurate is it?**
A: Very! RAG combines retrieval precision with AI generation power.

**Q: Will it work with any YouTube video?**
A: It works with videos that have captions/transcripts available.

## 🎯 Next Steps

1. ✅ Get it running (follow SETUP_GUIDE.md)
2. 📖 Learn how it works (read RAG_EXPLAINED.md)
3. 🔧 Connect your React frontend
4. 🚀 Deploy to production
5. 🎨 Add your own features!

## 💡 Pro Tips

- Read the comments in the code - they're very detailed
- Start with RAG_EXPLAINED.md if you're new to RAG
- The code is production-ready but start with development settings
- Use Redis cache in production for better performance
- Consider adding rate limiting for public APIs

## 🆘 Need Help?

1. Check SETUP_GUIDE.md troubleshooting section
2. Read the inline comments in the code
3. Review RAG_EXPLAINED.md for concept clarification
4. Check Django/DRF documentation
5. Make sure your API key is valid and in .env

## 📊 Project Statistics

- **Lines of Code**: ~500
- **Setup Time**: ~5 minutes
- **Learning Time**: 1-2 hours (if reading all docs)
- **Technologies**: 6+ (Django, DRF, Gemini, NumPy, etc.)
- **Documentation**: Extensive (comments everywhere!)

## 🎉 You're Ready!

Everything is set up and documented. Start with the setup guide, understand the concepts through the educational document, and you'll have a working AI chatbot for YouTube videos!

Happy coding! 🚀
