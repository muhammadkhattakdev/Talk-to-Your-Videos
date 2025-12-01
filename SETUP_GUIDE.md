# BotTube Backend Setup Guide

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- A Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## 🚀 Installation Steps

### 1. Create Virtual Environment

```bash
# Navigate to your project directory
cd server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
# You can use any text editor:
nano .env
# or
code .env
```

Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Configure Django Settings

Add the configurations from `settings_config.py` to your `server/settings.py` file.

Key sections to add:
- INSTALLED_APPS (add 'rest_framework', 'corsheaders', 'api')
- MIDDLEWARE (add 'corsheaders.middleware.CorsMiddleware')
- CORS_ALLOWED_ORIGINS
- REST_FRAMEWORK
- CACHES

### 5. Update Main URLs

Edit `server/urls.py`:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # Add this line
]
```

### 6. Run Migrations

```bash
# Create database tables
python manage.py makemigrations
python manage.py migrate
```

### 7. Create Superuser (Optional)

```bash
python manage.py createsuperuser
# Follow the prompts to create an admin user
```

### 8. Run the Development Server

```bash
python manage.py runserver
```

The API will be available at: `http://localhost:8000/`

## 🧪 Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/api/health/

# Chat with video
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ",
    "question": "What is this video about?"
  }'
```

### Using Postman or Thunder Client

1. **Health Check**
   - Method: GET
   - URL: `http://localhost:8000/api/health/`

2. **Chat Request**
   - Method: POST
   - URL: `http://localhost:8000/api/chat/`
   - Headers: `Content-Type: application/json`
   - Body:
     ```json
     {
       "video_id": "dQw4w9WgXcQ",
       "question": "What is the main topic of this video?"
     }
     ```

## 📁 Project Structure

```
server/
├── server/
│   ├── __init__.py
│   ├── settings.py         # Django settings
│   ├── urls.py             # Main URL config
│   └── wsgi.py
├── api/
│   ├── __init__.py
│   ├── views.py            # API endpoints
│   ├── serializers.py      # Data validation
│   ├── urls.py             # API URLs
│   ├── services/
│   │   ├── __init__.py
│   │   └── rag_service.py  # RAG implementation
│   ├── migrations/
│   └── tests.py
├── .env                     # Environment variables (create this)
├── .env.example            # Example env file
├── requirements.txt        # Python dependencies
└── manage.py               # Django management script
```

## 🔧 Configuration Options

### Cache Settings

**Development (Default):**
Uses Django's built-in memory cache. Works out of the box.

**Production:**
Use Redis for better performance across multiple processes:

```bash
# Install Redis
# On macOS:
brew install redis
# On Ubuntu:
sudo apt-get install redis-server

# Start Redis
redis-server
```

Update settings.py to use Redis (see settings_config.py).

### CORS Configuration

The default setup allows requests from:
- `http://localhost:3000` (React default)
- `http://localhost:5173` (Vite default)

To add more origins, update `CORS_ALLOWED_ORIGINS` in settings.py or .env file.

## 🐛 Troubleshooting

### Issue: "No module named 'rest_framework'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "CORS policy blocked"
**Solution:** Verify CORS settings in settings.py include your frontend URL.

### Issue: "Failed to fetch transcript"
**Solution:** 
- Check if the video has captions/subtitles
- Verify internet connection
- Some videos may have restricted transcripts

### Issue: "Invalid API key"
**Solution:**
- Verify your GEMINI_API_KEY in .env file
- Make sure .env is in the server directory
- Restart the Django server after updating .env

### Issue: "ModuleNotFoundError: No module named 'dotenv'"
**Solution:**
```bash
pip install python-dotenv
```

## 📊 Performance Tips

1. **Enable Redis Caching**
   - Much faster than memory cache
   - Persists between server restarts
   - Shared across multiple workers

2. **Use Production Server**
   ```bash
   pip install gunicorn
   gunicorn server.wsgi:application
   ```

3. **Optimize Chunk Size**
   - Smaller chunks = more precise but slower
   - Larger chunks = faster but less precise
   - Default (1000) is a good balance

4. **Cache Embeddings**
   - Consider caching embeddings along with transcripts
   - Modify RAG service to store embeddings

## 🔐 Security Notes

1. **Never commit .env file to Git**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables for sensitive data**
   - API keys
   - Secret keys
   - Database credentials

3. **Set DEBUG=False in production**

4. **Use HTTPS in production**

5. **Implement rate limiting** (consider Django Rate Limit package)

## 📚 Next Steps

1. Read `RAG_EXPLAINED.md` for detailed code explanations
2. Test the API with your React frontend
3. Experiment with different chunk sizes and models
4. Add user authentication if needed
5. Deploy to production (Heroku, AWS, Google Cloud, etc.)

## 🆘 Getting Help

If you encounter issues:
1. Check Django logs: Look at console output
2. Check the API response: Use browser DevTools or Postman
3. Verify environment variables: `python -c "import os; print(os.getenv('GEMINI_API_KEY'))"`
4. Check Django documentation: https://docs.djangoproject.com/

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health/` | GET | Check if API is running |
| `/api/chat/` | POST | Send question about video |

## 🎉 Success!

If everything is set up correctly:
- Health check returns: `{"status": "ok", "message": "API is running"}`
- Chat endpoint returns meaningful answers about videos
- No CORS errors in frontend console

Happy coding! 🚀
