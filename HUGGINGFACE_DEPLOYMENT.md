# Hugging Face Spaces Deployment Guide

## Quick Start

1. **Create a Hugging Face Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Set visibility (Public/Private)

2. **Upload Files**
   - Upload these files to your Space:
     - `app.py` (main application)
     - `Dockerfile` (container configuration)
     - `requirements_app.txt` (Python dependencies)
     - `README_HF.md` (Space description)
     - `src/` directory (your source code)

3. **Configure Space Settings**
   - Set the title: "AI News Hub"
   - Set the emoji: 📰
   - Set the color: blue to purple
   - Set the license: MIT

4. **Deploy**
   - The Space will automatically build and deploy
   - Monitor the build logs for any issues
   - Once deployed, your API will be available at `https://your-username-ai-news-hub.hf.space`

## API Usage

Once deployed, you can use the API:

```bash
# Health check
curl https://your-username-ai-news-hub.hf.space/health

# Ingest news
curl -X POST "https://your-username-ai-news-hub.hf.space/news/ingest?url=https://example.com/news"

# Get personalized news
curl "https://your-username-ai-news-hub.hf.space/news/personalize?user_id=user123&limit=5"

# Summarize article
curl -X POST "https://your-username-ai-news-hub.hf.space/news/summarize?article_id=article123"
```

## Local Development

To test locally:

```bash
# Build the Docker image
docker build -t ai-news-hub .

# Run locally
docker run -p 7860:7860 ai-news-hub

# Or run directly with Python
pip install -r requirements_app.txt
python app.py
```

## Features Included

- ✅ News ingestion from URLs
- ✅ Content ranking and personalization
- ✅ Article summarization
- ✅ Health monitoring
- ✅ CORS enabled for web integration
- ✅ Structured logging
- ✅ Error handling

## Customization

To integrate your full microservices:

1. **Import your services** in `app.py`
2. **Add service endpoints** for each microservice
3. **Update requirements** if needed
4. **Configure environment variables** for production

## Troubleshooting

- **Build fails**: Check Dockerfile and requirements
- **Port issues**: Ensure port 7860 is exposed
- **Import errors**: Verify all dependencies are in requirements_app.txt
- **Memory issues**: Consider upgrading Space hardware

## Next Steps

1. Deploy to Hugging Face Spaces
2. Test all API endpoints
3. Integrate with your frontend
4. Add authentication if needed
5. Scale with additional services
