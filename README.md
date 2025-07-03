# Squat Counter AI Server

This is a Flask-based server for squat detection in videos.

## Features
- Upload a video for squat detection
- Asynchronous processing
- Get results and download processed videos

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/MasterAffan/squatserver.git
   cd <squatserver>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   python app.py
   ```

## Deployment on Railway
1. Push this repo to GitHub.
2. On Railway, create a new project from your GitHub repo.
3. Railway will auto-detect the Python app and use the `Procfile`.
4. No extra config needed; the app will be available at your Railway domain.

## Environment Variables
- `PORT`: Set automatically by Railway. The app will use this if available.

## API Endpoints
- `GET /` - Welcome/info
- `GET /ping` - Health check
- `POST /upload` - Upload a video
- `GET /result/<job_id>` - Get processing result
- `GET /processed/<filename>` - Download processed video 