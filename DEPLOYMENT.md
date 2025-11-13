# Deployment Guide: Transformer Summarizer

This guide walks you through testing locally and deploying to Render (backend) and Vercel (frontend).

---

## Prerequisites

- Python 3.11+ installed
- Node.js 18+ and npm installed
- Git repository (GitHub recommended)
- Trained model artifacts (`artifacts/model.pt` and `artifacts/tokenizer.json`)
- Render account (free tier available)
- Vercel account (free tier available)

---

## Part 1: Local Testing

### Step 1: Train the Model (if not already done)

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # Linux/Mac
   ```

2. **Install training dependencies**:
   ```bash
   pip install -r ml_model/requirements.txt
   ```

3. **Install ml_model package**:
   ```bash
   pip install -e .
   ```

4. **Train the model**:
   ```bash
   python -m ml_model.train --epochs 30 --train-tokenizer --dataset mtsamples.csv
   ```

   This will create:
   - `artifacts/tokenizer.json`
   - `artifacts/model.pt`

### Step 2: Test Backend Locally

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment** (or reuse the main one):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install ml_model package** (from project root):
   ```bash
   cd ..
   pip install -e .
   cd backend
   ```

5. **Set environment variables** (optional, defaults work if artifacts are in `artifacts/`):
   ```bash
   # Windows CMD
   set SUMMARIZER_MODEL_PATH=..\artifacts\model.pt
   set SUMMARIZER_TOKENIZER_PATH=..\artifacts\tokenizer.json

   # Windows PowerShell
   $env:SUMMARIZER_MODEL_PATH="..\artifacts\model.pt"
   $env:SUMMARIZER_TOKENIZER_PATH="..\artifacts\tokenizer.json"

   # Linux/Mac
   export SUMMARIZER_MODEL_PATH=../artifacts/model.pt
   export SUMMARIZER_TOKENIZER_PATH=../artifacts/tokenizer.json
   ```

6. **Start the server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

7. **Test the API**:
   - Open browser: `http://localhost:8000/docs` (FastAPI Swagger UI)
   - Or test with curl:
     ```bash
     curl -X POST "http://localhost:8000/summarize" \
          -H "Content-Type: application/json" \
          -d "{\"text\": \"Your medical transcript text here...\"}"
     ```
   - Check health: `http://localhost:8000/healthz`

### Step 3: Test Frontend Locally

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Create `.env.local` file**:
   ```env
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

5. **Open browser**: `http://localhost:3000`

6. **Test the UI**:
   - Paste a medical transcript in the text area
   - Click "Generate Summary"
   - Verify the summary appears

---

## Part 2: Deploy Backend to Render

### Step 1: Prepare Repository

1. **Ensure your code is pushed to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Verify these files exist in your repo**:
   - `backend/requirements.txt`
   - `backend/app/main.py`
   - `ml_model/` directory (all Python files)
   - `setup.py` or `pyproject.toml` (for installing ml_model package)

### Step 2: Upload Model Artifacts

You have two options:

**Option A: Use Render Persistent Disk (Recommended for free tier)**
- We'll upload artifacts after deployment via Render's file system

**Option B: Use Cloud Storage (S3, GCS, etc.)**
- Upload `artifacts/model.pt` and `artifacts/tokenizer.json` to cloud storage
- Download them in a startup script

For now, we'll use Option A. After deployment, you can upload files via Render's dashboard or use a startup script.

### Step 3: Create Render Web Service

1. **Go to Render Dashboard**: https://dashboard.render.com

2. **Click "New +" → "Web Service"**

3. **Connect your GitHub repository**

4. **Configure the service**:
   - **Name**: `transformer-summarizer-api` (or your choice)
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or `backend` if you want to deploy only backend)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r backend/requirements.txt && pip install -e .
     ```
   - **Start Command**:
     ```bash
     cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Environment Variables**:
     ```
     SUMMARIZER_MODEL_PATH=/opt/render/project/src/artifacts/model.pt
     SUMMARIZER_TOKENIZER_PATH=/opt/render/project/src/artifacts/tokenizer.json
     SUMMARIZER_MAX_INPUT_LENGTH=512
     SUMMARIZER_MAX_OUTPUT_LENGTH=128
     ```

5. **Click "Create Web Service"**

### Step 4: Upload Model Artifacts to Render

1. **After the service is created**, note your service URL (e.g., `https://transformer-summarizer-api.onrender.com`)

2. **Option 1: Use Render Shell** (Recommended)
   - Go to your service → "Shell" tab
   - Run:
     ```bash
     mkdir -p /opt/render/project/src/artifacts
     ```
   - Upload files via Render's file browser or use `scp`/SFTP if available

3. **Option 2: Use a Startup Script**
   - Create `backend/startup.sh`:
     ```bash
     #!/bin/bash
     mkdir -p /opt/render/project/src/artifacts
     # Download from cloud storage if using Option B
     # Or copy from a mounted volume
     ```
   - Make it executable and reference it in build/start commands

4. **Option 3: Include in Git** (Not recommended for large files)
   - If artifacts are small enough, commit them to Git
   - Render will clone them automatically

### Step 5: Verify Backend Deployment

1. **Wait for deployment to complete** (first deploy takes 5-10 minutes)

2. **Test health endpoint**:
   ```bash
   curl https://your-service-name.onrender.com/healthz
   ```
   Should return: `{"status":"ok"}`

3. **Test summarize endpoint**:
   ```bash
   curl -X POST "https://your-service-name.onrender.com/summarize" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"Test medical transcript...\"}"
   ```

---

## Part 3: Deploy Frontend to Vercel

### Step 1: Prepare Frontend

1. **Ensure frontend code is in your Git repository**

2. **Verify `frontend/package.json` exists with build scripts**

### Step 2: Deploy to Vercel

1. **Go to Vercel Dashboard**: https://vercel.com

2. **Click "Add New..." → "Project"**

3. **Import your GitHub repository**

4. **Configure the project**:
   - **Framework Preset**: Next.js (auto-detected)
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `.next` (auto-detected)
   - **Install Command**: `npm install` (auto-detected)

5. **Add Environment Variable**:
   - **Key**: `NEXT_PUBLIC_API_BASE_URL`
   - **Value**: `https://your-render-service.onrender.com`
   - (Replace with your actual Render backend URL)

6. **Click "Deploy"**

### Step 3: Verify Frontend Deployment

1. **Wait for deployment** (usually 2-3 minutes)

2. **Visit your Vercel URL** (e.g., `https://your-project.vercel.app`)

3. **Test the UI**:
   - Enter a medical transcript
   - Click "Generate Summary"
   - Verify it calls your Render backend and displays the summary

---

## Troubleshooting

### Backend Issues

**Problem**: Model not found
- **Solution**: Verify `SUMMARIZER_MODEL_PATH` and `SUMMARIZER_TOKENIZER_PATH` point to correct locations
- Check Render logs: Service → "Logs" tab

**Problem**: Import errors (`ml_model` not found)
- **Solution**: Ensure `pip install -e .` runs in build command
- Verify `setup.py` exists in project root

**Problem**: Out of memory
- **Solution**: Render free tier has 512MB RAM. Consider:
  - Using CPU instead of GPU (model loads on CPU by default)
  - Reducing model size or batch size
  - Upgrading to paid tier

### Frontend Issues

**Problem**: CORS errors
- **Solution**: Add CORS middleware to FastAPI backend (see `backend/app/main.py`)

**Problem**: API calls fail
- **Solution**: 
  - Verify `NEXT_PUBLIC_API_BASE_URL` is set correctly in Vercel
  - Check Render backend is running and accessible
  - Check browser console for errors

**Problem**: Build fails
- **Solution**: 
  - Check Vercel build logs
  - Ensure all dependencies are in `package.json`
  - Verify TypeScript compilation passes locally

---

## Next Steps

- Add CORS middleware to backend for production
- Set up monitoring and logging
- Add rate limiting for API protection
- Implement authentication if needed
- Set up CI/CD pipelines

---

## Quick Reference

### Local Development URLs
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- Backend Docs: `http://localhost:8000/docs`

### Environment Variables

**Backend (Render)**:
```
SUMMARIZER_MODEL_PATH=/opt/render/project/src/artifacts/model.pt
SUMMARIZER_TOKENIZER_PATH=/opt/render/project/src/artifacts/tokenizer.json
```

**Frontend (Vercel)**:
```
NEXT_PUBLIC_API_BASE_URL=https://your-render-service.onrender.com
```

