# Quick Start Guide

## 🚀 Local Testing (5 minutes)

### 1. Train Model (if needed)
```bash
# Create venv
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install and train
pip install -r ml_model/requirements.txt
pip install -e .
python -m ml_model.train --epochs 30 --train-tokenizer --dataset mtsamples.csv
```

### 2. Start Backend
```bash
cd backend
pip install -r requirements.txt
cd .. && pip install -e . && cd backend
uvicorn app.main:app --reload --port 8000
```

**Test**: Open http://localhost:8000/docs

### 3. Start Frontend
```bash
cd frontend
npm install
# Create .env.local with: NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

**Test**: Open http://localhost:3000

---

## ☁️ Deploy to Render (Backend)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Create Render Service**
   - Go to https://dashboard.render.com
   - New → Web Service
   - Connect GitHub repo
   - Settings:
     - **Build**: `pip install -r backend/requirements.txt && pip install -e .`
     - **Start**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     - **Environment**:
       - `SUMMARIZER_MODEL_PATH=/opt/render/project/src/artifacts/model.pt`
       - `SUMMARIZER_TOKENIZER_PATH=/opt/render/project/src/artifacts/tokenizer.json`

3. **Upload Artifacts**
   - Use Render Shell or file browser
   - Upload `artifacts/model.pt` and `artifacts/tokenizer.json` to `/opt/render/project/src/artifacts/`

4. **Test**: `curl https://your-service.onrender.com/healthz`

---

## 🌐 Deploy to Vercel (Frontend)

1. **Go to Vercel**: https://vercel.com
2. **Import GitHub repo**
3. **Settings**:
   - Root Directory: `frontend`
   - Environment Variable: `NEXT_PUBLIC_API_BASE_URL=https://your-render-service.onrender.com`
4. **Deploy**

**Test**: Visit your Vercel URL and try summarizing text!

---

## 📝 Notes

- First Render deploy takes 5-10 minutes
- Free tier has 512MB RAM (may need CPU-only model)
- See `DEPLOYMENT.md` for detailed troubleshooting

