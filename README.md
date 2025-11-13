# Transformer Summarizer Project

A full-stack project that trains a custom transformer from scratch to summarize medical transcripts and exposes the model through a FastAPI backend (for Render) and a Next.js frontend (for Vercel).

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Quick 5-minute guide to get started
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Detailed step-by-step deployment instructions for Render and Vercel

## Project Structure

- `ml_model/` – reusable Python package with training and inference utilities.
- `backend/` – FastAPI service that wraps the trained model behind a `/summarize` endpoint.
- `frontend/` – Next.js application that calls the backend and provides a polished UI.
- `mtsamples.csv` – source dataset (MTSamples).

## Training Pipeline

1. **Create a virtual environment** and install deps:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r ml_model/requirements.txt
   ```

2. **Train the tokenizer & model** (artifacts saved under `artifacts/` by default):

   ```bash
   python -m ml_model.train --epochs 30 --train-tokenizer
   ```

   Adjust CLI flags (`--batch-size`, `--learning-rate`, etc.) as needed. Model checkpoints and tokenizer are saved to `artifacts/model.pt` and `artifacts/tokenizer.json`.

3. **(Optional)** Track training in TensorBoard:

   ```bash
   tensorboard --logdir runs
   ```

## Backend Service (Render)

1. **Local development**

   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8000
   ```

   Environment variables:

   - `SUMMARIZER_MODEL_PATH` – path to `model.pt` (default `artifacts/model.pt`)
   - `SUMMARIZER_TOKENIZER_PATH` – path to `tokenizer.json`
   - `SUMMARIZER_MAX_INPUT_LENGTH`, `SUMMARIZER_MAX_OUTPUT_LENGTH` – optional overrides.

2. **Deploy to Render**

   - Create a new *Web Service* from your Git repo.
   - Runtime: Python 3.11 (or newer).
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables pointing to your model/tokenizer. You can upload artifacts to Render persistent disk or store in an object store (S3, GCS) and download in a startup script.

## Frontend App (Vercel)

1. **Local development**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Configure a `.env.local`:

   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   ```

2. **Deploy to Vercel**

   - Push the repo to GitHub and import into Vercel.
   - Set `NEXT_PUBLIC_API_BASE_URL` to the public Render URL (e.g. `https://your-render-app.onrender.com`).
   - Trigger the build; Vercel runs `npm install` then `npm run build`.

## End-to-End Flow

1. Train the model locally using `ml_model.train`.
2. Upload `artifacts/model.pt` and `artifacts/tokenizer.json` to your Render backend (either via persistent disk or remote storage download step).
3. Deploy the backend; verify `GET /healthz`.
4. Deploy the frontend with `NEXT_PUBLIC_API_BASE_URL` pointed at the Render backend URL.
5. Visit the Vercel URL, paste medical text, and generate summaries.

## Testing Locally

Run the test script to verify your setup:

```bash
python test_local.py
```

This checks:
- All required packages are installed
- Model artifacts exist
- ml_model package can be imported
- Backend can be imported

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for complete instructions on:
- Testing locally (backend + frontend)
- Deploying backend to Render
- Deploying frontend to Vercel
- Troubleshooting common issues

Or use **[QUICK_START.md](QUICK_START.md)** for a condensed guide.

## Next Steps

- Add authentication & rate limiting for public usage.
- Implement streaming responses or progress indicators.
- Replace notebook prototyping with unit/integration tests (PyTest, Playwright).


