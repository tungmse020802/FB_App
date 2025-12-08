# Beauty API - Face Enhancement Platform

FastAPI + Next.js full-stack beauty filter platform with MediaPipe face detection and ML-based acne healing.

## Structure

- **FB.ApiCore**: FastAPI backend with 2 processing modes
- **FB.FE**: Next.js 16 frontend (TypeScript, Tailwind CSS)
- **GFPGAN**: Library reference (not modified)

## Features

### 🌸 Beauty Mode
- MediaPipe face mesh detection
- Skin smoothing (edge-preserving filter)
- Color grading (LAB space adjustment)
- Grain texture for natural look

### 🔬 Acne Healing Mode
- K-Means clustering for acne detection
- Frequency separation technique
- Face-only processing (background untouched)
- Natural skin texture preservation

## Quick Start

### Backend (port 8000)
```bash
cd FB.ApiCore
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend (port 3001)
```bash
cd FB.FE
npm install
NEXT_PUBLIC_API_BASE="http://localhost:8000" npm run dev -- --port 3001
```

## API Endpoints

- `GET /health` - Health check
- `POST /beautify?mode=beauty` - Beauty filter (default)
- `POST /beautify?mode=acne` - ML-based acne healing

## Tech Stack

**Backend:**
- FastAPI
- OpenCV
- MediaPipe
- scikit-learn (K-Means)
- NumPy

**Frontend:**
- Next.js 16 (App Router)
- TypeScript
- Tailwind CSS v4

## License

Educational/demo project. GFPGAN is under Apache 2.0 license.
