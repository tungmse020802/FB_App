# FB.ApiCore Beauty API

FastAPI wrapper with local beauty processor logic (no runtime dependency on GFPGAN package path).

## Setup
```bash
cd /workspaces/codespaces-blank/FB.ApiCore
pip install -r requirements.txt
```

## Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints
- `GET /health` – simple health check
- `POST /beautify` – multipart upload field `file`; returns JPEG of beautified image

## Example
```bash
curl -X POST \
  -F "file=@/path/to/input.jpg" \
  http://localhost:8000/beautify \
  --output beautified.jpg
```

Notes:
- Requires system `libgl1` for OpenCV (already installed in this dev container).
- Returns 422 if no face is detected; 400 if image cannot be decoded.
