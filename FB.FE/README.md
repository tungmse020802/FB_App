## FB.FE – Next.js client for FastAPI beautify service

Uploads an image to `http://localhost:8000/beautify` (FastAPI in `FB.ApiCore`) and shows the returned JPEG. Health check uses `GET /health`.

### Run
```bash
cd /workspaces/codespaces-blank/FB.FE
npm install
npm run dev -- --hostname 0.0.0.0 --port 3000
```
Visit http://localhost:3000 and ensure the FastAPI backend is running on port 8000.

### Config
- Override backend base URL with env `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).

### Usage
- Click "Chọn ảnh" to pick a JPEG/PNG.
- Press "Gửi tới API" to call the backend; the result preview will appear on the right.

