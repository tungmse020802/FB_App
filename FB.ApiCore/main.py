import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from beauty import HybridSkinRetouch, UltimateBeautyCam, decode_image

app = FastAPI(title="FB.ApiCore Beauty API", version="1.0.0")

# Allow browser calls from forwarded Codespaces URLs (e.g., *.app.github.dev)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.app\.github\.dev",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = UltimateBeautyCam()
ml_processor = HybridSkinRetouch(n_clusters=5)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/beautify")
def beautify_image(
    file: UploadFile = File(...),
    mode: str = Query("beauty", regex="^(beauty|acne)$"),
) -> StreamingResponse:
    file_bytes = file.file.read()
    img = decode_image(file_bytes)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    if mode == "acne":
        _, processed = ml_processor.run(img)
        if processed is None:
            raise HTTPException(status_code=422, detail="Processing failed")
    else:
        _, processed = processor.run(img)
        if processed is None:
            raise HTTPException(status_code=422, detail="No face detected")

    success, buffer = cv2.imencode(".jpg", processed)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=beautified.jpg"},
    )
