"""
LewdTagger — Anime Image Tagger with WD SwinV2 Tagger V2
Backend FastAPI server with ONNX Runtime inference.
"""

import asyncio
import hashlib
import io
import os
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"
IMAGE_SIZE = 448
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup in background."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model)
    yield

app = FastAPI(title="LewdTagger", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_state = {
    "session": None,
    "tags_df": None,
    "status": "not_loaded",  # not_loaded | loading | ready | error
    "error": None,
}

# In-memory store for scanned images and their results
image_store: dict[str, dict] = {}
current_folder: Optional[str] = None


# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────

class ScanRequest(BaseModel):
    folder_path: str


class TagRequest(BaseModel):
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD


class TagAllRequest(BaseModel):
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD


class ExportRequest(BaseModel):
    image_id: str
    use_hash: bool = True


class ExportAllRequest(BaseModel):
    use_hash: bool = True


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model():
    """Download and load the WD SwinV2 Tagger V2 model."""
    model_state["status"] = "loading"
    try:
        print("[LewdTagger] Downloading model from HuggingFace...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        tags_path = hf_hub_download(repo_id=MODEL_REPO, filename=TAGS_FILE)

        print("[LewdTagger] Loading ONNX model...")
        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            print("[LewdTagger] [OK] Using CUDA GPU acceleration")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(model_path, providers=providers)
        tags_df = pd.read_csv(tags_path)

        model_state["session"] = session
        model_state["tags_df"] = tags_df
        model_state["status"] = "ready"
        model_state["error"] = None

        print(f"[LewdTagger] [OK] Model loaded successfully ({len(tags_df)} tags)")
        print(f"[LewdTagger] [OK] Providers: {session.get_providers()}")

    except Exception as e:
        model_state["status"] = "error"
        model_state["error"] = str(e)
        print(f"[LewdTagger] [ERROR] Model loading failed: {e}")


# ──────────────────────────────────────────────
# Image processing & inference
# ──────────────────────────────────────────────

def compute_hash(filepath: str) -> str:
    """Compute SHA256 hash of an image file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for the model (448x448, float32, BGR)."""
    img = Image.open(image_path).convert("RGBA")

    # Create white background for transparent images
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("RGB")

    # Pad to square
    max_dim = max(img.size)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(img, ((max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2))

    # Resize to model input size
    padded = padded.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(padded, dtype=np.float32)

    # RGB -> BGR (model expects BGR)
    img_array = img_array[:, :, ::-1]

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def run_inference(image_path: str, general_threshold: float, character_threshold: float) -> dict:
    """Run the WD tagger model on an image and return categorized tags."""
    session = model_state["session"]
    tags_df = model_state["tags_df"]

    if session is None or tags_df is None:
        raise RuntimeError("Model not loaded")

    # Prepare image
    input_data = prepare_image(image_path)
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: input_data})[0][0]

    # Parse tags by category
    tag_names = tags_df["name"].tolist()
    tag_categories = tags_df["category"].tolist()

    rating_tags = {}
    character_tags = {}
    general_tags = {}

    for i, (name, category, score) in enumerate(zip(tag_names, tag_categories, outputs)):
        score = float(score)

        if category == 9:  # Rating
            rating_tags[name] = score
        elif category == 4:  # Character
            if score >= character_threshold:
                character_tags[name] = score
        elif category == 0:  # General
            if score >= general_threshold:
                general_tags[name] = score

    # Sort by score descending
    rating_tags = dict(sorted(rating_tags.items(), key=lambda x: x[1], reverse=True))
    character_tags = dict(sorted(character_tags.items(), key=lambda x: x[1], reverse=True))
    general_tags = dict(sorted(general_tags.items(), key=lambda x: x[1], reverse=True))

    return {
        "rating": rating_tags,
        "characters": character_tags,
        "general": general_tags,
    }


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Get model loading status."""
    return {
        "status": model_state["status"],
        "error": model_state["error"],
        "total_images": len(image_store),
        "tagged_images": sum(1 for v in image_store.values() if v.get("tags")),
        "current_folder": current_folder,
    }


@app.post("/api/scan")
async def scan_folder(req: ScanRequest):
    """Scan a folder for anime images."""
    global current_folder, image_store

    folder = Path(req.folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Pasta não encontrada: {req.folder_path}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Caminho não é uma pasta: {req.folder_path}")

    # Reset store
    image_store = {}
    current_folder = str(folder)

    # Scan for images recursively
    images = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_id = f.stem
            # Handle duplicate names
            counter = 1
            original_id = image_id
            while image_id in image_store:
                image_id = f"{original_id}_{counter}"
                counter += 1

            image_store[image_id] = {
                "id": image_id,
                "filename": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "hash": None,  # Computed lazily
                "tags": None,
            }
            images.append({
                "id": image_id,
                "filename": f.name,
                "size": f.stat().st_size,
            })

    if not images:
        raise HTTPException(status_code=404, detail="Nenhuma imagem encontrada na pasta")

    return {
        "folder": str(folder),
        "total": len(images),
        "images": images,
    }


@app.post("/api/tag/{image_id}")
async def tag_image(image_id: str, req: TagRequest):
    """Run tagger on a single image."""
    if model_state["status"] != "ready":
        raise HTTPException(status_code=503, detail="Modelo ainda não está carregado")

    if image_id not in image_store:
        raise HTTPException(status_code=404, detail=f"Imagem não encontrada: {image_id}")

    img_data = image_store[image_id]
    image_path = img_data["path"]

    try:
        # Compute hash if not already done
        if not img_data["hash"]:
            img_data["hash"] = compute_hash(image_path)

        # Run inference
        tags = run_inference(image_path, req.general_threshold, req.character_threshold)
        img_data["tags"] = tags

        return {
            "id": image_id,
            "filename": img_data["filename"],
            "hash": img_data["hash"],
            "tags": tags,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")


@app.post("/api/tag-all")
async def tag_all_images(req: TagAllRequest):
    """Run tagger on all scanned images. Returns results progressively."""
    if model_state["status"] != "ready":
        raise HTTPException(status_code=503, detail="Modelo ainda não está carregado")

    if not image_store:
        raise HTTPException(status_code=400, detail="Nenhuma imagem escaneada. Use /api/scan primeiro.")

    results = []
    errors = []

    for image_id, img_data in image_store.items():
        try:
            if not img_data["hash"]:
                img_data["hash"] = compute_hash(img_data["path"])

            tags = run_inference(
                img_data["path"],
                req.general_threshold,
                req.character_threshold,
            )
            img_data["tags"] = tags

            results.append({
                "id": image_id,
                "filename": img_data["filename"],
                "hash": img_data["hash"],
                "tags": tags,
            })

        except Exception as e:
            errors.append({"id": image_id, "error": str(e)})

    return {
        "total": len(image_store),
        "tagged": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors,
    }


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """Serve an image file."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail=f"Imagem não encontrada: {image_id}")

    image_path = image_store[image_id]["path"]
    return FileResponse(image_path)

# ──────────────────────────────────────────────
# Clustering and Renaming
# ──────────────────────────────────────────────

def compute_dhash(image_path: str, hash_size: int = 8) -> str:
    """Compute a difference hash (dHash) for an image."""
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.array(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return ''.join(['1' if b else '0' for b in diff.flatten()])
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return '0' * (hash_size * hash_size)

def hamming_distance(h1: str, h2: str) -> int:
    """Calculate the Hamming distance between two binary hash strings."""
    if len(h1) != len(h2):
        return 999
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

EXPOSURE_WEIGHTS = {
    "nude": 15, "nipples": 10, "pussy": 15, "sex": 20, "naked": 15,
    "uncensored": 5, "cum": 8, "cum_on_body": 10, "cum_in_pussy": 15,
    "cum_on_face": 10, "breasts_out": 8, "topless": 8, "bottomless": 8,
    "pussy_juice": 8, "areola": 5, "clothing": -5, "bikini": 2,
    "swimsuit": 2, "underwear": 2, "panties": 2, "bra": 2,
    "fully_clothed": -10,
}

def calculate_exposure_score(tags: dict) -> float:
    """Calculate an exposure score based on tags (higher = more exposed)."""
    score = 0.0
    general_tags = tags.get("general", {})
    for tag_name, confidence in general_tags.items():
        if tag_name in EXPOSURE_WEIGHTS:
            score += EXPOSURE_WEIGHTS[tag_name] * confidence
    return score


@app.post("/api/rename")
async def apply_renames():
    """Cluster and rename images by character."""
    global image_store
    
    char_groups = {}
    
    # 1. Group by primary character
    for img_id, data in image_store.items():
        tags = data.get("tags")
        if not tags or not tags.get("characters"):
            # Leave non-character images alone
            continue
            
        primary_char = max(tags["characters"].items(), key=lambda x: x[1])[0]
        # Clean character name (e.g., raven_(teen_titans) -> Raven)
        clean_char = primary_char.split("(")[0].strip().replace("_", " ").title().replace(" ", "")
        
        if clean_char not in char_groups:
            char_groups[clean_char] = []
        char_groups[clean_char].append(img_id)
        
    new_names_map = {}
    renamed_count = 0
    
    # 2. Cluster and Rename
    for char_name, img_ids in char_groups.items():
        items = []
        for img_id in img_ids:
            path = image_store[img_id]["path"]
            dhash = compute_dhash(path)
            score = calculate_exposure_score(image_store[img_id]["tags"])
            items.append({"id": img_id, "path": path, "hash": dhash, "score": score})
            
        # Cluster items (greedy)
        clusters = []
        for item in items:
            placed = False
            for cluster in clusters:
                rep_hash = cluster[0]["hash"]
                # 64 bits total. Distance <= 28 means >= 55% match
                if hamming_distance(item["hash"], rep_hash) <= 28:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])
                
        is_single_cluster = len(clusters) == 1
                
        # Sort and rename each cluster
        for cluster_idx, cluster in enumerate(clusters, 1):
            cluster.sort(key=lambda x: x["score"])
            
            for img_idx, item in enumerate(cluster, 1):
                img_id = item["id"]
                old_path = Path(item["path"])
                
                # Format: Raven_01.png if only 1 cluster, else Raven-1_01.png
                if is_single_cluster:
                    new_stem = f"{char_name}_{img_idx:02d}"
                else:
                    new_stem = f"{char_name}-{cluster_idx}_{img_idx:02d}"
                    
                new_path = old_path.parent / f"{new_stem}{old_path.suffix}"
                
                counter = 1
                while new_path.exists() and new_path != old_path:
                    if is_single_cluster:
                        new_stem = f"{char_name}_{img_idx:02d}_{counter}"
                    else:
                        new_stem = f"{char_name}-{cluster_idx}_{img_idx:02d}_{counter}"
                    new_path = old_path.parent / f"{new_stem}{old_path.suffix}"
                    counter += 1
                
                if old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                        image_store[img_id]["path"] = str(new_path)
                        image_store[img_id]["filename"] = new_path.name
                        new_names_map[img_id] = new_path.name
                        renamed_count += 1
                    except Exception as e:
                        print(f"Error renaming {old_path} to {new_path}: {e}")

    return {"success": True, "renamed_count": renamed_count, "renames": new_names_map}


@app.get("/api/thumbnail/{image_id}")
async def get_thumbnail(image_id: str):
    """Serve a thumbnail of an image (max 400px)."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail=f"Imagem não encontrada: {image_id}")

    image_path = image_store[image_id]["path"]

    try:
        img = Image.open(image_path)
        img.thumbnail((400, 400), Image.LANCZOS)

        # Convert to RGB if needed
        if img.mode in ("RGBA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
            img = bg

        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=85)
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/webp")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar thumbnail: {str(e)}")


def _format_export(img_data: dict, use_hash: bool) -> str:
    """Format tags for export to text file."""
    tags = img_data.get("tags")
    if not tags:
        return ""

    lines = []
    lines.append("# LewdTagger Export")
    lines.append(f"# Image: {img_data['filename']}")
    lines.append(f"# SHA256: {img_data.get('hash', 'N/A')}")
    lines.append(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Rating
    if tags.get("rating"):
        lines.append("## Rating")
        for name, score in tags["rating"].items():
            lines.append(f"{name} ({score:.4f})")
        lines.append("")

    # Characters
    if tags.get("characters"):
        lines.append("## Characters")
        for name, score in tags["characters"].items():
            lines.append(f"{name} ({score:.4f})")
        lines.append("")

    # General tags (comma separated for easy copy)
    if tags.get("general"):
        lines.append("## General Tags")
        tag_list = ", ".join(tags["general"].keys())
        lines.append(tag_list)
        lines.append("")
        lines.append("## General Tags (with scores)")
        for name, score in tags["general"].items():
            lines.append(f"{name} ({score:.4f})")

    return "\n".join(lines)


def _get_export_filename(img_data: dict, use_hash: bool) -> str:
    """Get the export filename based on preference."""
    if use_hash and img_data.get("hash"):
        return f"{img_data['hash'][:16]}_tags.txt"
    else:
        stem = Path(img_data["filename"]).stem
        return f"{stem}_tags.txt"


@app.post("/api/export")
async def export_tags(req: ExportRequest):
    """Export tags for a single image as a text file download."""
    if req.image_id not in image_store:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")

    img_data = image_store[req.image_id]
    if not img_data.get("tags"):
        raise HTTPException(status_code=400, detail="Imagem ainda não foi tageada")

    content = _format_export(img_data, req.use_hash)
    filename = _get_export_filename(img_data, req.use_hash)

    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/export-all")
async def export_all_tags(req: ExportAllRequest):
    """Export all tagged images as a ZIP file."""
    tagged = {k: v for k, v in image_store.items() if v.get("tags")}
    if not tagged:
        raise HTTPException(status_code=400, detail="Nenhuma imagem foi tageada ainda")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_id, img_data in tagged.items():
            content = _format_export(img_data, req.use_hash)
            filename = _get_export_filename(img_data, req.use_hash)
            zf.writestr(filename, content)

    buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="lewdtagger_export_{timestamp}.zip"'},
    )


@app.get("/api/image-tags/{image_id}")
async def get_image_tags(image_id: str):
    """Get cached tags for an already-tagged image."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")

    img_data = image_store[image_id]
    return {
        "id": image_id,
        "filename": img_data["filename"],
        "hash": img_data.get("hash"),
        "tags": img_data.get("tags"),
        "has_tags": img_data.get("tags") is not None,
    }


# ──────────────────────────────────────────────
# Static files & startup
# ──────────────────────────────────────────────

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse(str(static_dir / "index.html"))



# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("")
    print("  +-----------------------------------------+")
    print("  |       LewdTagger v1.0.0                  |")
    print("  |  Anime Image Tagger - WD SwinV2 V2      |")
    print("  |  http://localhost:8000                   |")
    print("  +-----------------------------------------+")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8000)

