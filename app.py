# app.py — Éditeur Web Qwen-Image-Edit (FastAPI)
from __future__ import annotations
import io, os, base64, asyncio
from typing import Optional
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()  # "local" | "endpoint"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit")
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "")  # si INFERENCE_MODE=endpoint
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cuda")  # "cuda" | "cpu"
DTYPE = os.getenv("DTYPE", "bfloat16")  # "bfloat16" | "float16" | "float32"
MAX_SIDE_DEFAULT = int(os.getenv("MAX_SIDE", 1024))

app = FastAPI(title="Qwen-Image-Edit Web Editor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------- Chargement pipeline local (optionnel) ----------
pipe = None
if INFERENCE_MODE == "local":
    try:
        import torch
        from diffusers import QwenImageEditPipeline
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch.set_grad_enabled(False)
        pipe = QwenImageEditPipeline.from_pretrained(MODEL_ID)
        pipe.to(dtype_map.get(DTYPE, torch.bfloat16))
        pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=None)
    except Exception as e:
        raise RuntimeError(f"Échec du chargement du pipeline local: {e}")

# ---------- Utilitaires ----------

def smart_resize(img: Image.Image, max_side: int = MAX_SIDE_DEFAULT) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))
    return img.resize((new_w, new_h), Image.LANCZOS)

async def run_local_inference(
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    true_cfg_scale: float = 4.0,
    seed: Optional[int] = None,
):
    import torch
    gen = torch.manual_seed(seed) if seed is not None else torch.Generator(device=DEVICE)
    inputs = dict(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else " ",
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        generator=gen,
    )
    def _run():
        with torch.inference_mode():
            out = pipe(**inputs)
            return out.images[0]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)

async def run_endpoint_inference(
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    true_cfg_scale: float = 4.0,
    seed: Optional[int] = None,
):
    import requests
    if not ENDPOINT_URL:
        raise HTTPException(500, "ENDPOINT_URL manquant pour INFERENCE_MODE=endpoint")
    # Encode image en base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "inputs": {
            "image": b64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
        }
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    r = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=600)
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Endpoint error: {r.text[:300]}")
    # On accepte deux formats: {image: b64} ou bien bytes direct
    try:
        data = r.json()
        if isinstance(data, dict) and "image" in data:
            out_b64 = data["image"]
            img_bytes = base64.b64decode(out_b64)
        else:
            img_bytes = r.content
    except Exception:
        img_bytes = r.content
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.post("/api/edit")
async def api_edit(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("") ,
    num_inference_steps: int = Form(30),
    true_cfg_scale: float = Form(4.0),
    # OPTION 1 — accepter seed vide (string) et convertir proprement
    seed: Optional[str] = Form(None),
    max_side: int = Form(MAX_SIDE_DEFAULT),
):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Image invalide")

    img = smart_resize(img, max_side=max(256, min(2048, int(max_side))))

    if not prompt.strip():
        raise HTTPException(400, "Prompt requis")

    # conversion robuste du seed ("" -> None)
    seed_val: Optional[int]
    if seed is None:
        seed_val = None
    else:
        s = str(seed).strip()
        seed_val = int(s) if s != "" else None

    if INFERENCE_MODE == "endpoint":
        out_img = await run_endpoint_inference(
            img, prompt, negative_prompt, num_inference_steps, true_cfg_scale, seed_val
        )
    else:
        out_img = await run_local_inference(
            img, prompt, negative_prompt, num_inference_steps, true_cfg_scale, seed_val
        )

    out_buf = io.BytesIO()
    out_img.save(out_buf, format="PNG")
    b64 = base64.b64encode(out_buf.getvalue()).decode()
    return JSONResponse({"image_base64": f"data:image/png;base64,{b64}"})

# ---------- HTML (front minimal DSFR) ----------
HTML = """
<!doctype html>
<html lang=\"fr\" data-fr-scheme=\"system\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Qwen‑Image‑Edit — Web Editor</title>
  <link href=\"https://cdn.jsdelivr.net/npm/@gouvfr/dsfr@1.11.2/dist/dsfr.min.css\" rel=\"stylesheet\">
  <style>
    body { padding: 1.25rem; }
    .thumb { width: 88px; height: 88px; object-fit: cover; border-radius: .5rem; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 1rem; }
    @media (min-width: 900px){ .grid { grid-template-columns: 360px 1fr; } }
    .history { display:flex; gap:.5rem; flex-wrap:wrap; }
    .visually-hidden { position:absolute; left:-10000px; }
  </style>
</head>
<body>
  <h1 class=\"fr-h3\">Qwen‑Image‑Edit — éditeur web</h1>
  <p class=\"fr-text--sm\">Téléversez une image, décrivez l'édition, ajustez les paramètres puis lancez.</p>

  <div class=\"grid\">
    <form id=\"editForm\" class=\"fr-card fr-p-3w\" style=\"align-self:start;\">
      <div class=\"fr-input-group\">
        <label class=\"fr-label\" for=\"file\">Image d'entrée</label>
        <input class=\"fr-upload\" id=\"file\" name=\"file\" type=\"file\" accept=\"image/*\" required />
      </div>

      <div class=\"fr-input-group\">
        <label class=\"fr-label\" for=\"prompt\">Prompt</label>
        <textarea id=\"prompt\" name=\"prompt\" class=\"fr-input\" rows=\"3\" placeholder=\"Ex: Remplacer le texte de l'enseigne par ‘Boulangerie du Pont’\" required></textarea>
      </div>

      <div class=\"fr-input-group\">
        <label class=\"fr-label\" for=\"neg\">Negative prompt <span class=\"fr-text--xs fr-text-mention--grey\">(optionnel)</span></label>
        <input id=\"neg\" name=\"negative_prompt\" class=\"fr-input\" placeholder=\"artefacts, blur\" />
      </div>

      <div class=\"fr-grid-row fr-grid-row--gutters\">
        <div class=\"fr-col-6\">
          <label class=\"fr-label\" for=\"steps\">Steps: <span id=\"stepsValue\">30</span></label>
          <input id=\"steps\" name=\"num_inference_steps\" type=\"range\" min=\"10\" max=\"75\" value=\"30\" class=\"fr-range\" />
        </div>
        <div class=\"fr-col-6\">
          <label class=\"fr-label\" for=\"cfg\">true_cfg_scale: <span id=\"cfgValue\">4.0</span></label>
          <input id=\"cfg\" name=\"true_cfg_scale\" type=\"range\" min=\"0\" max=\"10\" step=\"0.1\" value=\"4.0\" class=\"fr-range\" />
        </div>
      </div>

      <div class=\"fr-grid-row fr-grid-row--gutters\">
        <div class=\"fr-col-6\">
          <label class=\"fr-label\" for=\"seed\">Seed <span class=\"fr-text--xs fr-text-mention--grey\">(optionnel)</span></label>
          <input id=\"seed\" name=\"seed\" class=\"fr-input\" type=\"number\" placeholder=\"aléatoire si vide\" />
        </div>
        <div class=\"fr-col-6\">
          <label class=\"fr-label\" for=\"maxside\">Max side (px): <span id=\"maxsideValue\">1024</span></label>
          <input id=\"maxside\" name=\"max_side\" type=\"range\" min=\"512\" max=\"2048\" step=\"64\" value=\"1024\" class=\"fr-range\" />
        </div>
      </div>

      <div class=\"fr-btns-group fr-btns-group--inline-md fr-mt-3w\">
        <button id=\"runBtn\" class=\"fr-btn\">Lancer l'édition</button>
        <button id=\"clearBtn\" class=\"fr-btn fr-btn--secondary\" type=\"button\">Effacer</button>
      </div>

      <p id=\"status\" class=\"fr-text--sm fr-mt-2w\" aria-live=\"polite\"></p>
    </form>

    <section class=\"fr-card fr-p-3w\">
      <h2 class=\"fr-h5\">Résultat</h2>
      <div id=\"previewZone\" class=\"fr-mb-3w\" style=\"min-height: 240px; display:flex; align-items:center; justify-content:center; background:var(--background-alt-grey); border-radius:.5rem;\"></div>
      <div class=\"history\" id=\"history\"></div>
    </section>
  </div>

<script>
const steps = document.getElementById('steps');
const cfg = document.getElementById('cfg');
const maxside = document.getElementById('maxside');
const stepsValue = document.getElementById('stepsValue');
const cfgValue = document.getElementById('cfgValue');
const maxsideValue = document.getElementById('maxsideValue');
[steps, cfg, maxside].forEach(i => i.addEventListener('input', () => {
  stepsValue.textContent = steps.value; cfgValue.textContent = cfg.value; maxsideValue.textContent = maxside.value;
}));

const form = document.getElementById('editForm');
const statusEl = document.getElementById('status');
const preview = document.getElementById('previewZone');
const history = document.getElementById('history');

function showImage(b64){
  const img = new Image();
  img.src = b64; img.style.maxWidth = '100%'; img.style.borderRadius = '.5rem';
  preview.innerHTML = ''; preview.appendChild(img);
  const t = new Image(); t.src = b64; t.className = 'thumb'; history.prepend(t);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault(); statusEl.textContent = '⏳ Génération en cours…';
  const fd = new FormData(form);
  try{
    const res = await fetch('/api/edit', { method:'POST', body: fd });
    if(!res.ok){ throw new Error(await res.text()); }
    const data = await res.json();
    showImage(data.image_base64);
    statusEl.textContent = '✅ Fini';
  }catch(err){
    console.error(err); statusEl.textContent = '❌ ' + (''+err).slice(0,200);
  }
});

document.getElementById('clearBtn').addEventListener('click', ()=>{
  form.reset(); preview.innerHTML = ''; statusEl.textContent = '';
  stepsValue.textContent = steps.value = 30;
  cfgValue.textContent = cfg.value = 4.0;
  maxsideValue.textContent = maxside.value = 1024;
});
</script>
</body>
</html>
"""
root@ai-dev-3:~# cp -aZ /home/ailab/qwen-image-edit/app.py /home/ailab/qwen-image-edit/app.py-OK-25-09-25
root@ai-dev-3:~# nano /home/ailab/qwen-image-edit/requirements.txt 
root@ai-dev-3:~# ls -lart /home/ailab/qwen-image-edit/
total 60
-rw-rw-r-- 1 ailab ailab   697 Aug 20 17:46 .env
-rw-rw-r-- 1 ailab ailab  1609 Aug 20 17:47 run.sh
drwxr-x--- 7 ailab ailab  4096 Aug 20 18:01 ..
-rw-rw-r-- 1 ailab ailab 11518 Aug 25 09:24 app.py-OK-25-09-25
-rw-rw-r-- 1 ailab ailab 11518 Aug 25 09:24 app.py-OK-14-09-25
-rw-rw-r-- 1 ailab ailab 11518 Aug 25 09:24 app.py
drwxrwxr-x 2 ailab ailab  4096 Aug 25 09:24 __pycache__
-rw-rw-r-- 1 ailab ailab   304 Sep 26 17:11 requirements.txt
drwxrwxr-x 3 ailab ailab  4096 Sep 26 17:11 .
root@ai-dev-3:~# nano /home/ailab/qwen-image-edit/run.sh
root@ai-dev-3:~# > /home/ailab/qwen-image-edit/app.py
root@ai-dev-3:~# nano /home/ailab/qwen-image-edit/app.py
root@ai-dev-3:~# tmux
[exited]
root@ai-dev-3:~# cat /home/ailab/qwen-image-edit/app.py
# app.py — Éditeur Web Qwen-Image-Edit (FastAPI) — multi-image + front DSFR premium
from __future__ import annotations
import io, os, base64, asyncio
from typing import Optional, List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()  # "local" | "endpoint"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2509")
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "")  # si INFERENCE_MODE=endpoint
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cuda")          # "cuda" | "cpu"
DTYPE = os.getenv("DTYPE", "bfloat16")        # "bfloat16" | "float16" | "float32"
MAX_SIDE_DEFAULT = int(os.getenv("MAX_SIDE", 1280))

app = FastAPI(title="Qwen-Image-Edit — Web Studio (multi)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Chargement pipeline local ----------
pipe = None
pipeline_name = "QwenImageEditPipeline"
if INFERENCE_MODE == "local":
    try:
        import torch
        from diffusers import QwenImageEditPipeline
        try:
            # Nouveau pipeline pour 2509 (multi-images amélioré)
            from diffusers import QwenImageEditPlusPipeline
        except Exception:
            QwenImageEditPlusPipeline = None

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch.set_grad_enabled(False)

        use_plus = (("2509" in MODEL_ID) or MODEL_ID.endswith("-2509")) and (QwenImageEditPlusPipeline is not None)
        if use_plus:
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                MODEL_ID, torch_dtype=dtype_map.get(DTYPE, torch.bfloat16)
            )
            pipeline_name = "QwenImageEditPlusPipeline"
        else:
            pipe = QwenImageEditPipeline.from_pretrained(MODEL_ID)
            pipe.to(dtype_map.get(DTYPE, torch.bfloat16))
            pipeline_name = "QwenImageEditPipeline"

        pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=None)
    except Exception as e:
        raise RuntimeError(f"Échec du chargement du pipeline local: {e}")

# ---------- Utilitaires ----------

def smart_resize(img: Image.Image, max_side: int = MAX_SIDE_DEFAULT) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))
    return img.resize((new_w, new_h), Image.LANCZOS)

async def run_local_inference_one(
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
):
    """Traite une image (séquentiel pour limiter la VRAM)."""
    import torch
    gen = torch.manual_seed(seed) if seed is not None else torch.Generator(device=DEVICE)

    inputs = dict(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else " ",
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,      # supporté par 2509, ignoré par v1
        generator=gen,
        num_images_per_prompt=1,
    )

    def _run():
        with torch.inference_mode():
            out = pipe(**inputs)
            return out.images[0]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)

async def run_local_inference_multi(
    images: List[Image.Image],
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    seed: Optional[int],
):
    results: List[Image.Image] = []
    for idx, img in enumerate(images):
        s = None if seed is None else (seed + idx)
        out_img = await run_local_inference_one(
            img, prompt, negative_prompt,
            num_inference_steps, true_cfg_scale, guidance_scale, s
        )
        results.append(out_img)
    return results

async def run_endpoint_inference_multi(
    images: List[Image.Image],
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
):
    import requests
    if not ENDPOINT_URL:
        raise HTTPException(500, "ENDPOINT_URL manquant pour INFERENCE_MODE=endpoint")

    b64_list = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_list.append(base64.b64encode(buf.getvalue()).decode())

    payload = {
        "inputs": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
    }
    if len(b64_list) == 1:
        payload["inputs"]["image"] = b64_list[0]
    else:
        payload["inputs"]["images"] = b64_list

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    r = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=1200)
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Endpoint error: {r.text[:500]}")

    try:
        data = r.json()
        outs: List[bytes] = []
        if isinstance(data, dict):
            if "images" in data and isinstance(data["images"], list):
                for it in data["images"]:
                    if isinstance(it, str):
                        outs.append(base64.b64decode(it))
            elif "image" in data and isinstance(data["image"], str):
                outs.append(base64.b64decode(data["image"]))
        if outs:
            return [Image.open(io.BytesIO(b)).convert("RGB") for b in outs]
    except Exception:
        pass

    return [Image.open(io.BytesIO(r.content)).convert("RGB")]

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.post("/api/edit")
async def api_edit(
    files: List[UploadFile] = File(..., description="Une ou plusieurs images"),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(30),
    true_cfg_scale: float = Form(4.0),
    guidance_scale: float = Form(1.0),
    seed: Optional[str] = Form(None),
    max_side: int = Form(MAX_SIDE_DEFAULT),
):
    if not prompt or not prompt.strip():
        raise HTTPException(400, "Prompt requis")

    imgs: List[Image.Image] = []
    for f in files:
        try:
            content = await f.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Image invalide: {f.filename}")
        imgs.append(smart_resize(img, max_side=max(256, min(4096, int(max_side)))))

    if not imgs:
        raise HTTPException(400, "Aucune image fournie")

    seed_val: Optional[int]
    if seed is None:
        seed_val = None
    else:
        s = str(seed).strip()
        seed_val = int(s) if s != "" else None

    if INFERENCE_MODE == "endpoint":
        out_imgs = await run_endpoint_inference_multi(
            imgs, prompt, negative_prompt, num_inference_steps, true_cfg_scale, guidance_scale, seed_val
        )
    else:
        out_imgs = await run_local_inference_multi(
            imgs, prompt, negative_prompt, num_inference_steps, true_cfg_scale, guidance_scale, seed_val
        )

    out_b64_list = []
    for im in out_imgs:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out_b64_list.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())

    return JSONResponse({"images_base64": out_b64_list, "pipeline": pipeline_name})

# ---------- HTML (front DSFR premium + drag&drop + comparaison) ----------
HTML = r"""
<!doctype html>
<html lang="fr" data-fr-scheme="system">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen‑Image‑Edit — Web Studio</title>
  <link href="https://cdn.jsdelivr.net/npm/@gouvfr/dsfr@1.11.2/dist/dsfr.min.css" rel="stylesheet">
  <style>
    :root { --pane-bg: var(--background-alt-grey); }
    body { padding: 1.5rem; }
    .layout { display:grid; grid-template-columns: 380px 1fr; gap: 1.25rem; }
    @media (max-width: 1024px){ .layout { grid-template-columns: 1fr; } }
    .card { background: var(--background-default-grey); border-radius: 1rem; box-shadow: 0 10px 30px rgba(0,0,0,.05); }
    .pad { padding: 1rem 1.25rem; }
    .pane { background: var(--pane-bg); border-radius: .75rem; padding: 1rem; }
    .drop { border: 2px dashed var(--border-default-grey); border-radius: .75rem; padding: 1rem; text-align: center; transition: .2s; }
    .drop.drag { border-color: var(--border-active-blue-france); background: color-mix(in oklab, var(--background-contrast-grey), transparent 85%); }
    .chips { display:flex; gap:.5rem; flex-wrap:wrap; }
    .thumb { width: 88px; height: 88px; object-fit: cover; border-radius: .5rem; }
    .grid { display:grid; gap: .75rem; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); }
    figure { margin:0; }
    figcaption { font-size: .75rem; color: var(--text-mention-grey); margin-top:.35rem; display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
    .btn-icon { inline-size: 2.25rem; block-size: 2.25rem; border-radius: .5rem; display:grid; place-items:center; }
    .kv { display:grid; grid-template-columns: 1fr auto; gap:.75rem .75rem; align-items:center; }
    .kv label { color: var(--text-mention-grey); }
    .split { display:grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
    .compare { position:relative; overflow:hidden; border-radius:.75rem; }
    .compare > img { display:block; width:100%; height:auto; }
    .compare > .top { position:absolute; inset:0; overflow:hidden; }
    .compare > .top > img { width:100%; height:auto; display:block; }
    .compare input[type=range] { position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:ew-resize; }
    .divider { position:absolute; top:0; bottom:0; width:2px; background: var(--border-default-grey); left: 50%; }
  </style>
</head>
<body>
  <header class="fr-mb-3w">
    <h1 class="fr-h3 fr-mb-1w">Qwen‑Image‑Edit — Web Studio</h1>
    <p class="fr-text--sm">Multi‑image • Comparaison avant/après • DSFR • Pipeline: <span id="pipe">—</span></p>
  </header>

  <div class="layout">
    <aside class="card pad">
      <form id="editForm">
        <div class="fr-input-group fr-mb-2w">
          <label class="fr-label" for="prompt">Prompt</label>
          <textarea id="prompt" name="prompt" class="fr-input" rows="3" placeholder="Ex: Remplacer l'enseigne par ‘Boulangerie du Pont’, style photo réaliste, texte net" required></textarea>
        </div>

        <div class="fr-input-group fr-mb-2w">
          <label class="fr-label" for="neg">Negative prompt <span class="fr-text--xs fr-text-mention--grey">(optionnel)</span></label>
          <input id="neg" name="negative_prompt" class="fr-input" placeholder="artefacts, blur" />
        </div>

        <div class="kv fr-mb-2w">
          <label for="steps">Steps</label>
          <div>
            <span id="stepsValue" class="fr-badge fr-badge--new fr-mr-1w">30</span>
            <input id="steps" name="num_inference_steps" type="range" min="10" max="75" value="30" class="fr-range" />
          </div>
          
          <label for="cfg">true_cfg_scale</label>
          <div>
            <span id="cfgValue" class="fr-badge fr-mr-1w">4.0</span>
            <input id="cfg" name="true_cfg_scale" type="range" min="0" max="10" step="0.1" value="4.0" class="fr-range" />
          </div>

          <label for="guidance">guidance_scale</label>
          <div>
            <span id="guidanceValue" class="fr-badge fr-mr-1w">1.0</span>
            <input id="guidance" name="guidance_scale" type="range" min="0" max="10" step="0.1" value="1.0" class="fr-range" />
          </div>

          <label for="seed">Seed <span class="fr-text--xs fr-text-mention--grey">(optionnel)</span></label>
          <input id="seed" name="seed" class="fr-input" type="number" placeholder="aléatoire si vide" />

          <label for="maxside">Max side (px)</label>
          <div>
            <span id="maxsideValue" class="fr-badge fr-mr-1w">1280</span>
            <input id="maxside" name="max_side" type="range" min="512" max="4096" step="64" value="1280" class="fr-range" />
          </div>
        </div>

        <div class="fr-input-group fr-mb-2w">
          <label class="fr-label" for="files">Images d'entrée</label>
          <div id="drop" class="drop" role="button" tabindex="0">
            <p class="fr-text--sm fr-mb-1v">Glissez‑déposez vos images ici</p>
            <p class="fr-text--xs fr-text-mention--grey">ou cliquez pour sélectionner</p>
            <input id="files" name="files" type="file" accept="image/*" multiple hidden />
          </div>
          <div id="selInfo" class="fr-hint-text fr-mt-1w">Aucun fichier</div>
          <div id="inThumbs" class="chips fr-mt-1w"></div>
        </div>

        <div class="fr-btns-group fr-btns-group--inline-md fr-mt-3w">
          <button id="runBtn" class="fr-btn" type="submit">Lancer</button>
          <button id="clearBtn" class="fr-btn fr-btn--secondary" type="button">Effacer</button>
        </div>

        <p id="status" class="fr-text--sm fr-mt-2w" aria-live="polite"></p>
      </form>
    </aside>

    <main class="card pad">
      <h2 class="fr-h5 fr-mb-1w">Résultats</h2>
      <div class="grid" id="outGrid"></div>
      <h3 class="fr-h6 fr-mt-3w">Historique</h3>
      <div class="chips" id="history"></div>
    </main>
  </div>

<script>
const steps = document.getElementById('steps');
const cfg = document.getElementById('cfg');
const guidance = document.getElementById('guidance');
const maxside = document.getElementById('maxside');
const stepsValue = document.getElementById('stepsValue');
const cfgValue = document.getElementById('cfgValue');
const guidanceValue = document.getElementById('guidanceValue');
const maxsideValue = document.getElementById('maxsideValue');
[steps, cfg, guidance, maxside].forEach(i => i.addEventListener('input', () => {
  stepsValue.textContent = steps.value;
  cfgValue.textContent = cfg.value;
  guidanceValue.textContent = guidance.value;
  maxsideValue.textContent = maxside.value;
}));

const form = document.getElementById('editForm');
const statusEl = document.getElementById('status');
const outGrid = document.getElementById('outGrid');
const history = document.getElementById('history');
const pipeSpan = document.getElementById('pipe');
const drop = document.getElementById('drop');
const selInfo = document.getElementById('selInfo');
const inThumbs = document.getElementById('inThumbs');
const fileInput = document.getElementById('files');

function fmtBytes(bytes){
  if (!bytes && bytes !== 0) return '—';
  const units = ['o','Ko','Mo','Go'];
  let i = 0; while (bytes >= 1024 && i < units.length-1){ bytes/=1024; i++; }
  return (Math.round(bytes*10)/10)+' '+units[i];
}

function humanList(files){
  if(!files || files.length===0) return 'Aucun fichier';
  let total=0; [...files].forEach(f=> total+=f.size);
  return `${files.length} fichier(s) · ${fmtBytes(total)}`;
}

function makeThumb(file, url){
  const img = new Image();
  img.src = url; img.className = 'thumb'; img.title = file.name;
  return img;
}

function updateSelection(files){
  selInfo.textContent = humanList(files);
  inThumbs.innerHTML = '';
  [...files].forEach(f=>{
    const url = URL.createObjectURL(f);
    inThumbs.appendChild(makeThumb(f, url));
  });
}

// Drag & drop
['dragenter','dragover'].forEach(ev=> drop.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.add('drag'); }));
['dragleave','drop'].forEach(ev=> drop.addEventListener(ev, e=>{ e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('click', ()=> fileInput.click());
drop.addEventListener('drop', e=>{
  e.preventDefault();
  const dt = e.dataTransfer; if(!dt || !dt.files) return;
  fileInput.files = dt.files; updateSelection(fileInput.files);
});
fileInput.addEventListener('change', ()=> updateSelection(fileInput.files));

function addHistory(b64){
  const t = new Image(); t.src = b64; t.className = 'thumb'; history.prepend(t);
}

function makeControls(b64){
  const div = document.createElement('div');
  const dl = document.createElement('button'); dl.type = 'button'; dl.className = 'fr-btn fr-btn--secondary fr-btn--sm'; dl.textContent = 'Télécharger';
  dl.addEventListener('click', ()=>{
    const a = document.createElement('a'); a.href = b64; a.download = 'qwen-edit.png'; a.click();
  });
  const cp = document.createElement('button'); cp.type = 'button'; cp.className = 'fr-btn fr-btn--tertiary fr-btn--sm'; cp.textContent = 'Copier';
  cp.addEventListener('click', async ()=>{
    try { const blob = await (await fetch(b64)).blob(); await navigator.clipboard.write([new ClipboardItem({[blob.type]: blob})]); cp.textContent='Copié ✓'; setTimeout(()=>cp.textContent='Copier',1200); } catch(e){ alert('Clipboard non supporté'); }
  });
  div.appendChild(dl); div.appendChild(cp); return div;
}

function makeCompare(beforeUrl, afterB64){
  const wrap = document.createElement('div'); wrap.className='compare';
  const base = new Image(); base.src = beforeUrl; wrap.appendChild(base);
  const topWrap = document.createElement('div'); topWrap.className='top';
  const topImg = new Image(); topImg.src = afterB64; topWrap.appendChild(topImg);
  const divider = document.createElement('div'); divider.className='divider';
  const slider = document.createElement('input'); slider.type='range'; slider.min=0; slider.max=100; slider.value=50;
  slider.addEventListener('input', ()=>{
    const p = slider.value/100; topWrap.style.clipPath = `inset(0 ${(1-p)*100}% 0 0)`; divider.style.left = (p*100)+'%';
  });
  // init
  topWrap.style.clipPath = 'inset(0 50% 0 0)';
  wrap.appendChild(topWrap); wrap.appendChild(divider); wrap.appendChild(slider);
  return wrap;
}

function showOutputs(inputs, outputs){
  outGrid.innerHTML='';
  const count = Math.max(inputs.length, outputs.length);
  for(let i=0;i<count;i++){
    const card = document.createElement('figure');
    const beforeUrl = inputs[i] ? URL.createObjectURL(inputs[i]) : null;
    const afterB64 = outputs[i] || outputs[outputs.length-1];

    if(beforeUrl){
      card.appendChild(makeCompare(beforeUrl, afterB64));
    } else {
      const img = new Image(); img.src = afterB64; img.style.width='100%'; img.style.borderRadius='.75rem';
      card.appendChild(img);
    }

    const cap = document.createElement('figcaption');
    cap.innerHTML = `<span>Sortie #${i+1}</span>`;
    cap.appendChild(makeControls(afterB64));
    card.appendChild(cap);

    outGrid.appendChild(card);
    addHistory(afterB64);
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault(); statusEl.textContent = '⏳ Génération en cours…';
  if (!fileInput.files || fileInput.files.length === 0){ statusEl.textContent='❌ Aucune image sélectionnée'; return; }

  const fd = new FormData();
  for (const f of fileInput.files) fd.append('files', f);
  ['prompt','negative_prompt','num_inference_steps','true_cfg_scale','guidance_scale','seed','max_side']
    .forEach(k => { const v = document.getElementById(k)?.value; if (v !== undefined && v !== null) fd.append(k, v); });

  try{
    const res = await fetch('/api/edit', { method:'POST', body: fd });
    if(!res.ok){ throw new Error(await res.text()); }
    const data = await res.json();
    if(Array.isArray(data.images_base64)){
      showOutputs(fileInput.files, data.images_base64);
      pipeSpan.textContent = data.pipeline || '—';
      statusEl.textContent = '✅ Fini';
    } else {
      throw new Error('Réponse invalide');
    }
  }catch(err){
    console.error(err); statusEl.textContent = '❌ ' + (''+err).slice(0,300);
  }
});

// Reset
 document.getElementById('clearBtn').addEventListener('click', ()=>{
  form.reset(); outGrid.innerHTML=''; history.innerHTML=''; inThumbs.innerHTML=''; selInfo.textContent='Aucun fichier';
  stepsValue.textContent = steps.value = 30;
  cfgValue.textContent = cfg.value = 4.0;
  guidanceValue.textContent = guidance.value = 1.0;
  maxsideValue.textContent = maxside.value = 1280;
 });
</script>
</body>
</html>
"""
