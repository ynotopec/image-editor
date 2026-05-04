"""FastAPI web studio for Qwen image editing models.

This app supports the new Qwen-Edit-2509 Multiple Angles model which
introduces multi-image editing, extra guidance controls and a richer
front-end experience.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()  # "local" | "endpoint"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "")  # si INFERENCE_MODE=endpoint
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cuda")  # "cuda" | "cpu"
DTYPE = os.getenv("DTYPE", "bfloat16")  # "bfloat16" | "float16" | "float32"
MAX_SIDE_DEFAULT = int(os.getenv("MAX_SIDE", 1280))
API_TOKEN = os.getenv("API_TOKEN", "")
UNLOAD_IDLE_SECONDS = int(os.getenv("UNLOAD_IDLE_SECONDS", "300"))

last_activity_ts = time.time()
pipe_lock = asyncio.Lock()

app = FastAPI(title="Qwen-Image-Edit — Web Studio (multi)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Chargement pipeline local ----------
pipe = None
pipeline_name = "QwenImageEditPipeline"


async def ensure_local_pipeline_loaded() -> None:
    global pipe, pipeline_name
    if INFERENCE_MODE != "local":
        return
    if pipe is not None:
        return

    async with pipe_lock:
        if pipe is not None:
            return
        try:
            import torch
            from diffusers import QwenImageEditPipeline

            try:
                from diffusers import QwenImageEditPlusPipeline  # type: ignore
            except Exception:
                QwenImageEditPlusPipeline = None  # type: ignore

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch.set_grad_enabled(False)

            use_plus = (
                any(tag in MODEL_ID for tag in ("2509", "2511", "Multiple-Angles", "LoRA"))
                and QwenImageEditPlusPipeline is not None
            )
            if use_plus:
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype_map.get(DTYPE, torch.bfloat16),
                )
                pipeline_name = "QwenImageEditPlusPipeline"
            else:
                pipe = QwenImageEditPipeline.from_pretrained(MODEL_ID)
                pipe.to(dtype_map.get(DTYPE, torch.bfloat16))
                pipeline_name = "QwenImageEditPipeline"

            pipe.to(DEVICE)
            pipe.set_progress_bar_config(disable=None)
        except Exception as exc:
            raise RuntimeError(f"Échec du chargement du pipeline local: {exc}")


async def unload_pipeline_if_idle() -> None:
    global pipe
    if INFERENCE_MODE != "local" or UNLOAD_IDLE_SECONDS <= 0:
        return

    while True:
        await asyncio.sleep(15)
        idle_for = time.time() - last_activity_ts
        if idle_for < UNLOAD_IDLE_SECONDS:
            continue

        async with pipe_lock:
            idle_for = time.time() - last_activity_ts
            if pipe is not None and idle_for >= UNLOAD_IDLE_SECONDS:
                pipe = None
                try:
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass


@app.on_event("startup")
async def startup_event() -> None:
    if INFERENCE_MODE == "local":
        await ensure_local_pipeline_loaded()
    asyncio.create_task(unload_pipeline_if_idle())


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
) -> Image.Image:
    """Traite une image (séquentiel pour limiter la VRAM)."""
    import torch

    if pipe is None:
        raise RuntimeError("Pipeline local non initialisé")

    generator = (
        torch.manual_seed(seed)
        if seed is not None
        else torch.Generator(device=DEVICE)
    )

    inputs = dict(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else " ",
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,  # supporté par 2509, ignoré par v1
        generator=generator,
        num_images_per_prompt=1,
    )

    def _run() -> Image.Image:
        with torch.inference_mode():
            output = pipe(**inputs)
            return output.images[0]

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
) -> List[Image.Image]:
    results: List[Image.Image] = []
    for idx, img in enumerate(images):
        computed_seed = None if seed is None else (seed + idx)
        result = await run_local_inference_one(
            img,
            prompt,
            negative_prompt,
            num_inference_steps,
            true_cfg_scale,
            guidance_scale,
            computed_seed,
        )
        results.append(result)
    return results


async def run_endpoint_inference_multi(
    images: List[Image.Image],
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
) -> List[Image.Image]:
    import requests

    if not ENDPOINT_URL:
        raise HTTPException(
            500, "ENDPOINT_URL manquant pour INFERENCE_MODE=endpoint"
        )

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
    response = requests.post(
        ENDPOINT_URL, json=payload, headers=headers, timeout=1200
    )
    if response.status_code != 200:
        raise HTTPException(
            response.status_code,
            f"Endpoint error: {response.text[:500]}",
        )

    try:
        data = response.json()
        outputs: List[bytes] = []
        if isinstance(data, dict):
            if "images" in data and isinstance(data["images"], list):
                for item in data["images"]:
                    if isinstance(item, str):
                        outputs.append(base64.b64decode(item))
            elif "image" in data and isinstance(data["image"], str):
                outputs.append(base64.b64decode(data["image"]))
        if outputs:
            return [Image.open(io.BytesIO(b)).convert("RGB") for b in outputs]
    except Exception:
        pass

    return [Image.open(io.BytesIO(response.content)).convert("RGB")]




def verify_api_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(401, "Invalid API token")

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "mode": INFERENCE_MODE, "pipeline": pipeline_name, "loaded": pipe is not None})


@app.post("/api/edit", dependencies=[Depends(verify_api_token)])
async def api_edit(
    files: List[UploadFile] = File(..., description="Une ou plusieurs images"),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(30),
    true_cfg_scale: float = Form(4.0),
    guidance_scale: float = Form(1.0),
    seed: Optional[str] = Form(None),
    max_side: int = Form(MAX_SIDE_DEFAULT),
) -> JSONResponse:
    global last_activity_ts
    last_activity_ts = time.time()

    if not prompt or not prompt.strip():
        raise HTTPException(400, "Prompt requis")

    imgs: List[Image.Image] = []
    for file in files:
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as exc:
            raise HTTPException(400, f"Image invalide: {file.filename}") from exc
        imgs.append(
            smart_resize(
                img,
                max_side=max(256, min(4096, int(max_side))),
            )
        )

    if not imgs:
        raise HTTPException(400, "Aucune image fournie")

    if seed is None:
        seed_value: Optional[int] = None
    else:
        string_seed = str(seed).strip()
        seed_value = int(string_seed) if string_seed != "" else None

    if INFERENCE_MODE == "endpoint":
        out_imgs = await run_endpoint_inference_multi(
            imgs,
            prompt,
            negative_prompt,
            num_inference_steps,
            true_cfg_scale,
            guidance_scale,
            seed_value,
        )
    else:
        await ensure_local_pipeline_loaded()
        out_imgs = await run_local_inference_multi(
            imgs,
            prompt,
            negative_prompt,
            num_inference_steps,
            true_cfg_scale,
            guidance_scale,
            seed_value,
        )

    outputs_base64: List[str] = []
    for im in out_imgs:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        outputs_base64.append(
            "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        )

    return JSONResponse({"images_base64": outputs_base64, "pipeline": pipeline_name})


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
    .drop { border: 2px dashed var(--border-default-grey); border-radius: .75rem; padding: 1rem; text-align: center; transition:.2s; }
    .drop.drag { border-color: var(--border-active-blue-france); background: color-mix(in oklab, var(--background-contrast-grey), transparent 85%); }
    .chips { display:flex; gap:.5rem; flex-wrap:wrap; }
    .thumb { width: 88px; height: 88px; object-fit: cover; border-radius: .5rem; }
    .grid { display:grid; gap: .75rem; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); }
    figure { margin:0; }
    figcaption { font-size: .75rem; color: var(--text-mention-grey); margin-top:.35rem; display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
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

        <div class="fr-input-group fr-mb-2w">
          <label class="fr-label" for="apiToken">API token <span class="fr-text--xs fr-text-mention--grey">(optionnel)</span></label>
          <input id="apiToken" class="fr-input" type="password" placeholder="Bearer token" />
        </div>

        <div class="fr-grid-row fr-grid-row--gutters fr-mb-2w">
          <div class="fr-col-12">
            <label for="steps" class="fr-label">Steps <span id="stepsValue" class="fr-badge fr-badge--new fr-ml-1w">30</span></label>
            <input id="steps" name="num_inference_steps" type="range" min="10" max="75" value="30" class="fr-range" />
          </div>
          <div class="fr-col-12">
            <label for="cfg" class="fr-label">true_cfg_scale <span id="cfgValue" class="fr-badge fr-ml-1w">4.0</span></label>
            <input id="cfg" name="true_cfg_scale" type="range" min="0" max="10" step="0.1" value="4.0" class="fr-range" />
          </div>
          <div class="fr-col-12">
            <label for="guidance" class="fr-label">guidance_scale <span id="guidanceValue" class="fr-badge fr-ml-1w">1.0</span></label>
            <input id="guidance" name="guidance_scale" type="range" min="0" max="10" step="0.1" value="1.0" class="fr-range" />
          </div>
          <div class="fr-col-12">
            <label for="seed" class="fr-label">Seed <span class="fr-text--xs fr-text-mention--grey">(optionnel)</span></label>
            <input id="seed" name="seed" class="fr-input" type="number" placeholder="aléatoire si vide" />
          </div>
          <div class="fr-col-12">
            <label for="maxside" class="fr-label">Max side (px) <span id="maxsideValue" class="fr-badge fr-ml-1w">1280</span></label>
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
const apiTokenInput = document.getElementById('apiToken');
apiTokenInput.value = localStorage.getItem('api_token') || '';
apiTokenInput.addEventListener('change', ()=> localStorage.setItem('api_token', apiTokenInput.value.trim()));

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
    try {
      const blob = await (await fetch(b64)).blob();
      await navigator.clipboard.write([new ClipboardItem({[blob.type]: blob})]);
      cp.textContent='Copié ✓';
      setTimeout(()=>cp.textContent='Copier',1200);
    } catch(e){
      alert('Clipboard non supporté');
    }
  });
  div.appendChild(dl);
  div.appendChild(cp);
  return div;
}

function makeCompare(beforeUrl, afterB64){
  const wrap = document.createElement('div'); wrap.className='compare';
  const base = new Image(); base.src = beforeUrl; wrap.appendChild(base);
  const topWrap = document.createElement('div'); topWrap.className='top';
  const topImg = new Image(); topImg.src = afterB64; topWrap.appendChild(topImg);
  const divider = document.createElement('div'); divider.className='divider';
  const slider = document.createElement('input'); slider.type='range'; slider.min=0; slider.max=100; slider.value=50;
  slider.addEventListener('input', ()=>{
    const p = slider.value/100;
    topWrap.style.clipPath = `inset(0 ${(1-p)*100}% 0 0)`;
    divider.style.left = (p*100)+'%';
  });
  topWrap.style.clipPath = 'inset(0 50% 0 0)';
  wrap.appendChild(topWrap);
  wrap.appendChild(divider);
  wrap.appendChild(slider);
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
  if (!fileInput.files || fileInput.files.length === 0){
    statusEl.textContent='❌ Aucune image sélectionnée';
    return;
  }

  const fd = new FormData();
  for (const f of fileInput.files) fd.append('files', f);
  ['prompt','negative_prompt','num_inference_steps','true_cfg_scale','guidance_scale','seed','max_side']
    .forEach(k => {
      const el = document.getElementById(k);
      if(!el) return;
      const v = el.value;
      if (v !== undefined && v !== null) fd.append(k, v);
    });

  try{
    const headers = {};
    if (apiTokenInput.value.trim()) headers['Authorization'] = `Bearer ${apiTokenInput.value.trim()}`;
    const res = await fetch('/api/edit', { method:'POST', body: fd, headers });
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
    console.error(err);
    statusEl.textContent = '❌ ' + (''+err).slice(0,300);
  }
});

 document.getElementById('clearBtn').addEventListener('click', ()=>{
  form.reset();
  outGrid.innerHTML='';
  history.innerHTML='';
  inThumbs.innerHTML='';
  selInfo.textContent='Aucun fichier';
  stepsValue.textContent = steps.value = 30;
  cfgValue.textContent = cfg.value = 4.0;
  guidanceValue.textContent = guidance.value = 1.0;
  maxsideValue.textContent = maxside.value = 1280;
 });
</script>
</body>
</html>
"""
