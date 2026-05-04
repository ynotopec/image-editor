# Image Editor API (FastAPI + uv)

Configuration orientée **idempotence**, **uv**, et exécution serveur compatible **systemd** / GPU NVIDIA (**H100 / DGX Spark**).

## 1) Installation (idempotente)

```bash
./install.sh
```

Ce script:
- installe `uv` si absent,
- crée/réutilise le virtualenv dans `~/venv/<basename_project_dir>`,
- synchronise les dépendances via `uv sync` (compatible upgrade).

## 2) Configuration

```bash
cp .env.example .env
# puis éditez .env
```

Variables importantes dans `.env.example`:
- `API_TOKEN` (token d'accès API)
- `INFERENCE_MODE`, `MODEL_ID` (défaut recommandé: `Qwen/Qwen-Image-Edit-2511`), `DEVICE`, `DTYPE`
- options commentées avec `#` pour valeurs optionnelles/défaut (dont `UNLOAD_IDLE_SECONDS=300` pour décharger le modèle après inactivité).

## 3) Lancement

```bash
source ./run.sh [IP] [PORT]
# exemple
source ./run.sh 0.0.0.0 8080
```

`run.sh`:
- charge `.env` automatiquement,
- active `~/venv/<basename_project_dir>`,
- lance `uv run uvicorn app:app ...`.

## 4) API standard (Bearer token)

### Health check
```bash
curl http://127.0.0.1:8080/healthz
```

Retourne aussi `loaded: true|false` pour indiquer si le modèle est actuellement en mémoire.

### Edit endpoint
```bash
curl -X POST http://127.0.0.1:8080/api/edit \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "prompt=Make it cinematic" \
  -F "files=@input.png"
```

Réponse:
```json
{
  "images_base64": ["data:image/png;base64,..."],
  "pipeline": "QwenImageEditPlusPipeline"
}
```

## 5) Exemple service systemd

```ini
[Unit]
Description=Image Editor API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/image-editor
ExecStart=/usr/bin/bash -lc 'source /opt/image-editor/run.sh 0.0.0.0 8080'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## 6) Compatibilité UI et upgrade modèle

- L'UI web reste compatible. Si `API_TOKEN` est activé côté backend, renseignez le champ **API token** dans l'interface avant de lancer une édition.
- Alternative possible: `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` via `MODEL_ID` dans `.env` (pipeline `QwenImageEditPlusPipeline` requis côté diffusers). Le défaut est `Qwen/Qwen-Image-Edit-2511` pour rester sur le modèle officiel généraliste.

## 7) Compatibilité H100 / DGX Spark

- `DEVICE=cuda`
- `DTYPE=bfloat16` recommandé
- pilotes NVIDIA + CUDA correctement installés côté hôte
- ajuster `MAX_SIDE` et batch d'images selon la VRAM disponible


## Notes dépendances performance

- `accelerate` est inclus dans les dépendances pour améliorer le chargement modèle (moins de RAM CPU, init plus rapide).

- `torchvision` est requis par certains processeurs Qwen2VL (sinon erreur au chargement pipeline).
