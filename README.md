# License Plate Recognition — Commands

This README collects the commands used to inspect, train, evaluate, serve, and package the license plate recognition pipeline.

The project has two model components:

- **Detection:** custom-trained YOLO26n model using the Large License Plate Detection Dataset.
- **OCR:** pretrained Fast-Plate-OCR `cct-s-v2-global-model`, evaluated on the Brazilian License Plate OCR dataset.

The complete BentoML pipeline runs detection, crops each plate with Pillow, recognizes its text, and returns JSON metadata, Base64-encoded plate crops, and an annotated original image.

---

## Environment

Pin the project Python version and update the lockfile:

```powershell
uv python pin 3.13
uv lock
```

### Desktop — NVIDIA CUDA

```powershell
uv sync --extra cu130 --locked
```

### Laptop — CPU

```powershell
uv sync --extra cpu --locked
```

After changing dependencies with `uv add`, commit both `pyproject.toml` and `uv.lock`.

## Dependencies

Add the project dependencies:

```powershell
uv add ultralytics opencv-python bentoml pyyaml matplotlib tqdm pandas pillow
uv add "fast-plate-ocr[onnx]"
```

`fast-plate-ocr[onnx]` installs Fast-Plate-OCR and its ONNX Runtime inference extra.

## PyTorch / CUDA verification

```powershell
uv run --extra cu130 python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The Bento build configurations pin the same Torch versions as the project, one per device:

| Bento config | Torch | TorchVision |
|---|---|---|
| `bentofile.cu130.yaml` | `torch==2.14.1+cu130` | `torchvision==0.29.1+cu130` |
| `bentofile.cpu.yaml` | `torch==2.14.1+cpu` | `torchvision==0.29.1+cpu` |

If the installed project versions change, update the corresponding pins in both bentofiles before rebuilding the Bentos.

## GPU monitoring

```powershell
nvidia-smi -l 1
```

---

# Detection — YOLO26n

## Dataset inspection and validation

```powershell
uv run --extra cu130 python src/detection/inspect_dataset.py
uv run --extra cu130 python src/detection/validate_dataset.py
```

## Annotation preview

```powershell
uv run --extra cu130 python src/detection/preview_annotations.py
```

## YOLO training

### Smoke test

```powershell
uv run --extra cu130 python src/detection/train.py --smoke-test
```

### Quick experiment

```powershell
uv run --extra cu130 python src/detection/train.py --quick-experiment
```

### Full training

```powershell
uv run --extra cu130 python src/detection/train.py
```

## Detector evaluation

Evaluate the trained detector on the held-out test split. The script uses the GPU when CUDA is available and falls back to CPU otherwise.

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 python src/detection/evaluate.py
```

### Laptop — CPU

```powershell
uv run --extra cpu python src/detection/evaluate.py
```

Evaluation results and plots are saved under `runs/detection/yolo26n_test/`.

## Register the detector in BentoML

Register the trained `best.pt` checkpoint in the local BentoML Model Store:

```powershell
uv run --extra cu130 python src/detection/register_model.py
```

List registered models and inspect detector metadata:

```powershell
uv run --extra cu130 bentoml models list
uv run --extra cu130 bentoml models get license_plate_detector:latest
```

---

# OCR — Fast-Plate-OCR

The OCR component uses the pretrained `cct-s-v2-global-model`. The project does not train CRNN or LPRNet. The Brazilian dataset is used for evaluation, not fine-tuning.

## Run OCR on one cropped plate

```powershell
uv run --extra cu130 python src/ocr/predict.py --image "datasets/ocr/plates/plate_imgs/plate_0.png"
```

The script defaults to CPU ONNX Runtime inference. To select a device explicitly:

```powershell
uv run --extra cu130 python src/ocr/predict.py --image "path\to\plate.png" --device cpu
```

## Evaluate the Brazilian OCR dataset

Expected dataset layout:

```text
datasets/
└── ocr/
    └── plates/
        ├── plates.csv
        └── plate_imgs/
            ├── plate_0.png
            ├── plate_1.png
            └── ...
```

The CSV must contain `filename` and `plate` columns. The evaluation script resolves image references by filename and, if needed, by filename stem to tolerate path or extension differences.

Run a small smoke test first:

```powershell
uv run --extra cu130 python src/ocr/evaluate.py --limit 10
```

Run the complete evaluation:

```powershell
uv run --extra cu130 python src/ocr/evaluate.py
```

The evaluation defaults to `cct-s-v2-global-model` and CPU inference. To choose a device explicitly:

```powershell
uv run --extra cu130 python src/ocr/evaluate.py --device cpu
```

Results are saved under `runs/ocr/evaluation/`:

- `brazilian_plates_predictions.csv`
- `brazilian_plates_summary.json`

The current recorded baseline on 206 images is:

| Metric | Result |
|---|---:|
| Exact plate accuracy | 94.17% (194/206) |
| Character Error Rate | 1.46% |
| Mean inference time | 25.46 ms |
| Median inference time | 22.91 ms |
| P95 inference time | 38.95 ms |

These are the results from the evaluated dataset and CPU inference configuration; they are not a guarantee of performance on other data.

## Register the pretrained OCR model in BentoML

This downloads or retrieves the pretrained model and registers its ONNX file and matching `plate_config.yaml` in the local BentoML Model Store. It does not train or fine-tune the model.

```powershell
uv run --extra cu130 python src/ocr/register_model.py
```

Inspect the registered OCR model:

```powershell
uv run --extra cu130 bentoml models list
uv run --extra cu130 bentoml models get license_plate_ocr:latest
```

---

# BentoML — Export individual models

The `.bentomodel` archives contain individual registered model artifacts and BentoML metadata. They are useful for restoring the models into another machine's local Model Store.

Create the destination directory:

```powershell
New-Item -ItemType Directory -Force .\models\bentoml
```

## Export the detector

```powershell
uv run --extra cu130 bentoml models export license_plate_detector:latest .\models\bentoml\license_plate_detector.bentomodel
```

## Export the OCR model

```powershell
uv run --extra cu130 bentoml models export license_plate_ocr:latest .\models\bentoml\license_plate_ocr.bentomodel
```

## Import individual models on another machine

After cloning the repository and installing the project dependencies, restore both model archives into that machine's local BentoML Model Store.

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml models import .\models\bentoml\license_plate_detector.bentomodel
uv run --extra cu130 bentoml models import .\models\bentoml\license_plate_ocr.bentomodel
uv run --extra cu130 bentoml models list
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml models import .\models\bentoml\license_plate_detector.bentomodel
uv run --extra cpu bentoml models import .\models\bentoml\license_plate_ocr.bentomodel
uv run --extra cpu bentoml models list
```

**Important:** On a machine restored from GitHub, do not rerun the detector registration script if the training checkpoint under `runs/` is unavailable. Import the committed `.bentomodel` archive instead. The OCR registration script can download the pretrained model, but importing its committed archive avoids downloading it again.

---

# BentoML — Run individual services

## Detection service

Start the service from the project root.

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml serve services.detection_service:LicensePlateDetection
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml serve services.detection_service:LicensePlateDetection
```

The detection API is normally available at:

```text
http://localhost:3000/detect
```

## OCR service

### Desktop — NVIDIA CUDA project environment

```powershell
uv run --extra cu130 bentoml serve services.ocr_service:LicensePlateOCR
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml serve services.ocr_service:LicensePlateOCR
```

Stop an individual BentoML server with `Ctrl+C` in its server terminal.

---

# Complete pipeline

The pipeline service composes the detector and OCR services. It receives an original vehicle image, detects plates, crops each detection with Pillow, recognizes each crop, and returns:

- Detection bounding boxes and confidence values.
- OCR plate text and optional character confidence values.
- Each cropped plate as a Base64-encoded PNG.
- The annotated original image as a Base64-encoded PNG.
- Per-stage and total processing timings.

## Run the source pipeline service

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml serve services.pipeline_service:LicensePlatePipeline
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml serve services.pipeline_service:LicensePlatePipeline
```

The complete pipeline endpoint is normally available at:

```text
http://localhost:3000/recognize
```

Stop the server with `Ctrl+C`.

## Command-line client

With any pipeline server running (source service, built Bento, or Docker container), send an image from a second terminal:

```powershell
uv run --extra cpu license-plate-recognition "path\to\vehicle.jpg"
```

The client calls `/recognize`, prints each detected plate, and saves `annotated.png`, one `plate_<id>.png` per crop, and `result.json` (the response with Base64 data replaced by file paths) under `runs/pipeline/<image name>/`.

Useful options:

```powershell
uv run --extra cpu license-plate-recognition "path\to\vehicle.jpg" `
    --url http://localhost:3000 `
    --confidence-threshold 0.4 `
    --iou-threshold 0.7 `
    --output-dir runs\pipeline\my_test `
    --no-ocr-confidence
```

The client only sends HTTP requests, so the `--extra` choice here does not affect where inference runs.

---

# BentoML — Package the complete pipeline

The pipeline is packaged per device, mirroring the `cu130` / `cpu` project extras. Pick the configuration that matches the machine that will **run** the Bento or container:

| Device | Build config | Bento name | Torch wheels |
|---|---|---|---|
| NVIDIA CUDA 13.0 | `bentofile.cu130.yaml` | `license_plate_pipeline_cu130` | `torch==2.14.1+cu130`, `torchvision==0.29.1+cu130` |
| CPU only | `bentofile.cpu.yaml` | `license_plate_pipeline_cpu` | `torch==2.14.1+cpu`, `torchvision==0.29.1+cpu` |

Both configurations define the same Pipeline Service, both registered models, and the same Docker environment. They differ only in the Torch/TorchVision builds, which are fetched from the matching official PyTorch wheel pages; every other package comes from PyPI. Separate Bento names keep `:latest` unambiguous when both are built on one machine.

`.bentoignore` excludes datasets, training runs, local model archives, and development files from both builds.

## Verify both registered models

```powershell
uv run --extra cu130 bentoml models list
```

The Model Store must contain:

```text
license_plate_detector:latest
license_plate_ocr:latest
```

## Build the Bento

Run from the repository root.

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml build -f bentofile.cu130.yaml
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml build -f bentofile.cpu.yaml
```

List built Bentos:

```powershell
uv run --extra cpu bentoml list
```

Inspect the latest complete pipeline Bento:

```powershell
uv run --extra cu130 bentoml get license_plate_pipeline_cu130:latest
uv run --extra cpu bentoml get license_plate_pipeline_cpu:latest
```

## Test the built Bento

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml serve license_plate_pipeline_cu130:latest
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml serve license_plate_pipeline_cpu:latest
```

This tests the packaged Bento rather than loading the service directly from source. Locally, `bentoml serve` runs the Bento with the current `uv` environment, so the `--extra` must match the device.

## Containerize the Bento with Docker

Docker Desktop must be running and configured for Linux containers before containerization.

Verify that the Docker client can reach its engine:

```powershell
docker version
docker context ls
```

`docker version` must display both Client and Server sections.

Build the container image from the Bento for the target device.

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml containerize license_plate_pipeline_cu130:latest
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml containerize license_plate_pipeline_cpu:latest
```

The image's dependencies are the ones locked inside the Bento, so the **Bento name** decides whether the image is CUDA or CPU. The `--extra` only selects the local environment that runs the BentoML CLI.

Inspect the generated image:

```powershell
docker images
```

Run the image locally, replacing `YOUR_BENTO_VERSION` with the tag shown by Docker.

### Desktop — NVIDIA CUDA

The host needs an NVIDIA driver and GPU support in Docker (Docker Desktop with the WSL 2 backend provides it on Windows). The CUDA runtime itself ships inside the PyTorch wheels.

```powershell
docker run --rm --gpus all -p 3000:3000 license_plate_pipeline_cu130:YOUR_BENTO_VERSION serve
```

Without `--gpus all`, the CUDA image still starts but the detector falls back to CPU.

### Laptop — CPU

```powershell
docker run --rm -p 3000:3000 license_plate_pipeline_cpu:YOUR_BENTO_VERSION serve
```

The API is then available at:

```text
http://localhost:3000/recognize
```

### Docker troubleshooting

If containerization reports that the Docker backend is not healthy:

1. Start Docker Desktop and wait until its engine is running.
2. Ensure Docker Desktop is using Linux containers.
3. Run `docker version` and check that the Server section is present.
4. If the WSL backend is unresponsive, save work in WSL and run:

```powershell
wsl --shutdown
```

Then reopen Docker Desktop and wait for the engine to start.

The first container build can take time because it downloads the base image and installs the locked runtime dependencies. The CUDA image is considerably larger than the CPU image because of the bundled CUDA libraries.

## Export the complete Bento archive

A `.bento` archive is different from an individual `.bentomodel`: it packages the complete application, including source code, locked dependencies, configuration, and referenced model artifacts.

```powershell
New-Item -ItemType Directory -Force .\models\bentoml
```

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml export license_plate_pipeline_cu130:latest .\models\bentoml\license_plate_pipeline_cu130.bento
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml export license_plate_pipeline_cpu:latest .\models\bentoml\license_plate_pipeline_cpu.bento
```

Import, list, and serve the complete Bento on another machine.

### Desktop — NVIDIA CUDA

```powershell
uv run --extra cu130 bentoml import .\models\bentoml\license_plate_pipeline_cu130.bento
uv run --extra cu130 bentoml list
uv run --extra cu130 bentoml serve license_plate_pipeline_cu130:latest
```

### Laptop — CPU

```powershell
uv run --extra cpu bentoml import .\models\bentoml\license_plate_pipeline_cpu.bento
uv run --extra cpu bentoml list
uv run --extra cpu bentoml serve license_plate_pipeline_cpu:latest
```

The `.bento` archive is an alternative to deploying the Docker image. You do not need to export both unless you want both distribution formats.

---

## Useful references

- [BentoML — Packaging for deployment](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html)
- [BentoML — Bento build options](https://docs.bentoml.com/en/latest/reference/bentoml/bento-build-options.html)
- [BentoML — CLI reference](https://docs.bentoml.com/en/latest/reference/bentoml/cli.html)
- [BentoML — Model loading and management](https://docs.bentoml.com/en/latest/build-with-bentoml/model-loading-and-management.html)
- [Fast-Plate-OCR documentation](https://ankandrew.github.io/fast-plate-ocr/latest/)

## Datasets

- [Large-License-Plate-Detection-Dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset)
- [Brazilian license plate OCR](https://www.kaggle.com/datasets/luisasr2/brazilian-license-plate-ocr)