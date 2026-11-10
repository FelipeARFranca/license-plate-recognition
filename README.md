**# License Plate Recognition — Commands**

**## Environment**

```powershell
uv python pin 3.13
uv lock
```

**### Desktop — NVIDIA CUDA**

```powershell
uv sync --extra cu130 --locked
```

**### Laptop — CPU**

```powershell
uv sync --extra cpu --locked
```

**## Dependencies**

```powershell
uv add ultralytics opencv-python bentoml pyyaml matplotlib tqdm pandas
```

**## PyTorch / CUDA verification**

```powershell
uv run --extra cu130 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**## GPU monitoring**

```powershell
nvidia-smi -l 1
```

**## Dataset inspection and validation**

```powershell
uv run --extra cu130 python src/detection/inspect_dataset.py
uv run --extra cu130 python src/detection/validate_dataset.py
```

**## Annotation preview**

```powershell
uv run --extra cu130 python src/detection/preview_annotations.py
```

**## YOLO training**

**### Smoke test**

```powershell
uv run --extra cu130 python src/detection/train.py --smoke-test
```

**### Quick experiment**

```powershell
uv run --extra cu130 python src/detection/train.py --quick-experiment
```

**### Full training**

```powershell
uv run --extra cu130 python src/detection/train.py
```

**## Detector evaluation**

Evaluate the trained detector on the held-out test split:

```powershell
uv run --extra cu130 python src/detection/evaluate.py
```

Evaluation results and plots are saved under `runs/detection/yolo26n_test/`.

**## BentoML — Register the detector**

Register the trained `best.pt` checkpoint in the local BentoML Model Store:

```powershell
uv run --extra cu130 python src/detection/register_model.py
```

List registered models and inspect the detector metadata:

```powershell
uv run --extra cu130 bentoml models list
uv run --extra cu130 bentoml models get license_plate_detector:latest
```

**## BentoML — Export the model for GitHub**

Export the registered model as a `.bentomodel` archive inside the repository:

```powershell
New-Item -ItemType Directory -Force .\models\bentoml
uv run --extra cu130 bentoml models export license_plate_detector:latest .\models\bentoml\license_plate_detector.bentomodel
```

The archive contains the registered model and its BentoML metadata. It allows another machine to restore the model without running training again. Do not commit the original dataset or the `runs/` training outputs unless they are intentionally being versioned.

**## BentoML — Import the model on another machine**

After cloning the repository and installing the project dependencies, import the archive into that machine's local BentoML Model Store:

```powershell
uv run --extra cu130 bentoml models import .\models\bentoml\license_plate_detector.bentomodel
uv run --extra cu130 bentoml models list
```

On the CPU-only laptop, use the CPU extra explicitly:

```powershell
uv run --extra cpu bentoml models import .\models\bentoml\license_plate_detector.bentomodel
uv run --extra cpu bentoml models list
```

**\*\*Important:\*\*** On a machine restored from GitHub, do not run `register_model.py` again. That script expects the original training checkpoint under `runs/`. Import the committed `.bentomodel` archive instead; no retraining is required.

**## BentoML — Run the detection service**

Start the service from the project root. Use the CUDA extra on the desktop:

```powershell
uv run --extra cu130 bentoml serve services.detection_service:LicensePlateDetection
```

On the CPU-only laptop:

```powershell
uv run --extra cpu bentoml serve services.detection_service:LicensePlateDetection
```

The service is normally available at `[http://localhost:3000](http://localhost:3000)`, with the detection API at `[http://localhost:3000/detect](http://localhost:3000/detect)`.

**## BentoML — Test the detection endpoint**

Keep the BentoML server running, open a second terminal, and run:

**### Desktop — NVIDIA CUDA**

```powershell
uv run --extra cu130 python src/detection/test_service.py
```

**### Laptop — CPU**

```powershell
uv run --extra cpu python src/detection/test_service.py
```

After testing, stop the BentoML server with `Ctrl+C` in the server terminal.

**## BentoML — OCR model commands**

```powershell
uv run --extra cu130 python src/ocr/register_model.py

uv run --extra cu130 bentoml models list

uv run --extra cu130 bentoml models get license_plate_ocr:latest

uv run --extra cu130 bentoml serve services.ocr_service:LicensePlateOCR

New-Item -ItemType Directory -Force .\models\bentoml\

uv run --extra cu130 bentoml models export license_plate_ocr:latest .\models\bentoml\license_plate_ocr.bentomodel

uv run --extra cpu bentoml models import .\models\bentoml\license_plate_ocr.bentomodel

uv run --extra cpu bentoml models list

uv run --extra cpu bentoml serve services.ocr_service:LicensePlateOCR
```