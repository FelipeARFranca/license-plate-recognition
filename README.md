# license-plate-recognition

# License Plate Recognition — Commands

## Environment

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

## Dependencies

```powershell
uv add ultralytics opencv-python bentoml pyyaml matplotlib tqdm pandas
```

## PyTorch / CUDA verification

```powershell
uv run --extra cu130 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## GPU monitoring

```powershell
nvidia-smi -l 1
```

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

