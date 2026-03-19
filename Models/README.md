# CrossModal Deepfake Detection (Thesis Project)

This repository contains the code and notebooks used for a crossmodal deepfake detection research project (audio + video fusion) and benchmark comparison against several baseline networks.

## 📌 Project overview

- **Task:** Deepfake detection on the FakeAVCeleb dataset
- **Modality:** Crossmodal (audio + video); benchmarked against single-modality and other model families
- **Main contribution:** crossmodal model architecture and performance comparison with ResNet50, VGG16, MesoNet, ExceptionNet
- **Dataset:** `FakeAVCeleb_v1.2` (expected folder layout under dataset path)

## 📁 Repository structure

- `CrossModal/`
  - `combine with Aug.ipynb` (crossmodal model with augmentation)
  - `audio only/` (audio-only training and best_audioclassifier_model.pth)
  - `video only/` (video-only training and best_videoclassifier_model.pth)

- `resnet50/` (ResNet50 experiments with V+A, with/without augmentation)
- `vgg16/` (VGG16 experiments with V+A, with/without augmentation)
- `mesonet/` (MesoNet experiments with V+A, with/without augmentation)
- `exception/` (ExceptionNet / Xception experiments with V+A, with/without augmentation)

## ⚙️ Environment setup

```bash
pip install -r requirements.txt
```

> If `requirements.txt` does not exist, use:

```bash
pip install torch torchvision torchaudio transformers mediapipe librosa opencv-python ffmpeg-python timm onnxruntime-gpu insightface
```

## 🛠 Key config settings

In notebooks (example: `resnet50/resnet50-without-aug V and A.ipynb`), config is defined as:

- `DATASET_PATH`: `/kaggle/input/dataset2/FakeAVCeleb_v1.2`
- `FRAME_RATE = 10`, `FRAME_SIZE = (224, 224)`, `MAX_FRAMES = 30`
- `AUDIO_SAMPLE_RATE = 16000`, `AUDIO_DURATION = 4`, `MFCC_FEATURES = 20`
- `NUM_EPOCHS = 40`, `LEARNING_RATE = 2e-5`, `BATCH_SIZE = 8`
- `USE_AUDIO_FEATURES = True`, `USE_SPECTRAL_FEATURES = True`, `USE_FACE_DETECTION = True`

Update paths before running locally.

## 🚀 Run experiments

From root:

```bash
# example for crossmodal augmented experiment
jupyter nbconvert --to notebook --execute "CrossModal/combine with Aug.ipynb" --output "CrossModal/combine_with_aug_out.ipynb"

# example for baseline experiment
jupyter nbconvert --to notebook --execute "resnet50/resnet50-without-aug V and A.ipynb" --output "resnet50/result_resnet50_without_aug.ipynb"
```

Or open notebooks interactively in Jupyter.

## 📊 Top evaluation results

In your experiments, results include:

- AudioClassifier ROC AUC ≈ 0.9105
- VideoClassifier ROC AUC ≈ 0.9302
- High accuracy for both train and validation curves
- Confusion matrices with low false positives on real class and strong fake detection

## 🧪 Model comparisons (thesis)

- `CrossModal` (audio + video fusion): top performance, best F1/ACC balance
- `ResNet50`, `VGG16`, `MesoNet`, `ExceptionNet`: benchmarks
- Augmentation effect: compared with `with-aug` and `without-aug` variants
- Modality ablations: `audio only`, `video only`, `combined`

## ✅ Tips for reporting

1. Keep experiment IDs, seeds, and fold details in an organized log.
2. Report metrics: accuracy, precision, recall, F1, AUC, confusion matrix.
3. Visualize curve trends (train/val loss + accuracy and ROC) as in project plots.
4. Compare crossmodal vs. each baseline in one table.

## 📚 Citation

Include this project in your thesis with a short description of:
- dataset (FakeAVCeleb)
- model architecture (crossmodal fusion)
- baseline networks
- evaluation metrics

---

### Optional improvements

- add `requirements.txt` in root
- add a consolidated script `run_all.py` for standard experiments
- add `metrics.csv` for easy comparison table plotting
