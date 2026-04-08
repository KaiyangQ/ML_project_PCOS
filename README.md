# PCOS Detection from Ovarian Ultrasound Images

A binary image classification project that detects Polycystic Ovary Syndrome (PCOS) from ovarian ultrasound images using both traditional machine learning and deep learning approaches.

---

## Dataset

**Kaggle — PCOS Detection using Ultrasound Images**

| Split | Images |
|-------|--------|
| Train | ~1,924 |
| Test  | ~1,922 |

Binary labels: **infected** (PCOS) vs. **not infected**.

---

## Approaches

### 1. Traditional Machine Learning (`code/machine_learning_project.ipynb`)

Images are resized to 64×64 grayscale and converted into a 38-dimensional feature vector per image:

- **Grayscale statistics** — mean, standard deviation, and related global intensity descriptors
- **GLCM texture features** — Haralick features (contrast, correlation, energy, homogeneity, etc.) via `scikit-image`
- **Canny edge count** — total edge pixel count from Canny edge detection (OpenCV)
- **32-bin pixel intensity histogram** — normalized histogram of grayscale pixel values

Models trained on these features:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 99.9% | — |
| Random Forest | 100% | 5-fold cross-validation |

A t-SNE projection of the feature space confirms strong class separability.

---

### 2. Deep Learning (`code/deep_learning_project.ipynb`)

Transfer learning with a pretrained ResNet18 backbone (ImageNet weights), fine-tuned end-to-end.

| Setting | Value |
|---------|-------|
| Architecture | ResNet18 (all layers fine-tuned) |
| Pretrained weights | ImageNet |
| Optimizer | Adam |
| Epochs | 25 |
| Input resolution | 224×224 |
| Augmentation | RandomResizedCrop(224), RandomHorizontalFlip |

| Model | Test Accuracy |
|-------|---------------|
| ResNet18 (fine-tuned) | 100% |

---

## Results Summary

| Approach | Best Model | Test Accuracy |
|----------|-----------|---------------|
| Traditional ML | Random Forest | 100% |
| Deep Learning | ResNet18 | 100% |

---

## Visualizations

Exploratory and diagnostic plots are saved in `code/`:

- `tsne_plot.png` — t-SNE projection of the 38-dimensional feature space
- `edge_count_boxplot.png` — distribution of Canny edge counts by class
- `mean_brightness_hist.png` — mean brightness histogram by class
- `pixel_histogram_plot.png` — average 32-bin pixel intensity histograms per class
- `std_kde_plot.png` — KDE of pixel standard deviation by class

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Deep Learning | PyTorch, torchvision |
| Classical ML | scikit-learn |
| Image / Feature Engineering | scikit-image, OpenCV |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |

---

## File Structure

```
ML_project_PCOS/
└── code/
    ├── machine_learning_project.ipynb   # Traditional ML pipeline
    ├── deep_learning_project.ipynb      # Deep learning pipeline (ResNet18)
    ├── requirements.txt                 # Python dependencies
    ├── tsne_plot.png
    ├── edge_count_boxplot.png
    ├── mean_brightness_hist.png
    ├── pixel_histogram_plot.png
    └── std_kde_plot.png
```

---

## Setup

```bash
pip install -r code/requirements.txt
```

Then open either notebook in Jupyter and run all cells. Ensure the dataset is placed at the path expected by the notebook (see the data-loading cell at the top of each notebook).
