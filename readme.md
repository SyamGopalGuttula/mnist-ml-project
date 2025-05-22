```markdown
# MNIST Handwritten Digit Classification

A machine learning project that trains and compares multiple classifiers on the MNIST digit dataset using Python and Scikit-learn.

---

## Models Trained

This project compares the performance of four classifiers:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)

Each model was evaluated on:

- Accuracy
- F1 Score (Macro)
- ROC AUC
- Prediction Time
- Model Size

---

## Evaluation Summary (Latest)

| Model         | Accuracy | F1 Macro | ROC AUC | Time (s) | Size (KB) |
|---------------|----------|----------|---------|----------|-----------|
| Logistic      | 0.9202   | 0.9191   | 0.9919  | 0.03     | 62.2      |
| KNN           | 0.9705   | 0.9704   | 0.9930  | 16.51    | ~368000 ❗ |
| Random Forest | 0.9705   | 0.9703   | 0.9991  | 0.12     | ~140465 ❗ |
| SVM           | 0.9790   | 0.9789   | 0.9997  | 83.49    | 75676.76  |

 Large models are not pushed to GitHub due to file size limits — re-run notebooks to regenerate them.

---

## Running This Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mnist-ml-project.git
cd mnist-ml-project
````

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebooks

```bash
jupyter notebook
```

Open any notebook inside the `notebooks/` folder:

* `logistic_regression.ipynb`
* `knn_model.ipynb`
* `random_forest_model.ipynb`
* `svm_model.ipynb`
* `model_comparison.ipynb` (visualizes all results)

---

## Visual Outputs

Each model saves:

* Confusion matrix plot
* ROC curve plot
* Classification report (JSON)
* Full summary in `*_metrics.json`

These are stored in the `reports/` folder.

---

## Technologies Used

* Python 3.13
* scikit-learn
* pandas, matplotlib, seaborn
* Jupyter Notebook

---
