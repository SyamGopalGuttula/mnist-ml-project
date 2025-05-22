# MNIST Handwritten Digit Classification

A machine learning project that trains and compares multiple classifiers on the MNIST digit dataset using Python and Scikit-learn.

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
