# MNIST Handwritten Digit Classification – Model Evaluation Report

This report summarizes the training, evaluation, and comparison of four machine learning classifiers applied to the MNIST handwritten digit dataset using Scikit-learn.

---

## Models Compared

We evaluated the following models:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Support Vector Machine (SVM)**

Each model was trained on the same dataset and evaluated using:

- **Accuracy**
- **F1 Macro Score**
- **ROC AUC (macro)**
- **Prediction Time**
- **Model Size**

---

## Evaluation Summary

| Model         | Accuracy | F1 Macro | ROC AUC | Time (s) | Size (KB) |
|---------------|----------|----------|---------|----------|-----------|
| Logistic      | 0.9202   | 0.9191   | 0.9919  | 0.03     | 62.2      |
| KNN           | 0.9705   | 0.9704   | 0.9930  | 16.51    | ~368000   |
| Random Forest | 0.9705   | 0.9703   | 0.9991  | 0.12     | ~140465   |
| SVM           | 0.9790   | 0.9789   | 0.9997  | 83.49    | 75676.76  |

 Large models are not tracked in GitHub. Re-run the notebooks to regenerate `.pkl` files if needed.

---

## Key Findings

- **SVM** achieved the best overall performance with 97.9% accuracy and highest ROC AUC.
- **KNN** and **Random Forest** performed similarly well but KNN is significantly slower.
- **Logistic Regression** was the fastest and smallest, but with the lowest accuracy.
- Model size and prediction time were key trade-offs, especially for KNN and Random Forest.

---

## Visual Outputs (Generated in `reports/`)

Each model outputs:

- Confusion matrix
- ROC curve
- Classification report (`*_classification_report.json`)
- Evaluation summary (`*_metrics.json`)

Comparison plots are included in `model_comparison.ipynb`.

---

## Methodology

1. Dataset: [MNIST CSV from Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
2. No feature engineering — used raw pixel values.
3. All models trained using default or commonly tuned hyperparameters.
4. Metrics computed using Scikit-learn's classification tools and ROC AUC functions.

---

## Conclusion

- **SVM** is best for accuracy and AUC, but slow.
- **Random Forest** offers a great balance of speed and performance.
- **KNN** is accurate but impractical for large-scale inference.
- **Logistic Regression** is best for lightweight tasks or baselines.

---

## Next Steps

- [ ] Build an interactive Streamlit dashboard
- [ ] Add deep learning (CNN) baseline for comparison
- [ ] Write test cases for `evaluate_model.py`

---

## Technologies

- Python 3.13
- Scikit-learn
- pandas, matplotlib, seaborn
- Jupyter Notebook

---

