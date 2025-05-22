# MNIST Handwritten Digit Classification – Final Report

This report summarizes the development, evaluation, and comparison of four machine learning classification models applied to the MNIST handwritten digit dataset using Scikit-learn.

---

## Objective

The goal was to implement, evaluate, and compare different classification models on the MNIST dataset to identify which approach performs best in terms of accuracy, speed, and model size.

---

## Models Evaluated

### 1. **Logistic Regression**
- A linear model used for classification.
- Works well when data is linearly separable.
- **Limitations**: MNIST digits are non-linear, so performance is limited.

### 2. **K-Nearest Neighbors (KNN)**
- Instance-based learning algorithm.
- Makes predictions based on the closest training examples.
- **Strengths**: High accuracy on MNIST.
- **Weaknesses**: Extremely slow at prediction time, very large model size.

### 3. **Random Forest**
- Ensemble of decision trees.
- Good at handling non-linear data and reducing overfitting.
- **Strengths**: High accuracy, reasonable training/prediction time.
- **Weaknesses**: Still a large model size.

### 4. **Support Vector Machine (SVM)**
- Constructs hyperplanes in high-dimensional space.
- Uses RBF kernel to handle non-linear data.
- **Strengths**: Best overall accuracy and AUC.
- **Weaknesses**: Very slow training and prediction time.

---

## Evaluation Metrics

| Model         | Accuracy | F1 Macro | ROC AUC | Time (s) | Size (KB) |
|---------------|----------|----------|---------|----------|-----------|
| Logistic      | 0.9202   | 0.9191   | 0.9919  | 0.03     | 62.2      |
| KNN           | 0.9705   | 0.9704   | 0.9930  | 16.51    | ~368000   |
| Random Forest | 0.9705   | 0.9703   | 0.9991  | 0.12     | ~140465   |
| SVM           | 0.9790   | 0.9789   | 0.9997  | 83.49    | 75676.76  |

 Models exceeding 100MB were not committed to GitHub due to file size limits.

---

## Key Insights

- **SVM** achieved the best accuracy and AUC but has the slowest inference time.
- **Random Forest** is the best all-around performer for accuracy and speed.
- **KNN** is accurate but completely impractical for production due to size and speed.
- **Logistic Regression** is extremely lightweight and fast but not suitable for complex patterns in MNIST.

---

## Methodology

- Dataset: [MNIST CSV (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- No feature engineering — raw pixel values used as input.
- Training and testing done using Scikit-learn.
- Evaluation used Accuracy, F1 Macro, ROC AUC, Prediction Time, and Model Size.

---

## Outputs

All models generate:

- ROC curve image
- Confusion matrix
- Classification report JSON
- Summary metrics JSON

All are saved in the `reports/` folder.

---

## Conclusion

| Best For        | Model             |
|------------------|--------------------|
| Accuracy         | SVM                |
| Speed + Accuracy | Random Forest      |
| Simplicity       | Logistic Regression|
| Worst Trade-offs | KNN                |

---

## Future Work

- Add a CNN-based deep learning model (e.g. with TensorFlow)
- Create a Streamlit dashboard for live comparison
- Package evaluation as a reusable module or CLI tool
