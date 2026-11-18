# Base Classifiers vs Ensemble Learning Techniques

## Project Overview

This project compares different Machine Learning classification algorithms using the Breast Cancer Wisconsin Dataset. The main aim is to understand the performance of individual classifiers and ensemble learning techniques.

The project evaluates nine machine learning models using different performance metrics and graphical visualizations.

## Objectives

- To implement different base classification algorithms.
- To understand ensemble learning techniques.
- To compare the performance of base classifiers and ensemble models.
- To evaluate models using accuracy, precision, recall, F1-score and ROC-AUC.
- To visualize model performance using graphs and confusion matrices.

## Dataset

**Dataset Name:** Breast Cancer Wisconsin Dataset

**Source:** Scikit-learn built-in dataset.

Dataset details:
- Total Records: 569
- Total Features: 30
- Classes: Malignant and Benign
- Training Data: 80%
- Testing Data: 20%

## Machine Learning Algorithms Used

### 1. Base Classifiers
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes

### 2. Ensemble Learning Techniques
- Voting Classifier (Hard Voting)
- AdaBoost with Decision Stump
- AdaBoost with Decision Tree (Depth 3)
- Stacking Classifier with Logistic Regression as Meta Classifier

## Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Measures the overall correctness of predictions.
- **Precision:** Measures how many predicted positive cases are actually positive.
- **Recall:** Measures how many actual positive cases are correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes.
- **Cross-Validation:** Evaluates model performance using 5-fold cross-validation.

## Project Workflow

1. Import required Python libraries.
2. Load the Breast Cancer Wisconsin dataset.
3. Split the dataset into training and testing sets.
4. Create and train base classifiers.
5. Implement ensemble learning techniques.
6. Evaluate all nine models.
7. Compare performance metrics.
8. Generate graphical visualizations.
9. Save results in CSV and JSON files.

## Graphical Outputs

The project generates three graphs:

1. **Metrics Comparison:** Compares accuracy, precision, recall and F1-score of all nine models.
2. **Confusion Matrices:** Displays classification results for Voting, AdaBoost and Stacking models.
3. **Cross-Validation Accuracy:** Compares the mean and standard deviation of 5-fold cross-validation accuracy.

## Generated Files

| File Name | Description |
|---|---|
| results.csv | Performance comparison of all models |
| metrics_comparison.png | Model metrics comparison graph |
| confusion_matrices.png | Confusion matrix visualization |
| cv_accuracy.png | Cross-validation accuracy graph |
| cm.json | Confusion matrix values in JSON format |

## How to Run the Project

1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook.
3. Copy and paste the Python project code.
4. Run the code.
5. View the model comparison table and generated graphs.
6. Download the generated CSV, PNG and JSON files if required.

## Conclusion

This project demonstrates the implementation and comparison of base classifiers and ensemble learning techniques for breast cancer classification.

By evaluating different models using multiple performance metrics, we can understand their strengths and weaknesses. The comparison helps identify suitable classification techniques for this dataset.

## Author

Student Machine Learning Project

**Language:** Python  
**Domain:** Machine Learning  
**Project Type:** Classification and Ensemble Learning
