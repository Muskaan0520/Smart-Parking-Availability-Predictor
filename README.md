# Smart Parking Availability Predictor

## Project Overview
This project predicts parking availability using Machine Learning.

It classifies parking availability into three categories:
- High
- Medium
- Low

## Dataset
The project uses 300 synthetically generated observations with the following input features:
- Hour
- Weekend
- Event
- Peak_Hour

## Machine Learning Models
Five classification algorithms are implemented:

1. Decision Tree
2. Random Forest
3. Logistic Regression
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Evaluation
Models are compared using:
- Accuracy
- Precision
- Recall
- F1 Score
- 5-Fold Cross Validation

## How to Run

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Run the Python file:

```bash
python smart_parking_predictor.py
```
