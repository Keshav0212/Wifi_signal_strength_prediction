# WiFi Signal Strength Prediction
This project focuses on predicting WiFi signal strength and mobile phone localization using machine learning models. By analyzing signal strengths received from multiple WiFi access points, the model can estimate the precise location of a device in an indoor environment.

---

## Project Overview
WiFi localization is a technique used to determine the location of a device based on varying signal strengths from multiple WiFi access points. This method is applied to predict the room a device is in or estimate signal strength in different environments. Applications include:
- **Indoor Navigation**
- **Location-Based Services**
- **Context-Aware Services in Smart Buildings**
- **Security and Emergency Response Scenarios**

---

## Project Specifications
- **Objective 1**: Preprocess the WiFi signal dataset to ensure it is clean and suitable for training/testing models.
  - Steps: Handling missing values, normalizing signal strengths, and label encoding rooms.

- **Objective 2**: Train and evaluate classification models to determine the room based on WiFi signal strengths.
  - Models: Naive Bayes, SVM, Random Forest
  - Metrics: Accuracy, Precision, Recall, F1 Score

- **Objective 3**: Train and evaluate regression models to predict the signal strength of WiFi access points.
  - Models: Random Forest, Support Vector Regression (SVR)
  - Metrics: Mean Squared Error (MSE), R-squared

- **Objective 4**: Compare model performances based on accuracy, computational efficiency, and robustness.

---

## Data Preprocessing
The dataset consists of WiFi signal strengths recorded in different rooms. Preprocessing steps include:
- **Loading** the dataset from a CSV file.
- **Label Encoding** the rooms (1, 2, 3, 4).
- **Splitting** the dataset into training and testing sets.

Features: Signal strengths of 7 WiFi signals.  
Target: Room number.

---

## Model Training and Evaluation (Classification)
A comparative analysis was performed using various classification models. Here are the results:
- **Naive Bayes Classification**: Accuracy of 0.982
- **SVM (Linear Kernel)**: Accuracy of 0.978
- **SVM (RBF Kernel)**: Accuracy of 0.982 (with Grid Search tuning)
- **Random Forest Classification**: Accuracy of 0.982

Evaluation involved the **Confusion Matrix** and **Classification Report** to assess Precision, Recall, and F1-score.

---

## Regression Analysis
Regression models were used to predict the signal strength of WiFi signals. Results:
- **Random Forest Regressor**: Achieved 99.99% accuracy with the following parameters:
  - `max_depth`: None
  - `min_samples_leaf`: 1
  - `min_samples_split`: 2
  - `n_estimators`: 200

---

## Results and Visualizations
Correlation heatmaps were used to visualize the relationship between predicted and actual values. The Random Forest models showed superior performance for both classification and regression tasks.

---

## Conclusion
The machine learning models demonstrated high accuracy in WiFi localization tasks. Random Forest models provided the best performance. Future work could explore additional features or deep learning models to further enhance localization accuracy.

---

## Acknowledgements
This dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization).  

