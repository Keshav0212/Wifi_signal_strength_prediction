# WiFi Signal Strength Prediction and Localization

This project aims to predict WiFi signal strength and localize mobile devices within indoor environments using machine learning models. By analyzing signal strengths from multiple WiFi access points, the models can estimate both the signal strength of unknown access points and accurately pinpoint a device's location.

---

## Project Overview

WiFi-based indoor localization is a technique used to estimate a device's position by leveraging signal strength variations across multiple WiFi access points. This technology has significant real-world applications, such as:
- **Indoor Navigation**
- **Location-Based Services (LBS)**
- **Smart Building Context-Aware Services**
- **Security and Emergency Response Systems**

---

## Project Objectives and Workflow

### **Objective 1: Preprocessing the WiFi Signal Dataset**
- **Goal**: Ensure the dataset is clean, normalized, and properly formatted for model training.
- **Steps**:
  - Handling missing values and outliers.
  - Normalizing signal strength values.
  - Encoding room labels (1 - 4) to facilitate supervised learning.
  
  **Dataset Overview**:  
  - 2000 rows representing observations from 7 WiFi access points.
  - Each row contains signal strength data and a corresponding room number (1-4).

### **Objective 2: Classification Models for Room Localization**
- **Goal**: Train models to classify rooms based on WiFi signal strengths.
- **Models**: 
  - Naive Bayes
  - Support Vector Machine (SVM) with both linear and RBF kernels
  - Random Forest Classifier
- **Evaluation Metrics**: 
  - Accuracy, Precision, Recall, F1 Score

A comparative analysis found the **Random Forest Classifier** to have the highest accuracy at **98%**, making it the most effective for indoor localization. This model identifies which room the device is located in based on signal strengths.

### **Objective 3: Regression Models for WiFi Signal Strength Prediction**
- **Goal**: Predict the signal strength of one WiFi access point using the signal strengths of the other access points and the room number.
- **Model**: Random Forest Regressor
- **Evaluation Metrics**: 
  - Mean Squared Error (MSE), R-squared

First, **hyperparameter tuning** was applied to optimize the model’s parameters. Without tuning, the model achieved 60% accuracy. After tuning, the model improved significantly to **99.99% accuracy**. The results were validated using joint plots to compare predicted vs. actual signal strengths.

---

## Data Preprocessing

The dataset was preprocessed to ensure accuracy and consistency before model training:
- **Loading** the dataset from CSV files.
- **Label Encoding** the room numbers (1, 2, 3, 4).
- **Splitting** the dataset into training (80%) and testing (20%) sets.

### Features: 
- Signal strengths from 7 WiFi access points.

### Target: 
- Room number (for classification) or specific WiFi signal strength (for regression).

---

## Model Training and Evaluation

### **Classification Models**
A variety of models were trained to classify the room based on WiFi signal strength, and their performance was evaluated. Below are the results:
- **Naive Bayes**: Accuracy = 0.982
- **SVM (Linear Kernel)**: Accuracy = 0.978
- **SVM (RBF Kernel, with Grid Search Tuning)**: Accuracy = 0.982
- **Random Forest Classifier**: Accuracy = 0.982

**Evaluation** was done using a **Confusion Matrix** and a **Classification Report** to assess Precision, Recall, and F1-score.

### **Regression Model**
For signal strength prediction, a **Random Forest Regressor** was used. Hyperparameter tuning yielded the following best parameters:
- `max_depth`: None
- `min_samples_leaf`: 1
- `min_samples_split`: 2
- `n_estimators`: 200

This model achieved an accuracy of **99.99%** and was evaluated using MSE and R-squared metrics.

---

## Visualizations and Results

- **Correlation Heatmaps** were used to illustrate relationships between predicted and actual values.
- **Joint Plots** were employed to verify signal strength prediction performance.
- **Feature Importance** graphs from the Random Forest models highlighted the significance of different WiFi access points in both classification and regression tasks.

---

## Conclusion

The project successfully demonstrated the use of machine learning for both WiFi signal strength prediction and indoor localization. Among all models, the **Random Forest** algorithms excelled in both tasks. The classification model accurately identified rooms with **98% accuracy**, while the regression model predicted WiFi signal strengths with **99.99% accuracy** after tuning.

---

## Acknowledgements

This dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization).


