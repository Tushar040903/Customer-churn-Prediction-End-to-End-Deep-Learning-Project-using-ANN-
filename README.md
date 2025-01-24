# Customer Churn Prediction - End-to-End Deep Learning Project using ANN

## Overview  
This project demonstrates an end-to-end pipeline for predicting customer churn using Artificial Neural Networks (ANN). It includes data preprocessing, model development, evaluation, and deployment.

---

## Table of Contents  
- [Overview](#overview)
- [Problem Statement](#problem-statement)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Project Workflow](#project-workflow)  
- [Model Details](#model-details)  
- [Results](#results)  
- [Usage](#usage)  
- [License](#license)

---

## Problem Statement
In today's competitive banking industry, customer retention is crucial for maintaining profitability and growth. Losing customers, also known as "customer churn," directly impacts a bank's revenue and reputation. Identifying customers likely to leave the bank allows institutions to take proactive measures to retain them, such as offering personalized services, better incentives, or addressing their concerns.

This project leverages deep learning techniques to predict customer churn based on various factors such as customer demographics, account details, and transaction history. By accurately identifying at-risk customers, banks can strategically focus their retention efforts, thereby improving customer satisfaction and reducing churn rates.

---

## Features  
- Preprocessing customer data to handle missing values, encode categorical features, and normalize numerical data.  
- Building an Artificial Neural Network for binary classification using TensorFlow and Keras.  
- Training and evaluating the ANN model on accuracy, precision, recall, and F1-score.  
- Easily configurable and extendable for other datasets or business cases.  

---

## Technologies Used  
- Python  
- Pandas, NumPy (Data preprocessing)   
- TensorFlow, Keras (Deep Learning framework)  
- Scikit-learn (Metrics and preprocessing utilities)  

---

## Installation  

1. Clone this repository:  
   ```bash  
   git clone https://github.com/your-username/Customer-Churn-Prediction.git  
   cd Customer-Churn-Prediction  

---

## Project Workflow  
1. **Dataset Exploration and Preprocessing**  
   - Load and explore customer data (e.g., demographics, usage, tenure).  
   - Handle missing values, encode categorical features, and scale numerical data.  

2. **Model Development**  
   - Built an ANN with input, hidden, and output layers:  
     - Hidden Layers: ReLU activation.  
     - Output Layer: Sigmoid activation for binary classification.  
   - Optimized with the Adam optimizer and binary cross-entropy loss.  

3. **Model Evaluation**  
   - Evaluated the model on metrics such as Accuracy, Precision, Recall, and F1-Score.  

4. **Deployment (Optional)**  
   - Exported the trained model for inference on new customer data.  

---

## Model Details  
- Input layer: Accepts preprocessed features.  
- Hidden layers: Fully connected layers with ReLU activation.  
- Output layer: A single neuron with sigmoid activation (outputs probability of churn).  
- Optimizer: Adam  
- Loss Function: Binary Cross-Entropy  

---

## Results  
The ANN model performed well on the test data. Key performance metrics include:  
- **Accuracy:** 86%  
- **Precision:** 70%  
- **Recall:** 48%  
- **F1 Score:** 57%  

Example visualization of loss and accuracy trends during training:  
![Training Metrics](https://via.placeholder.com/600x300)  

---

## Usage  

1. Run all cells in `experiments.ipynb` to train the model and preprocess the data.  

2. Start the Streamlit application:  
   ```bash  
   streamlit run app.py  

---

## License
This project is licensed under the [MIT License](LICENSE).  

