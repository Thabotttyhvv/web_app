#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import re
import pandas as pd


#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression, DecisionTreeClassifier
#from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
df = pd.read_csv('train.csv')

# Split the data
#x_train, x_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.25)

# Vectorization using TfidfVectorizer
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

# Model training - Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)

# Model training - Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)

# Streamlit App
st.title("Hate Speech Detection App")

# Sidebar for user input
text_input = st.text_area("Enter a text for hate speech detection:")

if text_input:
    # Vectorize the input text
    vectorized_text = vectorization.transform([text_input])

    # Predictions
    logistic_prediction = logistic_model.predict(vectorized_text)[0]
    decision_tree_prediction = decision_tree_model.predict(vectorized_text)[0]

    # Display predictions
    st.subheader("Logistic Regression Prediction:")
    st.write("Hate Speech" if logistic_prediction else "Non-Hate Speech")

    st.subheader("Decision Tree Prediction:")
    st.write("Hate Speech" if decision_tree_prediction else "Non-Hate Speech")

# Display model evaluation results
st.subheader("Model Evaluation Results")

# Logistic Regression
st.write("Logistic Regression Accuracy on Training Data:", accuracy_score(y_train, logistic_model.predict(x_train)))
st.write("Logistic Regression Accuracy on Test Data:", accuracy_score(y_test, logistic_model.predict(x_test)))

# Decision Tree Classifier
st.write("Decision Tree Accuracy on Training Data:", accuracy_score(y_train, decision_tree_model.predict(x_train)))
st.write("Decision Tree Accuracy on Test Data:", accuracy_score(y_test, decision_tree_model.predict(x_test)))

# Confusion Matrix for Decision Tree Classifier
st.subheader("Confusion Matrix for Decision Tree Classifier")
cm = confusion_matrix(y_test, decision_tree_model.predict(x_test))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
st.pyplot(cm_display.plot())
