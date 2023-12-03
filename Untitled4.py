import pickle
import streamlit as st
import pandas as pd
import string

# Load the preprocessed DataFrame
df = pd.read_csv('train.csv')

# Sidebar for user input
st.sidebar.title("Hate Speech Detection App")
text_input = st.sidebar.text_area("Enter a text for hate speech detection:")

# Preprocessing functions
#def remove_punctuations(text):
  #  return text.translate(str.maketrans('', '', string.punctuation))

#def remove_stopwords(text):
 #   stop_words = set(stopwords.words('english'))
 #   words = [word for word in text.split() if word.lower() not in stop_words]
#  return ' '.join(words)

# Preprocess the input text
#if text_input:
  #  text_input = text_input.lower()
  #  text_input = remove_punctuations(text_input)
   # text_input = remove_stopwords(text_input)

# Display the input text
if text_input:
    st.subheader("Input Text:")
    st.write(text_input)

    # Vectorization using TfidfVectorizer
    vectorization = TfidfVectorizer()
    x_train = vectorization.fit_transform(df['tweet'])
    x_input = vectorization.transform([text_input])

    # Model training - Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, df['label'])

    # Model training - Decision Tree Classifier
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(x_train, df['label'])

    # Predictions
    logistic_prediction = logistic_model.predict(x_input)[0]
    decision_tree_prediction = decision_tree_model.predict(x_input)[0]

    # Display predictions
    st.subheader("Logistic Regression Prediction:")
    st.write("Hate Speech" if logistic_prediction else "Non-Hate Speech")

    st.subheader("Decision Tree Prediction:")
    st.write("Hate Speech" if decision_tree_prediction else "Non-Hate Speech")

# Display model evaluation results
st.subheader("Model Evaluation Results")

# Logistic Regression
#st.write("Logistic Regression Accuracy:", accuracy_score(df['label'], logistic_model.predict(x_train)))

# Decision Tree Classifier
#st.write("Decision Tree Accuracy:", accuracy_score(df['label'], decision_tree_model.predict(x_train)))

# Confusion Matrix for Decision Tree Classifier
st.subheader("Confusion Matrix for Decision Tree Classifier")
#cm = confusion_matrix(df['label'], decision_tree_model.predict(x_train))
#cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
#st.pyplot(plt.figure(figsize=(8, 6)))
#st.pyplot(cm_display.plot())
