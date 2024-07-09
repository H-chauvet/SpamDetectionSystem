import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

nltk.download('stopwords')

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
data = pd.read_csv(url, compression='zip', sep='\t', header=None, names=['label', 'message'])

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

data['message'] = data['message'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_vectorized, y_train)
dt_predictions = dt_classifier.predict(X_test_vectorized)

results = pd.DataFrame({'Message': X_test, 'Predicted Label': dt_predictions})

def label_to_text(label):
    return 'spam' if label == 'spam' else 'non-spam'

results['Predicted Label'] = results['Predicted Label'].apply(label_to_text)

for index, row in results.iterrows():
    print(f"Message: {row['Message']}\nPredicted Label: {row['Predicted Label']}\n")
