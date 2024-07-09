import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt

# Télécharger les stopwords de nltk
nltk.download('stopwords')

# Charger le dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
data = pd.read_csv(url, compression='zip', sep='\t', header=None, names=['label', 'message'])

# Prétraiter les données
def preprocess_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Supprimer les stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

data['message'] = data['message'].apply(preprocess_text)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectoriser les données textuelles
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Entraîner un classificateur Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_vectorized, y_train)
dt_predictions = dt_classifier.predict(X_test_vectorized)

# Afficher les résultats
results = pd.DataFrame({'Message': X_test, 'Predicted Label': dt_predictions})

def label_to_text(label):
    return 'spam' if label == 'spam' else 'non-spam'

results['Predicted Label'] = results['Predicted Label'].apply(label_to_text)

# Afficher les résultats
for index, row in results.iterrows():
    print(f"Message: {row['Message']}\nPredicted Label: {row['Predicted Label']}\n")
