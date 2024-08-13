import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Ensure nltk stopwords are downloaded
nltk.download('stopwords')
stemmer = SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Load the data
data = pd.read_csv("/Users/deepika/Downloads/dreaddit-test.csv")

# Map label values to more descriptive labels
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})

# Select only the text and label columns
data = data[["text", "label"]]

# Define the cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)
    return text

# Apply the cleaning function to the text column
data["cleaned_text"] = data["text"].apply(clean)

# Feature extraction using TfidfVectorizer
x = np.array(data["cleaned_text"])
y = np.array(data["label"])

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(x)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# Ensure classes include all possible labels
classes = np.unique(ytrain)
class_weights_array = compute_class_weight('balanced', classes=classes, y=ytrain)
class_weights = {classes[i]: class_weights_array[i] for i in range(len(classes))}

# Perform Grid Search for Logistic Regression
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid = GridSearchCV(LogisticRegression(class_weight=class_weights), param_grid, refit=True, verbose=0)
grid.fit(xtrain, ytrain)

# Save the trained model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(grid, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)
