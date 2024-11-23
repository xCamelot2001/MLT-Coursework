import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the file
file = 'text8.txt'
with open(file, 'r') as f:
    text = f.read()

# Function to clean the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

# Apply the cleaning function
cleaned_data = clean_text(text)

# Tokenize the cleaned text
tokens = word_tokenize(cleaned_data)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_tokens = [word for word in tokens if word not in stop_words]

# Convert filtered tokens into a single string
processed_text = ' '.join(filtered_tokens)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([processed_text])

# Display the TF-IDF matrix
print("TF-IDF Matrix Shape:", X.shape)
print("TF-IDF Sample (Sparse Matrix):", X)
