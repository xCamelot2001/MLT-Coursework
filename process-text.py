#Some code from NLTK.ORG

from urllib import request #import some library
import nltk, re, pprint
from nltk import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy

# Load the file
file = 'text8.txt'
with open(file, 'r') as f:
    text = f.read()

print(text)

url = "http://www.gutenberg.org/files/2554/2554-0.txt" #download this text file from the Internet
response = request.urlopen(url) #search for this function online
raw = response.read().decode('utf8') #Search online: https://www.tutorialspoint.com/python/string_decode.htm
type(raw) #what is the type of the variable
print(len(raw)) #what is the length of the text file, number of words
print(raw[:75]) #print the first 75 characters
tokens = word_tokenize(raw)
type(tokens)
print(len(tokens))
print(tokens[:10])
text = nltk.Text(tokens) #https://www.nltk.org/api/nltk.text.Text.html
print(type(text))
print(text[1024:1062])
print(text.collocations()) #https://www.nltk.org/api/nltk.collocations.html

#Let's build a vocabulary.
words = [w.lower() for w in tokens]
print(type(words))
vocab = sorted(set(words))
print(type(vocab))
print(vocab)

#https://www.geeksforgeeks.org/co-occurence-matrix-in-nlp/
#Let's build cooccurrence counts
window_size = 2 #How many words in sequence to consider to be in the window
# Create a list of co-occurring word pairs
co_occurrences = defaultdict(Counter)
for i, word in enumerate(words):
    for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
        if i != j:
            co_occurrences[word][words[j]] += 1

# Create a list of unique words
unique_words = list(set(words))
# Initialize the co-occurrence matrix
co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

# Populate the co-occurrence matrix
word_index = {word: idx for idx, word in enumerate(unique_words)}
for word, neighbors in co_occurrences.items():
    for neighbor, count in neighbors.items():
        co_matrix[word_index[word]][word_index[neighbor]] = count

# Create a DataFrame for better readability
co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

# Display the co-occurrence matrix
print(co_matrix_df)

#Convert the above matrix to sparse representation, saves memory
print(scipy.sparse.csr_matrix(co_matrix_df))

#TODO:
#1. Remove words containing non-ASCII characters
#2. Remove punctuations https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
#3. Remove stopwords https://www.geeksforgeeks.org/removing-stop-words-nltk-python/