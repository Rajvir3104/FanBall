import random
import pickle
import json
import numpy as np

import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?", ".", ",", "!"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

for word in words:
    if word in ignore_letters:
        words.remove(word)

print(words)


# lemmatized_words = [
#     lemmatizer.lemmatize(word) for word in words if word not in ignore_letters
# ]

# lemmatized_words = sorted(set(words))

# print(lemmatized_words)
