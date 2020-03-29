import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

np.random.seed(500)
text = " The best soundtrack ever to anything.: I'm reading a lot of reviews saying that this is the best 'game soundtrack' and I figured that I'd write a review to disagree a bit. This in my opinino is Yasunori Mitsuda's ultimate masterpiece. The music is timeless and I'm been listening to it for years now and its beauty simply refuses to fade.The price tag on this is pretty staggering I must say, but if you are going to buy any cd for this much money, this is the only one that I feel would be worth every penny."
# Step - a : Remove blank rows if any.
text = os.linesep.join([s for s in text.splitlines() if s])
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
text = text.lower()
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
text = word_tokenize(text)
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.# WordNetLemmatizer requires Pos tags
# to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# Declaring Empty List to store the words that follow the rules for this step
Final_words = []
# Initializing WordNetLemmatizer()
word_Lemmatized = WordNetLemmatizer()
# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
for word, tag in pos_tag(text):
    # Below condition is to check for Stop words and consider only alphabets
    if word not in stopwords.words('english') and word.isalpha():
        word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
        Final_words.append(word_Final)

# The final processed set of words for each iteration will be stored in 'text_final'
text = str(Final_words)

Tfidf_vect = joblib.load("tfidf_vectorizer.sav")
Predict_X_Tfidf = Tfidf_vect.transform([text])

# load the model from disk
loaded_model = joblib.load("finalized_model_svm.sav")
result = loaded_model.predict(Predict_X_Tfidf)
print(result)

labelEncoder = joblib.load("label_encoder.sav")
print(labelEncoder.inverse_transform(result))
