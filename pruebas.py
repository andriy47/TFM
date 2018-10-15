
import numpy as np
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools 
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
'%matplotlib inline'

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

 
import json
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt 
import gensim

nltk.download('stopwords')
from string import punctuation
from nltk.corpus import stopwords
from datetime import datetime
from textblob import TextBlob 
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from nltk.tokenize import TweetTokenizer


redexUrl = re.compile(r'^https?:\/\/.*[\r\n]*')

english_stopwords = stopwords.words('english')

data = []
#Variables para sentimentalAnalis()
popularidad_list = []
numeros_list = []
numero = 1 

def tweet_clean(tweet):
    print('Original tweet:', tweet, '\n')
    # Remove HTML special entities (e.g. &amp;)
    tweet_no_special_entities = re.sub(r'\&\w*;', '', tweet)
    print('No special entitites:', tweet_no_special_entities, '\n')
    # Remove tickers
    tweet_no_tickers = re.sub(r'\$\w*', '', tweet_no_special_entities)
    print('No tickers:', tweet_no_tickers, '\n')
    # Remove hyperlinks
    tweet_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', tweet_no_tickers)
    print('No hyperlinks:', tweet_no_hyperlinks, '\n')
    # Remove hashtags
    tweet_no_hashtags = re.sub(r'#\w*', '', tweet_no_hyperlinks)
    print('No hashtags:', tweet_no_hashtags, '\n')
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet_no_punctuation = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet_no_hashtags)
    print('No punctuation:', tweet_no_punctuation, '\n')
    # Remove https
    tweet_no_https = re.sub(r'https', '', tweet_no_punctuation)
    tweet_no_https = re.sub(r'http', '', tweet_no_punctuation)
    print('No https:', tweet_no_https, '\n')
    # Remove words with 2 or fewer letters
    tweet_no_small_words = re.sub(r'\b\w{1,2}\b', '', tweet_no_https)
    print('No small words:', tweet_no_small_words, '\n')
    # Remove whitespace (including new line characters)
    tweet_no_whitespace = re.sub(r'\s\s+', ' ', tweet_no_small_words) 
    tweet_no_whitespace = tweet_no_whitespace.lstrip(' ') # Remove single space remaining at the front of the tweet.
    print('No whitespace:', tweet_no_whitespace, '\n')
	# Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet_no_emojis = ''.join(c for c in tweet_no_whitespace if c <= '\uFFFF') # Apart from emojis (plane 1), this also removes historic scripts and mathematical alphanumerics (also plane 1), ideographs (plane 2) and more.
    print('No emojis:', tweet_no_emojis, '\n')
    # Tokenize: Change to lowercase, reduce length and remove handles
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True) # reduce_len changes, for example, waaaaaayyyy to waaayyy.
    tw_list = tknzr.tokenize(tweet_no_emojis)
    print('Tweet tokenize:', tw_list, '\n')
    # Remove stopwords
    list_no_stopwords = [i for i in tw_list if i not in english_stopwords]
    print('No stop words:', list_no_stopwords, '\n')
    # Final filtered tweet
    tweet_filtered =' '.join(list_no_stopwords)
    print('Final tweet:', tweet_filtered)
    return(tweet_filtered)


def clean_data_from_json(file):
    # Load the first sheet of the JSON file into a data frame
    df = pd.read_json(file, orient='columns')
    data = df['text'].tolist()
    l = []
    for tweet in data:
        s = tweet_clean(tweet)
        print(s)
        l.append(s)
    return(l)

data = clean_data_from_json('../P1THEDEMOCRATS_2000.json')

# def start(documento):
#     global data
#     for cadaTwit in stateDict:
#         for key, val in cadaTwit.iteritems(): 
#             if 'text' in key:
#                 clear_twit = tweet_clean(val)
#                 data.append(clear_twit)
#                 # sentimentalAnalis(clear_twit)

# Start 
# print "Sentimental Analysis Start:"
# start(stateDict)
print "\n"
print "Sentimental Analysis DONE:"


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])

#Dibujar grafica 
# GraficarDatos(numeros_list,popularidad_list,numero) 

# LDA
# print "LDA START:"
# ldaMethod(data)
# print "LDA DONE:"









#Twits de partidos politicos hashtag
#Plataforma de politicos por que votan o que proponen que   CongrecionalVoce Database GobTrack

#Extraccion de topics twit
#Agupar texto para tener cantidas 

#Fijarse en las fechas de los 2000 Twits 

#TextBlob
#Spelling Correction
