
import json
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt 
import gensim

# nltk.download('stopwords')
from string import punctuation
from nltk.corpus import stopwords
from datetime import datetime
from textblob import TextBlob 
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from nltk.tokenize import TweetTokenizer


redexUrl = re.compile(r'^https?:\/\/.*[\r\n]*')

stateDict = json.load(open('../P1THEDEMOCRATS_2000.json'))

english_stopwords = stopwords.words('english')

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

def sentimentalAnalis(twitText):
    global numero, popularidad_list, numeros_list
    if numero <= 10:
        analisis = TextBlob(twitText)
        popularidad = analisis.sentiment.polarity
        # popularidad = analisis.polarity
        popularidad_list.append(popularidad)
        numeros_list.append(numero)
        numero = numero + 1 


def GraficarDatos(numeros_list,popularidad_list,numero):
    axes = plt.gca()
    axes.set_ylim([-1, 2]) 
    
    plt.scatter(numeros_list, popularidad_list)

    popularidadPromedio = (sum(popularidad_list))/(len(popularidad_list))
    popularidadPromedio = "{0:.0f}%".format(popularidadPromedio * 100)
    time  = datetime.now().strftime("")
    # time  = datetime.now().strftime("A : %H:%M\n El: %m-%d-%y")
    plt.text(0, 1.25, 
             "Sentimiento promedio:  " + str(popularidadPromedio) + "\n" + time, 
             fontsize=12, 
             bbox = dict(facecolor='none', 
                         edgecolor='black', 
                         boxstyle='square, pad = 1'))
    
    plt.title("Sentimientos sobre P1THEDEMOCRATS_2000 en twitter")
    plt.xlabel("Numero de tweets")
    plt.ylabel("Sentimiento")
    plt.show()

def ldaMethod(data):
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    doc_set = [' '.join(val) for key,val in data.iteritems()]


    # list for tokenized twiits in loop
    texts = []

    for i in doc_set:

        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        
        # add tokens to list
        texts.append(stemmed_tokens)

    # twit tokenizado en documento 
    dictionary = corpora.Dictionary(texts)
        
    #token en document-termino matriz
    corpus = [dictionary.doc2bow(text) for text in texts]

    
    # Generacion LDA
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)

    print ldamodel.print_topics(5)


def start(documento):
    for cadaTwit in stateDict:
        for key, val in cadaTwit.iteritems(): 
            if 'text' in key:
                clear_twit = tweet_clean(val)
                sentimentalAnalis(clear_twit)


start(stateDict)

#Dibujar grafica
GraficarDatos(numeros_list,popularidad_list,numero) 









#Twits de partidos politicos hashtag
#Plataforma de politicos por que votan o que proponen que   CongrecionalVoce Database GobTrack

#Extraccion de topics twit
#Agupar texto para tener cantidas 

#Fijarse en las fechas de los 2000 Twits 

#TextBlob
#Spelling Correction
