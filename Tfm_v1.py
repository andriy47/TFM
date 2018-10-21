
import json
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt 
import gensim
from gensim import corpora, models
from string import punctuation
from nltk.corpus import stopwords
from datetime import datetime
from textblob import TextBlob 
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from nltk  import TweetTokenizer
from gensim.models import CoherenceModel 
from gensim.test.utils import common_corpus, common_dictionary

import pyLDAvis
import matplotlib.pyplot as plt
# nltk.download('stopwords')


english_stopwords = stopwords.words('english')
data = []

#Variables para sentimentalAnalis()
popularidad_list = []
numeros_list = []
numero = 1 

#Variable for LDA AND HLDA
corpus = ""
dictionary = ""

def clean_data_from_json(file):
    # Load the first sheet of the JSON file into a data frame
    df = pd.read_json(file, orient='columns')
    print "DF"
    print df
    data = df['text'].tolist()
    panda = []
    for tweet in data:
        s = tweet_clean(tweet)
        panda.append(s)
    return(panda)


def tweet_clean(tweet):
    #print('Original tweet:', tweet, '\n')
    # Remove HTML special entities (e.g. &amp;)
    tweet_no_special_entities = re.sub(r'\&\w*;', '', tweet)
    tweet_no_special_entities = re.sub(r'\@\w*;', '', tweet)
    #print('No special entitites:', tweet_no_special_entities, '\n')
    # Remove tickers
    tweet_no_tickers = re.sub(r'\$\w*', '', tweet_no_special_entities)
    #print('No tickers:', tweet_no_tickers, '\n')
    # Remove hyperlinks
    tweet_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', tweet_no_tickers)
    #print('No hyperlinks:', tweet_no_hyperlinks, '\n')
    # Remove hashtags
    tweet_no_hashtags = re.sub(r'#\w*', '', tweet_no_hyperlinks)
    #print('No hashtags:', tweet_no_hashtags, '\n')
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet_no_punctuation = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet_no_hashtags)
    #print('No punctuation:', tweet_no_punctuation, '\n')
    # Remove https
    tweet_no_https = re.sub(r'https', '', tweet_no_punctuation)
    tweet_no_https = re.sub(r'http', '', tweet_no_punctuation)
    #print('No https:', tweet_no_https, '\n')
    # Remove words with 2 or fewer letters
    tweet_no_small_words = re.sub(r'\b\w{1,2}\b', '', tweet_no_https)
    #print('No small words:', tweet_no_small_words, '\n')
    # Remove whitespace (including new line characters)
    tweet_no_whitespace = re.sub(r'\s\s+', ' ', tweet_no_small_words) 
    tweet_no_whitespace = tweet_no_whitespace.lstrip(' ') # Remove single space remaining at the front of the tweet.
    #print('No whitespace:', tweet_no_whitespace, '\n')
	# Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    # tweet_no_emojis = ''.join(c for c in tweet_no_whitespace if c <= '\uFFFF') # Apart from emojis (plane 1), this also removes historic scripts and mathematical alphanumerics (also plane 1), ideographs (plane 2) and more.
    # #print('No emojis:', tweet_no_whitespace, '\n')
    # Tokenize: Change to lowercase, reduce length and remove handles
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True) # reduce_len changes, for example, waaaaaayyyy to waaayyy.
    tw_list = tknzr.tokenize(tweet_no_whitespace)
    #print('Tweet tokenize:', tw_list, '\n')
    # Remove stopwords
    list_no_stopwords = [i for i in tw_list if i not in english_stopwords]
    #print('No stop words:', list_no_stopwords, '\n')
    # 

    # Final filtered tweet
    tweet_filtered =' '.join(list_no_stopwords)

    tweet_filtered = tweet_filtered.replace(')', '')

    tweet_filtered = tweet_filtered.encode('ascii', 'ignore')
    
    # print 'Final tweet: '+tweet_filtered 
    return(tweet_filtered)

def sentimentalAnalis(twitText):
    global numero, popularidad_list, numeros_list
    if numero <= 20:
        analisis = TextBlob(twitText)
        analisis = analisis.correct()
        popularidad = analisis.sentiment
        popularidad = popularidad.polarity
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


def ldaMethod(data,topics ):
    global corpus, dictionary
    tokenizer = RegexpTokenizer(r'\w+')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # list for tokenized twiits in loop
    palabras = []
    for i in data:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(word) for word in tokens]
        
        # add tokens to list
        palabras.append(stemmed_tokens)

    # twit tokenizado en documento 
    dictionary = corpora.Dictionary(palabras)

    #token en document-termino matriz
    corpus = [dictionary.doc2bow(text) for text in palabras]
    
    # Generacion LDA
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics, id2word = dictionary, passes=20)
    
    print "LDA"
    print ldamodel.print_topics(5)

    print "PERPLEXITY"
    print ldamodel.log_perplexity(corpus)

    coherence_model_lda = CoherenceModel(model=ldamodel, texts=palabras, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    print "COHERENCE:" 
    print coherence_lda

    # Visualize the topics
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    # vis
    

def hlda(corpus, dictionary, topic, probably_words):
    hldas = CoherenceModel(corpus, dictionary)
    topic_info = hldas.print_topics(num_topics=topic, num_words=probably_words)
    print topic_info

def start(documento):
    global data
    stateDict = clean_data_from_json(documento)
    for val in stateDict:
        clear_twit = tweet_clean(val)
        data.append(clear_twit)
          
# Start 

print "Sentimental Analysis Start:"
start('P1THEDEMOCRATS_2000.json')
#Dibujar grafica 
for tt in data:
    sentimentalAnalis(tt)
    
GraficarDatos(numeros_list,popularidad_list,numero)
print "Sentimental Analysis DONE:"
print "<------------------------------------------->"

# LDA
print "LDA & HLDA START:"
#ldaMethod(Twits , Numero de Opics)
ldaMethod(data, 5)
print "LDA + HLDA DONE:"

# print "<------------------------------------------->"
print "HLDA START:"
# def hlda(token en document-termino matriz, dictionary(seed), num_topics, most_probably_words):

# print "LALALALA"
# print dictionary
# hlda(common_corpus, common_dictionary, 3, 2)
print "HLDA DONE:"


# coherencia_modelo_lda = CoherenceModel(model=ldaMethod(data, 5), )




#Twits de partidos politicos hashtag
#Plataforma de politicos por que votan o que proponen que   CongrecionalVoce Database GobTrack

#Extraccion de topics twit
#Agupar texto para tener cantidas 

#Fijarse en las fechas de los 2000 Twits 

#TextBlob
#Spelling Correction

#Acumulador en TEXTBLOB intervalos de sentimientos
#LDA jerarquico HLDA
#hdqp
#URL extraer 
#Red follower 
#NetworkX