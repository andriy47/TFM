
import json
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt 
import gensim

# nltk.download('stopwords')
from nltk.corpus import stopwords
from datetime import datetime
from textblob import TextBlob 
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

redexUrl = re.compile(r'^https?:\/\/.*[\r\n]*')

stateDict = json.load(open('../P1THEDEMOCRATS_2000.json'))

popularidad_list = []
numeros_list = []
numero = 1 


def sentimentalAnalis(twitText):
    global numero, popularidad_list, numeros_list
    analisis = TextBlob(twitText)
    analisis = analisis.sentiment
    popularidad = analisis.polarity
    popularidad_list.append(popularidad)
    numeros_list.append(numero)
    numero = numero + 1 


def GraficarDatos(numeros_list,popularidad_list,numero):
    axes = plt.gca()
    axes.set_ylim([-1, 2]) 
    
    plt.scatter(numeros_list, popularidad_list)

    popularidadPromedio = (sum(popularidad_list))/(len(popularidad_list))
    popularidadPromedio = "{0:.0f}%".format(popularidadPromedio * 100)
    time  = datetime.now().strftime("A : %H:%M\n El: %m-%d-%y")
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


data = dict()
idTwit = ''
for cadaTwit in stateDict:
    for key, val in cadaTwit.iteritems():
        #ID of TWIT
        if '_id' in key:
            idTwit = val
        #Remove StopWord
        if 'text' in key:
            twit = [word for word in val.encode("utf-8").strip().split()]
            twit = [w.replace('@', '') for w in twit]
            twit = [w.replace('#', '') for w in twit]
            sentimentalAnalis(val)
            clean_tokens = twit[:]
            for palabra in twit:
                matchUrl = redexUrl.match(palabra)
                try:
                    # STOP_WORDS
                    if unicode(palabra) in stopwords.words('english'):
                        clean_tokens.remove(palabra)
                    if matchUrl:
                        clean_tokens.remove(palabra)
                except UnicodeDecodeError:
                    clean_tokens.remove(palabra) 
            if data.get(idTwit):
                var = data.get(idTwit)
                data[idTwit] = var.append(clean_tokens)
            else :
                data[idTwit] = clean_tokens

# GraficarDatos(numeros_list,popularidad_list,numero) 

datDf = {key:pd.Series(val) for key,val in data.iteritems()}
# Panda
merged = pd.DataFrame(datDf)

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
     
# compile sample documents into a list
doc_set = [' '.join(val) for key,val in data.iteritems()]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

 
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)

print ldamodel.print_topics(2)







#Twits de partidos politicos hashtag
#Plataforma de politicos por que votan o que proponen que   CongrecionalVoce Database GobTrack

#Extraccion de topics twit
#Agupar texto para tener cantidas 

#Fijarse en las fechas de los 2000 Twits 

#TextBlob
#Spelling Correction
