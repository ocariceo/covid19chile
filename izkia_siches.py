# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:20:48 2020

@author: ocariceo
"""


#importar los paquetes necesarios para hacer la búsqueda
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Establecer las credenciales para trabajar con la API de Twitter
consumerKey = "xxxxxxx"
consumerSecret = "xxxxxx"
accessToken = "xxxxxxx"
accessTokenSecret = "xxxxxx"

# Crear el objeto de autentificación
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    
# Configurar el acceso con las credenciales
authenticate.set_access_token(accessToken, accessTokenSecret) 
    
api = tweepy.API(authenticate, wait_on_rate_limit = True)

posts = api.user_timeline(screen_name="izkia", count = 1000, lang ="es", tweet_mode="extended")

# Mostrar los 5 tweets más recientes
print("Tweets más recientes:\n")
i=1
for tweet in posts[:5]:
    print(str(i) +') '+ tweet.full_text + '\n')
    i= i+1

# Crear una base de datos con una columna que incluya los tweets
izdf = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

izdf.head()


# Remover puntuación del texto de cada tweet
izdf['textcl'] = izdf['Tweets'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convertir palabras de mayúsculas a minúsculas
izdf['textcl'] = izdf['Tweets'].map(lambda x: x.lower())
izdf["textcl"] = izdf["textcl"].map(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))
izdf["textcl"] = izdf["textcl"].map(lambda x: re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", x))
izdf["textcl"] = izdf["textcl"].map(lambda x: re.sub('RT|cc', '', x))  # remover RT y cc
izdf["textcl"] = izdf["textcl"].map(lambda x: re.sub('#\S+', '', x))  # remover hashtags
izdf["textcl"] = izdf["textcl"].map(lambda x: re.sub('@\S+', '', x))  # remover menciones
izdf["textcl"] = izdf["textcl"].map(lambda x: re.sub('rt', '', x))  # remove menciones
izdf["textcl"] = izdf["textcl"].map(lambda x: re.sub("covid|pandemia|covid19|19", '', x)) 

spanish_stopwords = stopwords.words('spanish')
  
izdf["textcl"] = izdf["textcl"].apply(lambda x: ' '.join([word for word in x.split() if word not in (spanish_stopwords)]))


long_string = ','.join(list(izdf["textcl"].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=30, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

allWords = ' '.join([twts for twts in izdf['textcl']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='Términos más usados por Izkia Siches en Twitter')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.show()
    
count_vectorizer = CountVectorizer(stop_words=spanish_stopwords)
count_data = count_vectorizer.fit_transform(izdf['textcl'])
# Visualizar las 10 palabras más comunes
plot_10_most_common_words(count_data, count_vectorizer)

# Crear la funciín para obtener la subjetividad de los tweets
def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Crear la function para obtener la polaridad
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity


# Creaar dos columnas que desplieguen la subjectividad y la polaridad en la base de datos
izdf['Subjectividad'] = izdf['Tweets'].apply(getSubjectivity)
izdf['Polaridad'] = izdf['Tweets'].apply(getPolarity)




# Crear una función para calcular los valores negativos (-1), neutrales (0) y positivos (+1)
def getAnalysis(score):
    if score < 0:
        return 'Negativo'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positivo'
izdf['Analisis'] = izdf['Polaridad'].apply(getAnalysis)
# Show the dataframe



# Mostrar tweets positivos 
print('Tweets positivos: \n')
j=1
sortedDF = izdf.sort_values(by=['Polaridad']) #Sort the tweets
for i in range(0, sortedDF.shape[0] ):
  if( sortedDF['Analisis'][i] == 'Positivo'):
    print(str(j) + ') '+ sortedDF['Tweets'][i])
    print()
    j= j+1


# Mostrar tweets negativos   
print('Tweets negativos:\n')
j=1
sortedDF = izdf.sort_values(by=['Polaridad'],ascending=False) #Sort the tweets
for i in range(0, sortedDF.shape[0] ):
  if( sortedDF['Analisis'][i] == 'Negativo'):
    print(str(j) + ') '+sortedDF['Tweets'][i])
    print()
    j=j+1
    


# Grafico polaridad vs subjetividad
plt.figure(figsize=(8,6)) 
for i in range(0, izdf.shape[0]):
  plt.scatter(izdf["Polaridad"][i], izdf["Subjectividad"][i], color='Blue') 
# plt.scatter(x,y,color)   
plt.title('Análisis de sentimiento') 
plt.xlabel('Polaridad') 
plt.ylabel('Subjectividad') 
plt.show()

# Desplegar el porcentaje de tweets positivos
ptweets = izdf[izdf.Analisis == 'Positivo']
ptweets = ptweets['Tweets']
ptweets

round( (ptweets.shape[0] / izdf.shape[0]) * 100 , 1)

# Desplegar el porcentaje de tweets negativos
ntweets = izdf[izdf.Analisis == 'Negativo']
ntweets = ntweets['Tweets']
ntweets

round( (ntweets.shape[0] / izdf.shape[0]) * 100, 1)

# desplegar valores
izdf['Analisis'].value_counts()

# Visualizar los valores
plt.title('Análisis de sentimento')
plt.xlabel('Sentimento')
plt.ylabel('Frecuencia')
izdf['Analisis'].value_counts().plot(kind = 'bar')
plt.show()
