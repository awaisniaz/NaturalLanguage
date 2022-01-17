import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = "from Google that allows you to explore satellite images showing the cities and landscapes of Pakistan and all of Asia in fantastic detail. It works on your desktop computer, tablet, or mobile phone. The images in many areas are detailed enough that you can see houses, vehicles and even people on a city street. Google Earth is free and easy-to-use."
text = re.sub(r'\[[0-9]"',' ',paragraph)
text = re.sub(r'\s+', ' ',text)
text = text.lower()
text = re.sub(r'\d', ' ',text)
text = re.sub(r'\s+', ' ',text)

sentenses = nltk.sent_tokenize(text)
sentenses = [nltk.word_tokenize(sen) for sen in sentenses]

for i in range(len(sentenses)):
    sentenses[i] = [word for word in sentenses[i]  if word not in stopwords.words('english')]

model1 = Word2Vec(text,min_count=1)
words = model1.wv.vocab
vector = model1.wv('explore')
print(vector)

similarWord = model1.wv.most_similar('explore')
