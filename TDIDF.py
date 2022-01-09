import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
sentense = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentense = nltk.sent_tokenize(sentense)

corpus = []

for i in range(len(sentense)):
    review = re.sub('[^a-zA-Z]',' ',sentense[i])
    review = review.lower()
    review = review.split()
    review = [WordNetLemmatizer().lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = TfidfVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()

print(pd.DataFrame(x))