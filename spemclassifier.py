import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
message = pd.read_csv("smsspamcollection/SMSSpamCollection",sep='\t',
                      names = ["label","message"])
ps = PorterStemmer()

corpus = []
for i in range(0,len(message)):
    review = re.sub('[^a-zA-Z]', ' ',message['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(message["label"],drop_first=True)
# print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
spem_detect_modal = MultinomialNB().fit(X_train, y_train)

y_pred = spem_detect_modal.predict(X_test)

from sklearn.metrics import  confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)
print(confusion_m)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

