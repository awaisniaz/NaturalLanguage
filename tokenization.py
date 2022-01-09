import nltk
# nltk.download()
# nltk.download('punkt')

from nltk.stem import PorterStemmer
from nltk.corpus import  stopwords

paragraph = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."

sentence = nltk.sent_tokenize(paragraph)
print(sentence)

stemmer  = PorterStemmer()

for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words("english"))]
    sentence[i] = ' '.join(words)


print(sentence)