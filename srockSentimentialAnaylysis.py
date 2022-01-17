import pandas as pd
df = pd.read_csv('Data.csv',encoding='ISO-8859-1')
# print(df.head(5))
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data = train.iloc[:,2:27]
data.replace('[^a-zA-Z]' , " ",regex = True,inplace = True)
list1 = [i for i in range(25)]
newIndex = [str(i) for i in list1 ]
data.columns = newIndex
data.head(5)

for index in newIndex:
    data[index] = data[index].str.lower()
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
    
counterVectoe = CountVectorizer(ngram_range=(2,2))
traindataset = counterVectoe.fit_transform(headlines)
    
randomClassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomClassifier.fit(traindataset,train['Label'])

test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = counterVectoe.transform(test_transform)
prediction = randomClassifier.predict(test_dataset)



    
    
