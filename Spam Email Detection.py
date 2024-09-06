import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("TrainingData.csv") # this is some traning data that I created and stored in a csv file
print(data)

X = data['text']
Y = data['label'] # in my model a label with 1 will be marked as spam and 0 will be marked as a relevant email 

#training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

# Intialization of a vector 
vector = TfidfVectorizer()

# fit/ transformation of training data 
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.fit_transform(X_train)

# initalization of a classifier 
classifier = MultinomialNB()

# traning the classifier 
classifier.fit(X_test_vec, Y_train)

