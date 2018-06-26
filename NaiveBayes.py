import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from read_messages import read_messages

split_ratio = .2
model = MultinomialNB()
labels, counts, count_vectorizer = read_messages("./spam.csv")
# model.fit(counts, labels)


num_examples_train = int(labels.shape[0] * split_ratio) 
trainX, trainY =  counts[:num_examples_train], labels[:num_examples_train]
testX, testY = counts[num_examples_train:], labels[num_examples_train:]

print("trainX", np.shape(trainX))
print("trainY", np.shape(trainY))
print("testX", np.shape(testX))
print("testY", np.shape(testY))

model.fit(trainX, trainY)

def predict(x, model, count_vectorizer):
	counts = count_vectorizer.transform([x])
	pred = model.predict(counts)
	return pred

def predict_examples(examples, model, count_vectorizer):
	for example in examples:
		pred = predict(example, model, count_vectorizer)
		if pred == 0:
			print("spam:", example)
		else:
			print("ham:", example)

examples = ["Free Viagra call today!", "I'm going to attend the Linux users group tomorrow.", "This model is ok"]
predict_examples(examples, model, count_vectorizer)

print(model.score(testX, testY))

