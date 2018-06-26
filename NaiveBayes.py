import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from read_messages import read_messages

model = MultinomialNB()
labels, counts, count_vectorizer = read_messages("./spam.csv")
trainX, tr
model.fit(counts, labels)



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

examples = ["Free Viagra call today!", "I'm going to attend the Linux users group tomorrow.", "This model is bad"]
predict_examples(examples, model, count_vectorizer)

print(model.score(counts, labels))

