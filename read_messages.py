import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


def read_messages(filepath):
	data = pd.read_csv("./spam.csv", encoding="latin1", names=["labels", "text","","",""])
	data = data.filter(["labels", "text"]) #remove extra columns
	mapping = {"spam" : 0, "ham" : 1}
	data = data.replace({"labels":mapping})		

	#counts the number of uses per word 
	counts = CountVectorizer().fit_transform(data["text"])
	labels = data["labels"]	
	# assert(len(labels) == len(counts))
	# print("Number of examples", len(labels))
	return labels, counts

# labels, counts = read_messages("./spam.csv")
# print("labels", np.shape(labels), type(labels))
# print("counts", np.shape(counts), type(counts))
# print(sum(counts.todense()))
# data = pd.read_csv("./spam.csv", encoding="latin1", names=["labels", "text","","",""])
# print("Pre: \n", data.head())
# data = data.filter(["labels", "text"])
# print("Examples by type: \n", data["labels"].value_counts())
# mapping = {"spam" : 0, "ham" : 1}
# data = data.replace({"labels":mapping})
# # print("Post: \n", data.head())
# # random = np.permunaatons
# # print(data)

# count_vectorizer = CountVectorizer()
# counts = count_vectorizer.fit_transform(data["text"])

# model = MultinomialNB()
# labels = data["labels"]
# model.fit(counts, labels)

# def predict(x, model):
# 	counts = count_vectorizer.transform([x])
# 	pred = model.predict(counts)
# 	return pred

# examples = ["Free Viagra call today!", "I'm going to attend the Linux users group tomorrow.", "This model is bad"]
# for example in examples:
# 	pred = predict(example, model)
# 	if pred == 0:
# 		print("spam:", example)
# 	else:
# 		print("ham:", example)


# example_counts = count_vectorizer.transform(examples)
# predictions = model.predict(example_counts)
# print("Predictions", predictions)

# (5572, 8672)
# print([x for label in data["labels"] if label == "spam" x = 0 else x = 1])
# print("Pre", list(data.columns))
# data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
# data = data.rename(columns={"v1":"label", "v2":"text"})
# print(data.head())
# print("Post", list(data.columns))
# print(list(data))
# print(data.v1)
# print(data.label.value_counts())
# print(data.head())