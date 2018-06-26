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
	count_vectorizer = CountVectorizer()
	counts = count_vectorizer.fit_transform(data["text"])
	labels = data["labels"]	
	# assert(len(labels) == len(counts))
	# print("Number of examples", len(labels))
	print("params", count_vectorizer.get_params())
	print("type", type(count_vectorizer))
	print("shape", np.shape(counts))
	return labels, counts, count_vectorizer

# labels, counts = read_messages("./spam.csv")
# print("labels", np.shape(labels), type(labels))
# print("counts", np.shape(counts), type(counts))
# print(sum(counts.todense()))
