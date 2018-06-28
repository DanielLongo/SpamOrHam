import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def porter_stemmer(text, ps):
	final = []
	text = text.split(" ")
	for word in text:
		try:
			word = word_tokenize(word)
			if len(word) == 0:
				continue
			word = word[0] #remvoes extra punctuation
			word = ps.stem(word)
			final += [word]

		except Exception as e:
			print("ERROROROR", e)
			print(word)
			raise Exception("LOOOK")
	assert(len(np.shape(text)) == 1), "text ERROROROR"
	return final


	# return [ps.stem(word_tokenize(word)[-1]) for word in text]

def read_messages(filepath):
	data = pd.read_csv("./spam.csv", encoding="latin1", names=["labels", "text","","",""])
	data = data.filter(["labels", "text"]) #remove extra columns
	mapping = {"spam" : 0, "ham" : 1}
	data = data.replace({"labels":mapping})

	ps = PorterStemmer()

	for index, value in data.iterrows():
		text = value["text"]
		text = porter_stemmer(text, ps)
		data.set_value(index, "text", text)

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
