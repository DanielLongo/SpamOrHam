import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from read_messages import read_messages

from sklearn.tree import export_graphviz
import pydot

split_ratio = .2
model = RandomForestClassifier()
labels, counts, count_vectorizer = read_messages("./spam.csv")


num_examples_train = int(labels.shape[0] * split_ratio) 
trainX, trainY =  counts[:num_examples_train], labels[:num_examples_train]
testX, testY = counts[num_examples_train:], labels[num_examples_train:]

print("trainX", np.shape(trainX))
print("trainY", np.shape(trainY))
print("testX", np.shape(testX))
print("testY", np.shape(testY))

model.fit(trainX, trainY)
score = model.score(testX, testY)
print("Score", score)

#to visualize
tree = model.estimators_[5]

export_graphviz(tree, out_file = "tree.dot")

(graph,)  = pydot.graph_from_dot_file("tree.dot")
graph.write_png("tree.png")
