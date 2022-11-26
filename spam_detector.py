# a spam detector using two different model (Bayes and Decision tree model)
from sklearn import tree
import graphviz 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# read in the vocabulary file 
def readvocab(vocab_path="vocab.txt"):
	# keep track of the number of words
	lexiconsize = 0

	# initialize an empty dictionary
	word_dict = {}
	# create a feature for unknown words
	word_dict["@unk"] = lexiconsize
	lexiconsize += 1
	# read in the vocabular file
	with open(vocab_path, "r") as f:
		data = f.readlines()
	# Process the file a line at a time.
	for line in data:
		# The count is the first 3 characters
		count = int(line[0:4])
		# The word is the rest of the string
		token = line[5:-1]
		# Create a feature if it’s appeared at least twice
		if count > 1: 
			word_dict[token] = lexiconsize
			lexiconsize += 1
	# squirrel away the total size for later reference
	word_dict["@size"] = lexiconsize
	return(word_dict)


# Turn string str into a vector.
def tokenize(email_string, word_dict):
	# initially the vector is all zeros
	vec = [0 for i in range(word_dict["@size"])]
	# for each word
	for t in email_string.split(" "):
		# if the word has a feature, add one to the corresponding feature
		if t in word_dict: vec[word_dict[t]] += 1
		# otherwise, count it as an unk
		else: 
			vec[word_dict["@unk"]] += 1
	return(vec)

# read in labeled examples and turn the strings into vectors
def getdata(filename, word_dict):
	with open(filename, "r") as f:
		data = f.readlines()
	dat = []
	labs = []
	for line in data:
		labs = labs + [int(line[0])]
		dat = dat + [tokenize(line[2:], word_dict)]
	return(dat, labs)

import numpy as np

def plotsentence(sentence, clf, word_dict):
	acc = 1.0
	labs = []
	facs = []
	factor = np.exp(clf.class_log_prior_[0]- clf.class_log_prior_[1])
	labs += ["PRIOR"]
	facs += [factor]
	acc *= factor
	for w in sentence:
		i = word_dict[w]
		factor = np.exp(clf.feature_log_prob_[0][i]- clf.feature_log_prob_[1][i])
		labs += [w]
		facs += [factor]
	acc *= factor
	labs += ["POST"]
	facs += [acc]
	return((labs,facs))

def main():
	word_dict = readvocab('../data/vocab.txt')
	traindat, trainlabs = getdata("../data/spam-train.csv", word_dict)
	testdat, testlabs = getdata("../data/spam-test.csv", word_dict)

	clf = tree.DecisionTreeClassifier(max_leaf_nodes = 6)	
	clf = clf.fit(traindat, trainlabs)	

	yhat = clf.predict(testdat)

	wordlist = list(word_dict.keys())[:-1]
	dot_data = tree.export_graphviz(clf, feature_names=wordlist,
	                      filled=True, rounded=True) 
	graph = graphviz.Source(dot_data)	
	graph.save()
	return
	acc = sum([yhat[i] == testlabs[i] for i in range(len(testdat))])/len(testdat)

	clf = tree.DecisionTreeClassifier(max_leaf_nodes = 6)	
	clf = clf.fit(traindat, trainlabs)	

	yhat = clf.predict(testdat)

	sum([yhat[i] == testlabs[i] for i in range(len(testdat))])/len(testdat)

	clf = MultinomialNB().fit(traindat, trainlabs)
	clf = clf.fit(traindat, trainlabs)	
	yhat = clf.predict(testdat)
	acc = sum([yhat[i] == testlabs[i] for i in range(len(testdat))])/len(testdat)
	
	print(confusion_matrix(testlabs, yhat))

	#visualising 
	(labs,facs) = plotsentence(['yo', 'come', 'over', 'carlos', 'will', 'be', 'here', 'soon'], clf, word_dict)
	facs = [ fac if fac >= 1.0 else -1/fac for fac in facs ]
	v = [(l,round(f,1)) for (l,f) in zip(labs,facs)]
	

	(labs,facs) = plotsentence(['congratulations', 'thanks', 'to', 'a', 'good', 'friend', 'u', 'have', 'won'], clf, word_dict)
	facs = [ fac if fac >= 1.0 else -1/fac for fac in facs ]
	v = [(l,round(f,1)) for (l,f) in zip(labs,facs)]
	
	for x in v:
		print(x)

if __name__ == '__main__':
	main()