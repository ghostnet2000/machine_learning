#
# a recomendation system for academic papers.
# 

import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from spam_detector import readvocab, tokenize, getdata

def playgame(chooser, rounds, alpha, data, labs):
    score = 0
    trainset = []
    trainlabs = []
    b = 5
    clf = MultinomialNB()

    curitem = 0
    while curitem < rounds:
        choosenitem = chooser(curitem, b, trainlabs, trainset, alpha, clf, data)
        score = score + labs[choosenitem]
        trainset = trainset + [data[choosenitem]]
        trainlabs = trainlabs + [labs[choosenitem]]
        curitem += b
    return(score)

def argmax(indices, vals):
    best = max(vals)
    for i in range(len(indices)):
        if vals[i] == best:
            return (indices[i])

def probachooser(curitem, b, trainset, trainlabs, alpha, clf, data):
    if len(trainset) == alpha:
        clf = clf.fit(trainlabs, trainset)
    if len(trainset) < alpha:
        chosenitem = random.randint(curitem,curitem+b-1)
    else:
        yhat = clf.predict_proba(data[curitem:(curitem+b)])
        chosenitem = argmax(range(curitem,curitem+b), [p for (c,p) in yhat])
    return  (chosenitem)

def main():
    vocab_dict = readvocab('../data/vocab2.txt')
    (dat, labs) = getdata('../data/cb.txt', vocab_dict)

    alphas = range(10,200,5)
    print(alphas)
    ress = []
    for alpha in alphas:
        res = playgame(probachooser, 1000, alpha, dat, labs)
        ress += [res]

    plt.scatter(alphas, ress)
    plt.plot(alphas, ress)
    plt.show()

if __name__ == '__main__':
    main()

