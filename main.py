# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 4 - Naive Bayes and Logistic Regression

# Classify the spam data base by using Gaussian Naive Bayes in part 1
# Classify using Linear Regressino in part 2

import sklearn.metrics as metric
from sklearn.model_selection import train_test_split
from math import pi
import numpy 
import gaussian_naive as g

#========= Data Import and Preprocessing ==========
DATA_PATH = "../../HW3/spambase.data"
DELIMITER = ","
data  = numpy.loadtxt(DATA_PATH,delimiter=DELIMITER)
label = data[:,-1]
data  = numpy.delete(data,-1,axis=1)

# Randomize data and split in half for test and train sets
traind, testd, trainl, testl = train_test_split(data, label, test_size=0.5, random_state=0)

model = g.Spam_Naive_Bayes(traind,trainl)
a = model.predict(testd)

accuracy = metric.accuracy_score(testl,a)
print("Accuracy = {0:.2%}".format(accuracy))
