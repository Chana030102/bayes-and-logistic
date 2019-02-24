# gaussian_naive.py
#
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 4 - Naive Bayes and Logistic Regression

# Class for organization and convenience to classify spam database

import sklearn.metrics as metric
from sklearn.model_selection import train_test_split
from math import pi
import numpy

EPSILON = 0.0001

class Spam_Naive_Bayes:
    def __init__(self, train_data, train_label):
        # Class probabilities
        self.p_spam = numpy.sum(train_label)/len(train_label)
        self.p_nonspam = 1 - self.p_spam

        # Find indices of spam and not spam
        i_spam = []
        i_nonspam = []

        for i in range(len(train_label)):
            if(train_label[i] == 1):
                i_spam.append(i)
            else:
                i_nonspam.append(i)

        # Save data for each class
        self.spam_data = train_data[i_spam]
        self.nonspam_data = train_data[i_nonspam]

        self.spam_mean = numpy.sum(self.spam_data,axis=0)/len(self.spam_data)
        self.nonspam_mean = numpy.sum(self.nonspam_data,axis=0)/len(self.nonspam_data)
        
        # std = sqrt(((mean-data)^2)/mean)
        std = numpy.subtract(self.spam_mean,self.spam_data)
        std = numpy.square(std)
        std = numpy.sum(std,axis=0)
        std = numpy.sqrt(std)
        self.spam_std = numpy.add(std,EPSILON)

        std = numpy.subtract(self.nonspam_mean,self.nonspam_data)
        std = numpy.square(std)
        std = numpy.sum(std,axis=0)
        std = numpy.divide(std,self.nonspam_mean)
        std = numpy.sqrt(std)
        self.nonspam_std = numpy.add(std,EPSILON)

    # Classify provided data
    def predict(self,data):
 
        a = 1/(numpy.sqrt(2*pi)*self.spam_std)
        b = numpy.exp(-1*numpy.square(data-self.spam_mean)/(2*numpy.square(self.spam_std)))
        n_spam = numpy.multiply(a,b)
        n_spam = numpy.log(n_spam)
        self.n_spam = numpy.sum(n_spam,axis=1) + numpy.log(self.p_spam)

        a = 1/(numpy.sqrt(2*pi)*self.nonspam_std)
        b = numpy.exp(-1*numpy.square(data-self.nonspam_mean)/(2*numpy.square(self.nonspam_std)))
        n_nonspam = numpy.multiply(a,b)
        n_nonspam = numpy.log(n_nonspam)
        self.n_nonspam = numpy.sum(n_nonspam,axis=1) + numpy.log(self.p_nonspam)

        return numpy.argmax([self.n_nonspam,self.n_spam],axis=0)

