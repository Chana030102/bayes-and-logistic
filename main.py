
import sklearn.metrics as metric
from sklearn.model_selection import train_test_split
import numpy

EPSILON = 0.0001

#========= Data Import and Preprocessing ==========
DATA_PATH = "../HW3/spambase.data"
DELIMITER = ","
data  = numpy.loadtxt(DATA_PATH,delimiter=DELIMITER)
label = data[:,-1]
data  = numpy.delete(data,-1,axis=1)

# Randomize data and split in half for test and train sets
traind, testd, trainl, testl = train_test_split(data, label, test_size=0.5, random_state=0)

# Prior probabilities for each class from training data
p_Spam = numpy.sum(trainl)/len(trainl)
p_NotSpam = 1 - p_Spam
print("P(spam) = {}\tP(not spam) = {}".format(p_Spam,p_NotSpam))

mean = numpy.mean(traind,axis=0).reshape(1,-1) # Calculate mean of each feature (columns)
std = numpy.std(traind,axis=0).reshape(1,-1)   # Calculate standard deviation of each feature

