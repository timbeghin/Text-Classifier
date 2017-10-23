
# coding: utf-8

# In[174]:

# ================================================================================================================#
###            Multinomial Naive Bayes Language Classifier given sentences and labelled languages              ###
#================================================================================================================#
                                          # Author: Tim Beghin #

                                   ## Part 1: Importing dependencies ##

import pandas as pd    # Used to import csv data & clean data
import numpy as np     # Used to handle arrays & compare arrays
import string          # Used to remove punctuation from sentences 
from sklearn.model_selection import train_test_split   # Used to split the training & test data while ... 
                                                       # ... maintaining ratios of languages to original data
from sklearn.naive_bayes import MultinomialNB          # Used to call the Multinomial Naive Bayes algorithm



                                   ## Part 2: Data Import and cleaning ##
    
# >> import the csv file    
path = 'C:\\Users\\DELL\\Downloads\\lang_data.csv' 
DATA = pd.read_csv(path)

#  >>drop NaN
CLEANDATA = DATA.dropna()
# imported data now contains no empty entries

#  >>drop duplicates
CLEANDATA = CLEANDATA.drop_duplicates(subset='text', keep='first')
# imported data now contains no duplicates

#  >>remove punctuation
CLEANDATA.ix[:, 0] = [i.translate(string.punctuation) for i in CLEANDATA.ix[:, 0].values]
# string.punctuation removes characters considered punctuation
# imported data now contains no punctuation


                                 ## Part 3: Create training and tests sets ##

#  >> split into train and test sets
X_sent_train, X_sent_test, y_train, y_test = train_test_split(CLEANDATA.ix[:, 0], CLEANDATA.ix[:, 1], test_size=0.2)

# train_test_split function is useful as it maintains ratio of languages from clean data to training and tests data... 
# ...i.e - if 2/3 of the sentences are in English in the clean data, then 2/3 of the training and testing data will be English

# X_sent_train: Array of sentences for training
# X_sent_test: Array of sentences for testing
# y_train: Array of languages corresponding to the training sentences
# y_test: Array of languages corresponding to the testing sentences
# test_size is the ratio of samples to be kept as testing data. 0.2 divides the total data 80/20 testing/training

#  >>split sentences into words
WORDS = [i.split() for i in X_sent_train]
# WORDS now list containing each sentence of training data as list of strings

#  >>flatten list
WORDS = [item for sublist in WORDS for item in sublist]
# WORDS now list of all words in training data, each word as its own row

#  >>obtain unique words
UNIQUEWORDS = list(set(WORDS))
# UNIQUEWORDS is list of unique words from total list of words in training data


                                    ##  Part 4: Train Multinomial Naive Bayes Model ##

#  >>preallocate training and test boolean arrays
X_train = np.zeros((len(X_sent_train), len(UNIQUEWORDS)), dtype=np.bool_)
# X_train is an empty numpy boolean array of same size training sentences used
X_test = np.zeros((len(X_sent_test), len(UNIQUEWORDS)), dtype=np.bool_)
# X_test is empty numpy boolean array of same size testing sentences used

# >>Create boolean vector for each sentence in training data
for count, sent in enumerate(X_sent_train):
             X_train[count, :] = np.in1d(UNIQUEWORDS, sent.split())
# the in1d() function tests whether the array of the current sentence (sent) is in the array of UNIQUEWORDS. Returns a boolean array

# X_train is boolean vector where each training sentence corresponds to a row. if the a word in the sentence is also in the...
# ... UNIQUEWORDS vector, then true is returned.

# >> Create boolean vector for each sentence in testing data
for count, sent in enumerate(X_sent_test):
             X_test[count, :] = np.in1d(UNIQUEWORDS, sent.split())             

# >>Call the Mulitnomial Naive Bayes classifier from skitik learn
clf = MultinomialNB()
# >> Train the model by giving the classifier the training data
clf.fit(X_train, y_train)

#  >>Predict the languages given the test sentences
print("Results:\n",np.column_stack((X_sent_test,clf.predict(X_test))))
# clf.predict() classifies the array of test vectors X_test. Result is vector of languages for corresponding test sentence
# X_sent_test is the test sentences used
# clf.predict(X_test) is the models predicted language for each sentence


#  print prediction skill
print ("Accuracy of Model: ",round(clf.score(X_test, y_test)*100,2), " %")
# clf.score() returns the mean accuracy of the test sentences and their languages.


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



