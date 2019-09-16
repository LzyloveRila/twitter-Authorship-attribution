#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

"""-----------------------------------------------------"""
# f=open('preprocessing_havestopword_part.txt')
f=open('lemmer_PosTag.txt')
Trainning_set = f.readlines()
tweets=[]
label=[]
for line in Trainning_set:
    tweets.append(line.split("\t")[1])
    label.append(line.split("\t")[0])


X_train, X_test, Y_train, Y_test = train_test_split(np.array(tweets), label, test_size=0.05, random_state=90051)
sample_split= "Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_test.shape[0])


def my_tokenize(s):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(s)
    #return nltk.word_tokenize(s)

count_vect = CountVectorizer(tokenizer=my_tokenize,lowercase=False)
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts_shape = "X_train_counts shape:",X_train_counts.shape


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


text_clf = Pipeline([('vect', CountVectorizer(tokenizer=my_tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, 
      random_state=42,max_iter=20, tol=None)),])

text_clf.fit(X_train, Y_train) 
predicted = text_clf.predict(X_test)
accuracy = np.mean(predicted == Y_test)
print(accuracy)  

# predict test data
f2=open('preprocess_lemm_test_postag.txt')
predict = []
predict = f2.readlines()
print(len(predict))

predicted = text_clf.predict(predict)
f2.close()

#output
f=open('1_6.txt','w')

for i in range(len(predicted)):
	f.write(str(i)+","+str(predicted[i]))
	f.write('\n')
f.close()		

f1 = open('record1_4.txt','w')
f1.write("Training1: Preprocess:nostemmer,postag,twitter token; Feature:countervectorizer+tfidf"+
  "Loss:hinge, max_iter:20, set_split:0.05")
# f1.write(sample_split)
# f1.write(X_train_counts_shape)
f1.write(str(accuracy))
# f1.write("predict length:",len(predict))
f1.close()

# # #save model to disk
# import pickle
# file_name = "BOW SGD1.sav"
# pickle.dump(text_clf,open(file_name,'wb'),protocol=4)


# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

