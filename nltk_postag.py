import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from tqdm import tqdm

# text = nltk.word_tokenize("@ let' tri and catch up live next week !\n")
# pos_tag1 = nltk.pos_tag(text)
# # pos_tag2 = nltk.BigramTagger(text,backoff=nltk.UnigramTagger())
# print(pos_tag1)
# # print(pos_tag2)

f=open('preprocess_lemm.txt')
Trainning_set = f.readlines()
tweets=[]
label=[]
tweetsWithTag = []
for line in Trainning_set:
	# tweets.append(line)
    tweets.append(line.split("\t")[1])
    label.append(line.split("\t")[0])

tokenizer = RegexpTokenizer('\w+|\d+')
# POS_tag extraction
def POS_tag(s):
	#text = nltk.word_tokenize(s)
	text = tokenizer.tokenize(s)
	#re_punc = [word for word in text if word not in english_punctuations]
	return nltk.pos_tag(text)


#english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

for tweet in tqdm(tweets):
	PosTag = POS_tag(tweet)
	tags = []
	#print(PosTag)
	for word_tag in PosTag:
		tags.append(word_tag[1])
	tag_ = " ".join(tags)
	t = '%s%s' % (tag_,"\n")
	#t = '%s%s%s' % (tag_," ",tweet)
	tweetsWithTag.append(t)


f=open('preprocess_postag.txt','w')
i = 0
for t in tweetsWithTag:
	# f.write(t)
	f.write(label[i]+"\t"+t)
	i+=1
	# f.write('\n')
f.close()		
#print(tweetsWithTag)

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(np.array(tweetsWithTag), label, test_size=0.2, random_state=90051)
# print("Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_test.shape[0]))

# from sklearn.feature_extraction.text import CountVectorizer
# def my_tokenize(s):
# 	tknzr = TweetTokenizer()
# 	return tknzr.tokenize(s)
#     #return nltk.TweetTokenizer(s)

# count_vect = CountVectorizer(tokenizer=my_tokenize)
# X_train_counts = count_vect.fit_transform(X_train)


# # """feature set shape" and detail"""
# print(X_train_counts.shape)
# print(X_train_counts[0])
#print(count_vect.vocabulary_)  
#print(count_vect.get_feature_names())




#PosTag =  [POS_tag(s) for s in X_train]
# for i in PosTag:
# 	print(i)
# 	for word,tag in i:
# 		print(tag,end=',')
		
# from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import Pipeline
# text_clf = Pipeline([('vect', CountVectorizer(tokenizer=my_tokenize)),
#                      #('tfidf', TfidfTransformer()),
#      ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                           alpha=1e-3, random_state=42,
#                            max_iter=5, tol=None)),])
# text_clf.fit(X_train, Y_train) 
# # clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,
# #                     max_iter=5, tol=None).fit(X_train_counts,Y_train)
# #predicted = clf.predict(X_test)
# predicted = text_clf.predict(X_test)
# np.mean(predicted == Y_test)  
#print(count_vect.vocabulary_)              #nltk.help.upenn_tagset('RB')
# #建空数组ret，遍历pos_tags，把有我们需要的词性的数组保存到ret[]
# ret = []
# for word,pos in pos_tags:
#         if (pos in tags):
#             ret.append(word)
#  return ' '.join(ret)