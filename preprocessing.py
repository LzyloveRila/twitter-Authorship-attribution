import nltk
from nltk.corpus import stopwords
#import string
#from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

#resolve certificate problem
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

f=open('test_tweets_unlabeled.txt')
Trainning_set = f.readlines()
tweets=[]
label=[]
for line in Trainning_set:
	tweets.append(line)
    # tweets.append(line.split("\t")[1])
    # label.append(line.split("\t")[0])

#print(tweets)

def preprocessing(text):
	text=str(text)
	#reomve_html
	text=text.replace('{html}',"")
	#remove http links
	rem_url=re.sub(r'http\S+', '',text)
	tokenizer = TweetTokenizer()
	#tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(rem_url)
	#filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
	#stem_words=[stemmer.stem(w) for w in tokens]
	lemma_words=[lemmatizer.lemmatize(w) for w in tokens]
	return " ".join(lemma_words).replace("handle","")

f=open('preprocess_lemm_test.txt','w')
i = 0
for twitter in tqdm(tweets):
	clean_text = preprocessing(twitter)
	f.write(clean_text)
	# f.write(label[i]+"\t"+clean_text)
	i+=1
	f.write('\n')
f.close()		

 
