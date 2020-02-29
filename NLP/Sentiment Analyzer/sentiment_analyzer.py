##Import libs
import nltk
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# Data - http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
# positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="html5lib")
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="html.parser")
positive_reviews = positive_reviews.findAll('review_text')

# negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html5lib")
negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html.parser")
negative_reviews = negative_reviews.findAll('review_text')

##Functions
#Our tokenizer
def my_tokenizer(s):
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) > 2]
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	tokens = [t for t in tokens if t not in stopwords]
	return tokens

#Tokens to vector - terms count
def tokens_to_vector(tokens, label):
	x = np.zeros(len(word_index_map) + 1)
	for t in tokens:
		i = word_index_map[t]
		x[i] += 1
	x = x / x.sum()
	x[-1] = label
	return x

##Preprocessing and model
#Create word frequency vector
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

#iterate over positive reviews
for review in positive_reviews:
	orig_reviews.append(review.text)
	tokens = my_tokenizer(review.text)
	positive_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1

#iterate over negative reviews
for review in negative_reviews:
	orig_reviews.append(review.text)
	tokens = my_tokenizer(review.text)
	negative_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1

print('Length of word_index_map:', len(word_index_map))

#Create N x D+1 matrix
N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
	xy = tokens_to_vector(tokens, 1)
	data[i,:] = xy
	i += 1

for tokens in negative_tokenized:
	xy = tokens_to_vector(tokens, 0)
	data[i,:] = xy
	i += 1

#Shuffle data, create train/test splits
orig_reviews, data = shuffle(orig_reviews, data)

X = data[:,:-1]
Y = data[:,-1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

#Train model, check accuracy
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print('Train accuracy:', model.score(Xtrain, Ytrain))
print('Test accuracy:', model.score(Xtest, Ytest))







##

