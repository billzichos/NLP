import json
import nltk
from nltk.probability import FreqDist
import re

print('Begin loading training data...')

filepath = 'C:\\Users\\Bill\\Documents\\GitHub\\Project-Files\\Kaggle - Random Acts of Pizza Train.json'

json_data = open(filepath)

data = json.load(json_data)

print('...Complete - Training data has been loaded successfully.')

def bzSentenceCount(text):
        return len(nltk.sent_tokenize(text))

def bzWordCount(text):
        return len(nltk.word_tokenize(text))

def bzLexicalDiversity(text):
        return len(set([words.lower() for words in nltk.word_tokenize(text)])) / len(nltk.word_tokenize(text))

#def bzHasPictureFlag

#[item['request_id'] for item in data if re.search('.jpg|.png', item['request_text'])]

##def bzPictureSearch(text):
##        if 1=1:
##                return "Yes"
##        else:
##                return "No"


## Let's capture some attributes about each request.
print('Begin adding token-based features...')

for item in data:
	if len(item['request_text'])>0:
		item['SentCount']=bzSentenceCount(item['request_text'])
		item['WordCount']=bzWordCount(item['request_text'])
		item['LexicalDiversity']=bzLexicalDiversity(item['request_text'])
		item['Picture']=item['request_text'].find('.jpg')

picList = [item['request_id'] for item in data if re.search('.jpg|.png', item['request_text'])]

for item in data:
	if item['request_id'] in picList:
		item['Picture']=1
	else:
		item['Picture']=0


# [item['request_text'].find('.jpg') for item in data]

print('...Token-based features (word count, sentence count and lexical diversity) have been added to the dataset.')


## Let's explore differences b/w successful requests and unsuccessful.
print('Begin calculating conversion rate...')

goodRequests = [item for item in data if item['requester_received_pizza']==1]
badRequests = [item for item in data if item['requester_received_pizza']==0]

print('...' + str((len([item for item in data if item['requester_received_pizza']==1]) / len([item for item in data])) * 100) + '% of requests are fullfilled.') 


## Is there anything interesting about word counts?
print('Begin calculating average word counts...')

goodAvgWordCnt = sum([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1]) / len([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1])
print('...Average word count for converted requests: ' + str(goodAvgWordCnt))
badAvgWordCnt = sum([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0]) / len([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0])
print('...Average word count for unconverted requests: ' + str(badAvgWordCnt))
## Interesting, it appears the more lengthy writers had a better conversion rate.
## Average word count for for converted requests is @110 while failed attempts had on average 86.


## Do we see similar behavior with sentence counts?
print('Begin calculating average sentence counts...')

goodAvgSentCnt = sum([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1]) / len([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1])
print('...Average sentence count for converted requests: ' + str(goodAvgSentCnt))
badAvgSentCnt = sum([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0]) / len([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0])
print('...Average sentence count for unconverted requests: ' + str(badAvgSentCnt))
## Sentence counts are pretty similar to the word counts. 6 to 4.8 sentences for converted and unconverted requests respectively.


## How about Lexical diversity?
print('Begin calculating lexical diversity...')

goodAvgLexDty = sum([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1]) / len([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1])
print('...Average lexical diversity for converted requests: ' + str(goodAvgLexDty))
badAvgLexDty = sum([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0]) / len([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0])
print('...Average lexical diversity for unconverted requests: ' + str(badAvgLexDty))
## Converted lexical diversity = 74%
## Unconverted lexical diversity = 77%
