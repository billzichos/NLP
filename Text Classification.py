## Author: Bill Zichos

## Step 1: Combine the text for each category
## Step 2: Create distinct word lists for each category
## Step 3: Tokenize the test text.
## Step 4: Loop through each word in the tokenized text.
## Step 5: Does the token exist in either of categories?
## Step 6: Summarize the results.

#for word in text:
    # if word in cat1 then 1 else 0
    # if word in cat2 then -1 else 0
    # add them


#******************************************************
# Objective
#******************************************************
# Determine whether or not a person's free pizza request will be fullfilled.


import json
import nltk
from nltk.probability import FreqDist

print('Begin loading training data...')

filepath = 'C:\\Users\\Bill\\Documents\\GitHub\\Project-Files\\Kaggle - Random Acts of Pizza Train.json'

json_data = open(filepath)

data = json.load(json_data)

print('...Complete - Training data has been loaded successfully.')
      
cnvtText = ' '.join([item['request_text'] for item in data
                     if len(item['request_text'])>0
                     and item['requester_received_pizza']==1])

uncnvtText = ' '.join([item['request_text'] for item in data
                     if len(item['request_text'])>0
                     and item['requester_received_pizza']==0])

print('...Complete - Text Categories have been created.')

inputText = 'Hi I am in need of food for my 4 children we are a military family that has really hit hard times and we have exahusted all means of help just to be able to feed my family and make it through another night is all i ask i know our blessing is coming so whatever u can find in your heart to give is greatly appreciated'

data = nltk.word_tokenize(inputText)

for item in data:
    print(item)
    if item in cnvtText:
        print(1)
    if item in uncnvtText:
        print(-1)

#def twoFactorTextClassification(inputText, convertedText, unconvertedText):
