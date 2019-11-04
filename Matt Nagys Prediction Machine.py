import numpy as np
import pandas as pd
import re
from string import punctuation


#import training data
train = pd.read_csv('train.csv', dtype={'WindSpeed': 'object'})

#create a new feature 
train['DefendersInTheBox_vs_Distance'] = train['DefendersInTheBox']/train['Distance']

#identify the categorical features and how may unique values are in each column
cat_features = []
for col in train.columns:
    if train[col].dtype == 'object':
        cat_features.append((col, len(train[col].unique())))
        
#preprocess stadium features
train['StadiumType'].value_counts()
#create a function for cleaning
def cleanStadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.replace('outside', 'outdoor')
    #txt = txt.replace('outdoor retr roofopen', 'roof rtrd')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('bowl', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('ourdoor', 'outdoor')
    #txt = txt.replace('open', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('dome closed', 'indoor')
    txt = txt.replace('domed closed', 'indoor')
    txt = txt.replace('indoor roof closed', 'indoor')
    txt = txt.replace('heinz field', 'outdoor')
    txt = txt.replace('cloudy', 'outdoor')
    txt = txt.replace('closed dome', 'indoor')
    txt = txt.replace('retr roof closed', 'indoor')
    txt = txt.replace('retr roofclosed', 'indoor')
    txt = txt.replace('domed open', 'roof rtrd')
    txt = txt.replace('retr roofopen', 'roof rtrd')
    txt = txt.replace('outdoor retr roofopen', 'roof rtrd')
    txt = txt.replace('retr roof open', 'roof rtrd')
    txt = txt.replace('indoor open roof', 'roof rtrd')
    return  txt
    
train['StadiumType'] = train['StadiumType'].apply(cleanStadiumType)

train['StadiumType'].value_counts()