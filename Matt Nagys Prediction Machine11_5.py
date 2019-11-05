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
#create a function for cleaning
def cleanStadiumType(txt):
    if pd.isna(txt): #isna for newer p
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.replace('outdoor retr roofopen', 'roof rtrd')
    txt = txt.replace('outdoor retr roofopen', 'roof rtrd')
    txt = txt.replace('retr roof open', 'roof rtrd')
    txt = txt.replace('indoor open roof', 'roof rtrd')
    txt = txt.replace('domed open', 'roof rtrd')
    txt = txt.replace('retr roofopen', 'roof rtrd')
    txt = txt.replace('retr roof closed', 'indoor')
    txt = txt.replace('retr roofclosed', 'indoor')
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('bowl', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('open', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('dome closed', 'indoor')
    txt = txt.replace('domed closed', 'indoor')
    txt = txt.replace('indoor roof closed', 'indoor')
    txt = txt.replace('heinz field', 'outdoor')
    txt = txt.replace('cloudy', 'outdoor')
    txt = txt.replace('closed dome', 'indoor')
    txt = txt.replace('domed', 'indoor')
    return  txt
    
train['StadiumType'] = train['StadiumType'].apply(cleanStadiumType)

train['StadiumType'].value_counts()

#turn the cleaned text into numbers
def transformStadiumType(txt):
    if pd.isna(txt):
        return np.nan
    if 'outdoor' in txt or 'roof rtrd' in txt:
        return 2
    if 'indoor' in txt:
        return 1
    if 'dome' in txt or 'retractable roof' in txt:
        return 0
    return np.nan
    
train['StadiumType'] = train['StadiumType'].apply(transformStadiumType)

#standardize turf and grass names   
Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 

#map the names
train['Turf'] = train['Turf'].map(Turf)
#Natural = True Artificial = False
train['Turf'] = train['Turf'] == 'Natural'

#standardize Field position and possesion
team_abbr = {'ARI' : 'ARZ', 'BAL' : 'BLT', 'CLE' : 'CLV', 'HOU' : 'HST'}

for abb in train['PossessionTeam'].unique():
    team_abbr[abb] = abb
    
train['PossessionTeam'] = train['PossessionTeam'].map(team_abbr)
train['HomeTeamAbbr'] = train['HomeTeamAbbr'].map(team_abbr)
train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].map(team_abbr)
train['FieldPosition'] = train['FieldPosition'].map(team_abbr)

