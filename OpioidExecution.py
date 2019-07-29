'''This file is the primary file that the web app calls to make predictions
of opioid misuse risk.

INPUTS
inputDict {dictionary}: The inputs from the user selections on the web form, of
    format {<feature_name1>: feature_value1, <feature_name2>: feature_value2, etc.}
    
    WHERE:    
    feature_name {string}:  The feature name as listed in the codebook
    feature_value {int}:  The value of the feature corresponding to the choice
        in the codebook. The expectation is that values have been assigned in
        user input from the web form according to their corersponding item
        in the codebook. E.g., if the user selects 'Female' from a dropdown for 
        gender, the form returns the integer 2.
        
    INPUT EXAMPLE:
    {'NAME': 'Joe Capstone', 'IRSEX': 1, 'IREDUHIGHST2': 10, 'AGE2': 16, 'IRALCRC': 2,
    'IRALCFY': 12, 'BNGDRKMON': 1, 'HVYDRKMON': 1, 'IRALCAGE': 17, 'TXYRRECVD2': 0,
    'TXEVRRCVD2': 0, 'IRCIGRC': 9, 'CIGDLYMO': 91, 'CIGAGE': 18, 'PIPEVER': 2,
    'IRCGRRC': 4, 'IRSMKLSSREC': 4, 'IRMJRC': 1, 'MJYRTOT': 250, 'FUMJ18': 2, 'FUMJ21': 2,
    'ADDPREV': 2, 'ADDSCEV': 2, 'BOOKED': 1}

RETURNS
outputDict {dictionary}: The outputs of the report of format
    {'predProb':predProb, 'predPercentile':predPercentile, 'predFI':predFI}
    
    WHERE:
    predProb {float, between 0-1}:  The user's predicted probability of opioid misuse
    predPercentile {float, between 0-1}:  The user's percentile (how they compare
        to other users)
    predFI {dict}:  A list of features and their importance, of format
        {<feature_name1>: shapley_value1, <feature_name2>: shapley_value2, etc.}
        
        WHERE:
        feature_name {string}:  The feature name as listed in the codebook
        shapley_value {float, between 0-1}:  The contribution of this specific
            feature to the user's predicted probability of opioid misuse (predProb).
'''

import numpy as np

def generateReport(inputDict):
    
    ##### 0. Load Libraries and Set Global Variables
    
    #Import Required Libraries
    import pandas as pd
    import numpy as np
    from sklearn.externals import joblib  #Used to save/load (pickle) models
    from collections import defaultdict
    import operator
    from scipy import stats

    #Custom data prep function used in both training and prediction 
    import OpioidDataPrep as odp

    #Set initial parameter(s)
    dataDir = './data/'
    
    
    ##### 1. Preprocess
    
    #Convert inputs to list (pandas conversion to dataframe requires dict values to be lists)
    for k in inputDict:
        inputDict[k] = [inputDict[k]]
    
    #Convert dict to dataframe (prediction input expects a dataframe)
    df = pd.DataFrame.from_dict(inputDict)

    #Run preprocessing on dataframe
    df = odp.preprocess(df)
    
    #Resort by column name (necessary to feed the model)
    c = list(df.columns)
    c.sort()
    df = df[c]
    
    #Convert to numpy
    inputArr = df.values
    inputArr
    
    ##### 2. Generate Predictions
    
    #Load Models
    model = joblib.load(dataDir+'calibXGB.model')
    explainer = joblib.load(dataDir+'modelXGBCal.explainer')
    probs = np.load(dataDir+'modelXGBCalPredProbs.npy')
    
    #Load feature names (column names)
    colNamesList = joblib.load(dataDir+'colNamesList.zip')
    
    #Calculate Prediciton
    predProb = model.predict_proba(inputArr)[0][1]
    
    #Calculate Percentile
    predPercentile = stats.percentileofscore(probs, predProb)/100
    
    #Generate shapley values from this row
    shapVal = explainer.shap_values(inputArr)
    
    #Aggregate shapley values for one-hot vectors
    shapDict = defaultdict(list) #Handy: creates blank list if key doesn't exist, or appends to it if it does.

    #Get everything before the '_' character of each column name
    #Then create the column index numbers for those keys 
    #These numbers correspond to the locations in the shapley output array
    
    for i, colName in enumerate(colNamesList):
        shapDict[colName.split('_')[0]].append(i)        
        
    #Make a list of aggregated shapley values
    for k in shapDict: #Loop through every key in the dict
        shapSum = 0.0 #Initialize at 0
        for index in shapDict[k]: #Loop through every item in the key's value (a list of column indexes)
            shapSum += shapVal[1][0][index] #Add the value for each item
            #Also Old: shapSum += np.array(shapVal[0])[0][index] #Add the value for each item
            #Old version: shapVal[0][index]
            #print('index',index,' | k', k, ' | shapVal[1][0][index]', shapVal[1][0][index])
        shapDict[k] = shapSum #Replace the list with the aggregated shapley value (the sum of each individual value)        
        #print('NEXT k')

    predFI = dict(sorted(shapDict.items(), key=operator.itemgetter(1)))
    
    return predProb, predPercentile, predFI
    
def prepAge(intIn):
    '''Quick function to take an integer age as entered on the web form and emit
    the age code (integers 1 - 17)as defined in the codebook.  This function is
    used to pass the expected age code to the preprocess function.    
    '''
    
    cutPointsArr = np.array([-99,12,13,14,15,16,17,18,19,20,21,23,25,29,34,49,64,150])
    return np.argmax(cutPointsArr >= intIn)
	
def getFriendlyVarNames():
    dictFriendlyVarNames = \
	    {'IRSEX': 'Gender',
         'EDUHIGHCAT':'Highest Education',
         'AGE2':'Age',
                        
         #ALCOHOL
         'IRALCRC':'Alcohol Recency',
         'IRALCFY':'Alcohol Frequency ',
         'CABINGEVR':'Binge Drinking',
         'IRALCAGE':'Age First Used Alcohol',
 
         #DRUGS + ALCOHOL
         'TXYRRECVD2':'Alc/Drug Tx, Past Yr',
         'TXEVRRCVD2':'Alc/Drug Tx, Lifetime',
 
         #TOBACCO
         'IRCIGRC':'Cigarette Recency',
         'CIGAGE':'Daily Cigarette Use Age',
         'TOBYR':'Tobacco Use, Past Year',
         'FUCIG18':'Cigarettes Under 18',
 
         #WEED
         'IRMJRC':'Marijuana Recency',
         'IRMJFY':'Marijuana Frequency',
         'FUMJ18':'Marijuana Under 18',
 
         #Hard Drugs
         'IRCOCRC':'Cocaine Recency',
         'IRCRKRC':'Crack Recency',
         'IRHERRC':'Heroine Recency',
         'IRHALLUCREC':'Hallucinogen Recency',
         'IRLSDRC':'LSD Recency',
         'IRECSTMOREC':'Ecstacy Recency',
         'IRINHALREC':'Inhalant Recency',
         'IRMETHAMREC':'Meth Recency',
 
         #DEPRESSION
         'ADDPREV':'Multi-Day Depression',
         'ADDSCEV':'Multi-Day Discouraged',
 
         ##OTHER
         'BOOKED':'Arrested and Booked',
        }

    return dictFriendlyVarNames