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