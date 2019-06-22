import re
import pandas as pd
import numpy as np

#Function to subtract 1 from values
def prepSubtractOne(dfIn, colIn, reverse=True):
    '''For variables initially coded as 1=Yes, 2=No, the reverse flag
    re-codes them such that 1=Yes, 0=No.
    '''
    if reverse:
        dfIn[colIn] = 2-dfIn[colIn]
    else:
        dfIn[colIn] = dfIn[colIn]-1
		
#Function to scale and center data
def prepScaleAndCenter(dfIn, colIn):
    '''Quick function to scale and center (non-binary) numerical data'''
    dfIn[colIn] = dfIn[colIn] - np.mean(dfIn[colIn])
    dfIn[colIn] = dfIn[colIn] / np.std(dfIn[colIn])
	
#Function to recode a variable
def prepRecode(dfIn, colIn, recodeDict):
    dfIn[colIn] = dfIn[colIn].map(recodeDict)
	
def prepOneHot(dfIn, colIn, dropOrigCol=True):
    dfIn = pd.concat([dfIn, pd.get_dummies(dfIn[colIn], prefix=colIn, drop_first=True)], axis=1)
    #Set drop_first=True to prevent multicollinearity
    if dropOrigCol:
        dfIn.drop([colIn], axis=1, inplace=True)
    return dfIn

def prepBin(dfIn, colIn, cutPoints, dropOrigCol=True):
    '''Function to split data into bins.
    
    First makes the bin assignments, then uses the prepOneHot to one hot encode the results.
    
    Cutpoints submitted will occur between the values in cutPoints, but not outside of them.
    For example, if cutPoints is [1, 2, 3], it will include all numbers between 1 and 2,
    including 1 and 2, and then all numbers between 2 and 3, including 3, but it will return
    NaN for values < 1 and > 3.
    
    Recall cutpoint notation: (a, b] = all real numbers between a and b, including b.    
    
    Labels are required because the default (a, b] notation is rejected by subsequent ML 
    algorithms. 'GT' refers to 'Greater than or equal to, and LTET' refers to 'Less than 
    or equal to'.
    '''
    #labels = ['LTET'+str(cutPoint) for cutPoint in cutPoints[1:]] #[1:] drops the first label
    labels = []
    for i, cutPoint in enumerate(cutPoints[:-1]):
        labels.append('GT'+str(cutPoint)+'LTET'+str(cutPoints[i+1]))
    #print(labels)
    
    dfIn[colIn+'_'] = pd.cut(dfIn[colIn], bins=cutPoints, include_lowest=True, labels=labels)
    if dropOrigCol:
        dfIn.drop([colIn], axis=1, inplace=True)
    return prepOneHot(dfIn, colIn+'_')
	
#Column finder
def colFinder(df, pattern):	    
    print(list(filter(re.compile('.*'+pattern).match, df.columns)))
