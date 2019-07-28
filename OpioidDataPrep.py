import re
import pandas as pd
import numpy as np

#Function to sort a data frame alphabetically by column name
def sortDFbyColName(dfIn):
    c = list(dfIn.columns)
    c.sort()
    return dfIn[c]

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
def prepScaleAndCenter(dfIn, colIn, meanIn, stdIn):
    '''Quick function to scale and center (non-binary) numerical data'''
    dfIn[colIn] = dfIn[colIn] - meanIn
    dfIn[colIn] = dfIn[colIn] / stdIn
    
    #These were the original, but switched to hard code for predicitons
    #dfIn[colIn] = dfIn[colIn] - np.mean(dfIn[colIn])
    #dfIn[colIn] = dfIn[colIn] / np.std(dfIn[colIn])
    
#Function to recode a variable
def prepRecode(dfIn, colIn, recodeDict):
    dfIn[colIn] = dfIn[colIn].map(recodeDict)
    
def prepOneHot(dfIn, colIn, hotCols=None, dropOrigCol=True, trainOrPred='train'):
    if trainOrPred=='train':
        dfIn=pd.concat([dfIn,pd.get_dummies(dfIn[colIn],prefix=colIn,drop_first=True)],axis=1)
        #drop_first=True to prevent multicollinearity
        if dropOrigCol:
            dfIn.drop([colIn], axis=1, inplace=True)
    else:
        pass
        #Add columns 2-n (hotCols) to df, set=0
        #Change
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
    
######
def preprocess(dfIn):

    #Drop fields used in report output but not in prediction
    if 'NAME' in dfIn.columns:
        dfIn.drop('NAME', axis=1, inplace=True)

    #Gender
    prepSubtractOne(dfIn, 'IRSEX')

    #Tobacco / Smoking
    cutPoints = [0,1,2,3,4,9]
    dfIn = prepBin(dfIn, 'IRCIGRC', cutPoints)

    cutPoints = [0,10,13,15,17,18,19,20,22,25,30,40,50,99,985,991,994,997,998,999]
    dfIn = prepBin(dfIn, 'CIGAGE', cutPoints)  #Age when smoked daily (also catches smoked/never smoked)
    '''Codes:
    2-99 = Age
    985 = Bad Data
    991 = Never Used Cigarettes
    994 = Don't Know
    997 = Refused
    998 = Blank (no answer)
    999 = Never smoked daily
    '''
    
    prepSubtractOne(dfIn, 'FUCIG18')
    
    #'TOBYR' #No action required, just including here for completeness
    
    #REMOVING CIGDLYMO BECAUSE ALL IT'S OPTIONS ARE MULTICOLLINEAR WITH OTHER VARIABLES
	#prepRecode(dfIn, 'CIGDLYMO', {1:1,2:2,5:1,91:91,94:94,97:97,98:98})
    #cutPoints = [0,1,2,91,94,97]
    #dfIn = prepBin(dfIn, 'CIGDLYMO', cutPoints)

    #Weed
    cutPoints = [0,1,2,3,9]
    dfIn = prepBin(dfIn, 'IRMJRC', cutPoints)

    cutPoints = [0,1,2,3,7,10,20,30,40,50,100,200,300,365,991,993]
    dfIn = prepBin(dfIn, 'IRMJFY', cutPoints)  #Days used weed in past year
    '''Codes:
    0-365 = Days
    991 = Never Used Weed
    993 = Never used in past year    
    '''

    prepSubtractOne(dfIn, 'FUMJ18')
    
    #Drugs (or Drugs + Alcohol)
    #'TXYRRECVD2' #No action required, just including here for completeness
    #'TXEVRRCVD2' #No action required, just including here for completeness
    
    #The cutPoints for the following hard drugs are all the same (no need to redefine every time)
    cutPoints = [0,1,2,3,9]
    dfIn = prepBin(dfIn, 'IRCOCRC', cutPoints)
    dfIn = prepBin(dfIn, 'IRCRKRC', cutPoints)
    dfIn = prepBin(dfIn, 'IRHERRC', cutPoints)
    dfIn = prepBin(dfIn, 'IRHALLUCREC', cutPoints)    
    dfIn = prepBin(dfIn, 'IRLSDRC', cutPoints)    
    dfIn = prepBin(dfIn, 'IRECSTMOREC', cutPoints)    
    dfIn = prepBin(dfIn, 'IRINHALREC', cutPoints)    
    dfIn = prepBin(dfIn, 'IRMETHAMREC', cutPoints)    
    
    #Alcohol
    cutPoints = [0,1,2,3,9]
    dfIn = prepBin(dfIn, 'IRALCRC', cutPoints)

    cutPoints = [1,11,49,99,299,365,991,993]
    dfIn = prepBin(dfIn, 'IRALCFY', cutPoints)  #Days used in past year
    '''Codes:
    0-365 = Days
    991 = Never Used Alc
    993 = Never used in past year
    '''

    prepRecode(dfIn, 'CABINGEVR', {1:1,2:2,3:1,5:1,85:85,91:91,94:94,97:97,98:98})
    cutPoints = [0,1,2,85,91,94,97,98]
    dfIn = prepBin(dfIn, 'CABINGEVR', cutPoints)
    
    cutPoints = [0, 14, 17, 20, 21, 100, 991]
    dfIn = prepBin(dfIn, 'IRALCAGE', cutPoints)  #First age used alcohol
    '''Codes:
    1-78 = Years
    991 = Never Used Alc
    '''

    #Depression
    prepRecode(dfIn, 'ADDPREV', {1:1,2:2,85:85,94:94,97:97,98:98,99:2})
    cutPoints = [0,1,2,85,94,97,98]
    dfIn = prepBin(dfIn, 'ADDPREV', cutPoints)
    
    #dfIn = prepOneHot(dfIn, 'ADDSCEV')
    cutPoints = [0,1,2,94,97,98,99]
    dfIn = prepBin(dfIn, 'ADDSCEV', cutPoints)

    #Educaiton (one hot)
    #Replaced IREDUHIGHST2 with EDUHIGHCAT
    #IREDUHIGHST2 = {1:5.0, 2:6.0, 3:7.0, 4:8.0, 5:9.0, 6:10.0, 7:11.0, 8:12.0, 9:14.0, 10:15.0, 11:16.0}
    #prepRecode(dfIn, 'IREDUHIGHST2', IREDUHIGHST2)
    #prepScaleAndCenter(dfIn, 'IREDUHIGHST2',
        #meanIn=12.759511226057949, stdIn=2.6718811981473394)
        #Need to remove hardcode
        
    cutPoints = [0,1,2,3,4,5]
    dfIn = prepBin(dfIn, 'EDUHIGHCAT', cutPoints)

    #Other
    prepRecode(dfIn, 'BOOKED', {1:1,2:2,3:1,85:85,94:94,97:97,98:98})
    #dfIn = prepOneHot(dfIn, 'BOOKED')
    cutPoints = [0,1,2,94,97]  #BOOKED HAS NO DATA FOR 3 (because we recoded it as 1), 85, or 98.
    dfIn = prepBin(dfIn, 'BOOKED', cutPoints)

    #Age
    AGE2 = {1:12.0, 2:13.0, 3:14.0, 4:15.0, 5:16.0, 6:17.0, 7:18.0, 8:19.0, 9:20.0, 10:21.0,
            11:np.mean([22,23]), 12:np.mean([24,25]), 13:np.mean([26,29]), 14:np.mean([30,34]),
            15:np.mean([35,49]), 16:np.mean([50,64]), 17:70.0
           }
    '''Note, category 17 is age 65+, so the value of 70 is somewhat arbitrary but reasonable.
    Moreover, there are relatively few respondents of this age, making the choice minimally 
    impactful.'''
    prepRecode(dfIn, 'AGE2', AGE2)
    prepScaleAndCenter(dfIn, 'AGE2', 
        meanIn=34.442670265002434, stdIn=16.35478216502389)
        #Need to remove hardcode
        
    # Drop columns with almost no responses (<10)
    colsToDrop = ['ADDPREV__GT2LTET85', 'CABINGEVR__GT2LTET85',
        'CABINGEVR__GT97LTET98', 'CIGAGE__GT997LTET998',
        'CIGAGE__GT99LTET985']
        
    #Also drop columns with identical or nearly identical questions (keeping 1)
    #Never Drank Alcohol (keeping IRALCFY)
    colsToDrop.extend(['CABINGEVR__GT85LTET91', 'IRALCFY__GT365LTET991', 
        'IRALCAGE__GT100LTET991','IRALCFY__GT991LTET993'])
    
    #Never Used Marijuana (keeping IRMJRC)
    #Note, 991 is 'used ever', and 993 is 'used in past year'. They're both duplicates
    #of other options, but not of each other.
    colsToDrop.extend(['IRMJFY__GT991LTET993','IRMJFY__GT365LTET991'])
    
    #Never Used Tobacco
    colsToDrop.extend(['CIGAGE__GT985LTET991'])
    #colsToDrop.extend(['CIGDLYMO__GT2LTET91','CIGDLYMO__GT1LTET2'])
        
    #Mental Health
    colsToDrop.extend(['ADDPREV__GT97LTET98'])
        
    #Perform the actual drop
    for col in colsToDrop:
        dfIn.drop([col], axis=1, inplace=True)
        
    return dfIn