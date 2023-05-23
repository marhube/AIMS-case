#************** Start importing python modules
import sys # For sys.exit()
import time
import os # For
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import PySimpleGUI as sg
import scipy.cluster.hierarchy as spc
from functools import partial # For partial functions
#******* Start importing imputation tools
from scipy.interpolate import CubicSpline
#************** End importing python modules
#
def convertJSON2Frame(jsonPath):
    df = pd.read_json(jsonPath,orient='index')
    return df
#

#******* End importing imputation tools
# v needs to be a single column/vector
def find_index_missing(v,include_nonmissing = True):
    index_missing_values = [index for index in range(len(v)) if np.isnan(v[index])]
    returnValue = index_missing_values
    if(include_nonmissing):
        index_nonmissing = [index for index in range(len(v)) if index not in index_missing_values]
        returnValue = index_missing_values,index_nonmissing
    #
    return returnValue
#
# By default 'extrapolate' is set to False since the behavior of cubic splines outside of first lowest and
# and last spline can sometimes be erratic
def imputeCubicSpline(x,y,extrapolate = False):
    index_missing_values,index_nonmissing = find_index_missing(
        y,include_nonmissing=True)
    #
    if len(index_missing_values) > 0 :
        cs = CubicSpline(x[index_nonmissing],y[index_nonmissing],extrapolate = extrapolate)
        y = cs(x)
    #    
    return y
#
# If 'timeCol' is  None, then the index is assumed to be a timestamp. Oth
# This function imputes missing values by fitting a cubic spline to the non-missing values.
# If "imputationCols" is a string (single column),then it is converted to a list of length 1.
# I 'imputationCols" is None,then every column in the dataframe is imputed 
def df_impute_cubic(df,imputationCols=None,timeCols = None,crop = False,**kwargs):
    if type(imputationCols) is str:
        imputationCols = [imputationCols]
    elif imputationCols is None:
        imputationCols = list(df.columns)
    #Datetimes need to be converted to unix time
    listOfTimes = None
    if type(timeCols) is str: 
        listOfTimes = df[timeCol].tolist()
    elif timeCols is None:
        listOfTimes = list(df.index)
    #
    timeValues =  np.asarray([ts.timestamp() for ts in listOfTimes])
    #
    imputer = partial(imputeCubicSpline,timeValues,**kwargs)
    df[imputationCols] = df[imputationCols].apply(imputer)
    #
    if crop:
        df = df.dropna()
    return df
#
# Calculate distance between all pairs of columns
# The distance measure chosen here is correlation
#
#From documentation:
#* method='complete' assigns
#
#        .. math::
#           d(u, v) = \max(dist(u[i],v[j]))
#
#        for all points :math:`i` in cluster u and :math:`j` in
#        cluster :math:`v`. This is also known by the Farthest Point
#        Algorithm or Voor Hees Algorithm.
#
# df can either by a pandas data
#
# First center and normalize each column
def CCorDistance(u,v,max_lag = None):
    centered_u = u - np.mean(u) # Demean
    centered_v = v - np.mean(v)  # Demean
    cc_full  = np.correlate(a=centered_u, v=centered_v,mode = 'full')
    cc_full = cc_full/ (len(u) * np.std(u) * np.std(v)) # Normalization
    # Only negative lags are used, following the code in the function "CCorDistanc "  TSDist R-package
    # https://cran.r-project.org/web/packages/TSdist/TSdist.pdf
    #
    cc_non_positive_lags = cc_full[range(len(v)),] 
    squared_cc = [cc_non_positive_lags[ind,]**2 for ind in range(cc_non_positive_lags.shape[0])]    
    # 
    D = sqrt((1.0-squared_cc[-1]) /sum(squared_cc[:-1]))
    return D
#
def CCcondensedSet(df):
    listDists = []
    # col1 ie every column except for the last one
    # col2 ie every column except for the first
    for i in range(df.shape[1]-1):
        for j in range(i+1,df.shape[1]):
            listDists.append(CCorDistance(df.iloc[:,i].to_numpy(),df.iloc[:,j].to_numpy()))
    #
    listDists = np.asarray(listDists)
    return listDists
#
def clusterByCrossCorr(df,max_intra_dist = None):
    pdist_condensed = CCcondensedSet(df)
    # if max_intra_dist is not provided then st the max distance within the same group to the 20% quantile of all unique distances
    if max_intra_dist is None:
        max_intra_dist = np.quantile(pdist_condensed,0.2)
    #
    linkage = spc.linkage(pdist_condensed, method='complete')
    idx = spc.fcluster(linkage, max_intra_dist, 'distance')
    # Put the cluster information into a data frame. The first column is the name of each column in the input 'df'.
    # The second columns is an indicator that says which cluster group each column is assigned to by the clustering algorithm
    groupingFrame = pd.DataFrame(list(df.columns),columns=['Series'])
    groupingFrame['Group'] = list(idx)
    #
    clusterDict = {}
    #
    for group in list(set(list(idx))):
        clusterDict['_'.join(['group',str(group)])] = [
            list(df.columns)[col] for col in list(groupingFrame[groupingFrame['Group']== group].index)
            ]
        #
    #
    return linkage,clusterDict
#
def read_and_cluster(jsonPath):
    df = convertJSON2Frame(jsonPath)
    #First impute values
    df = df_impute_cubic(df,crop=True)
    linkage,clusterDict = clusterByCrossCorr(df) # Do clustering by crosscorrelation
    #    
    return     linkage,clusterDict
#
sg.theme("DarkTeal2")
layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],[sg.Button("Submit")]]
###Building Window
window = sg.Window('My File Browser', layout, size=(600,150))
#
event, values = window.read()
#
if event == "Submit":
    linkage,clusterDict = read_and_cluster(values["-IN-"])
    print(f'clusterDict is')
    print(clusterDict)
    #
    spc.dendrogram(linkage)
    plt.xlabel("Indexes of points in node ")
    plt.show()
#


