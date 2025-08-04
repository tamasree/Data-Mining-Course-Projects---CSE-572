#!/usr/bin/env python3
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.impute import KNNImputer
import scipy
import gc
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from train import no_meal_feature_extractor
from pickle import dump, load
import joblib

##################################################
test_df=pd.read_csv('test.csv',header=None)

# # create dataframes for features
# feature_columns=['CGM_Max_Min_Diff','Time_diff_min_max (in minutes)',
#                                    'fft_feq_1','fft_peak1','fft_feq_2','fft_peak2','gradient_1st','gradient_2nd']

# features_df=pd.DataFrame(columns=feature_columns )

#################################################################
# populating features df with meal data
feature_list=[]
for i in range(len(test_df)):
    feature_list.append(no_meal_feature_extractor(test_df.iloc[i].to_list()))

# # Initialize an index counter
# index = 0

# # Append each list as a row in the DataFrame using .loc[]
# for item in frature_list:
#     features_df.loc[index] = item
#     index += 1  # Increment the index for the next row 

####################################################
# load the model
model = load(open('model.pkl', 'rb'))
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))

pred_list=[]
for i in range(len(feature_list)): 
    scaled_iunput = scaler.transform(np.array([feature_list[i]]))
    prediction = model.predict(scaled_iunput)
    pred_list.append(int(prediction[0]))

result=np.array(pred_list).reshape(len(pred_list),1)
result_df= pd.DataFrame(result)
result_df.to_csv("Result.csv",index=False,header=False)

#######################################################
