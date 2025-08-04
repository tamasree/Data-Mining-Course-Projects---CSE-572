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
from pickle import dump
from sklearn.linear_model import LogisticRegression

DATE_FORMAT = '%m/%d/%Y'
DATE_FORMAT_1 = '%Y-%m-%d'
TIME_FORMAT = '%H:%M:%S'
###########################################################################################################
# Reading CSV Files
InsulinData_df = pd.read_csv("InsulinData.csv",usecols=['Date', 'Time','BWZ Carb Input (grams)'],low_memory=False)
CGMData_df= pd.read_csv("CGMData.csv",usecols=['Date', 'Time','Sensor Glucose (mg/dL)'],low_memory=False)
Insulin_patient2_df=pd.read_csv("Insulin_patient2.csv",usecols=['Date', 'Time','BWZ Carb Input (grams)'],low_memory=False)
CGM_patient2_df=pd.read_csv("CGM_patient2.csv",usecols=['Date', 'Time','Sensor Glucose (mg/dL)'],low_memory=False)

#################################################################################################################################################
# Extracting meal and no meal dataframes from first set of cgm and insulin datasets
###############################################################################################################################################


########################################################################################################
# adding Time stamp columns to dataframes
InsulinData_df['Timestamp'] = pd.to_datetime(InsulinData_df['Date'] + ' ' + InsulinData_df['Time'], format=f"{DATE_FORMAT} {TIME_FORMAT}")
InsulinData_df['Date'] = pd.to_datetime(InsulinData_df['Date'], format='%m/%d/%Y').dt.date
InsulinData_df['Time'] = pd.to_datetime(InsulinData_df['Time'], format='%H:%M:%S').dt.time

CGMData_df['Timestamp'] = pd.to_datetime(CGMData_df['Date'] + ' ' + CGMData_df['Time'], format=f"{DATE_FORMAT} {TIME_FORMAT}")
CGMData_df['Date'] = pd.to_datetime(CGMData_df['Date'], format='%m/%d/%Y').dt.date
CGMData_df['Time'] = pd.to_datetime(CGMData_df['Time'], format='%H:%M:%S').dt.time

# sort the rows based on timestamp
InsulinData_df.sort_values('Timestamp', inplace=True)
CGMData_df.sort_values('Timestamp', inplace=True)

########################################################################################################

# filter nan and 0 values  from Insulin dataframe based on meal column
mask = pd.notna(InsulinData_df['BWZ Carb Input (grams)'])
InsulinData_df_filtered = InsulinData_df[mask & (InsulinData_df['BWZ Carb Input (grams)']!=0)].reset_index(drop=True)

########################################################################################################
# Extract meal times from filtered Insuline Df if difference between two meal time is less than 2 hours

# Initialize a list to store indices to drop
to_drop = []

# Loop over consecutive rows
for i in range(1, len(InsulinData_df_filtered)):
    # Calculate time difference in minutes
    time_diff = (InsulinData_df_filtered.iloc[i]['Timestamp'] - InsulinData_df_filtered.iloc[i-1]['Timestamp']).total_seconds() / 60
    
    # If time difference is <= 120 mins, mark the earlier one for dropping
    if time_diff <= 120:
        to_drop.append(i-1)

# Drop the rows based on the indices marked
InsulinData_df_cleaned = InsulinData_df_filtered.drop(to_drop).reset_index(drop=True)

#####################################################################################################################################################
# constructing meal_df1
################################################################################################################################################


# Initialize an empty list to store the meal start times from CGM_df

CGM_Meal_start_time=[]

# ##############################################################################
# Generate column names from TM(-25) to TM(+120) with 30 total columns
meal_df_column_names = [f'TM({i})' for i in range(-25, 125, 5)][:30]
# Create an empty DataFrame with these column names
meal_df1 = pd.DataFrame(columns=meal_df_column_names)

#############################################################################
CGM_Meal_data=[]

# Iterate over each timestamp in df_main
for meal_time in InsulinData_df_cleaned['Timestamp']:
   
    # Find the first timestamp in CGMDf that is the same or greater than the meal_time
    glucose_start_df = CGMData_df[(CGMData_df['Timestamp'] >= meal_time) & (CGMData_df['Timestamp']<= (meal_time + pd.Timedelta(minutes=15)))]
    glucose_start_time= glucose_start_df['Timestamp'].min()
    CGM_Meal_start_time.append(glucose_start_time)
    # print(glucose_start_time)
    
    if pd.notna(glucose_start_time):  # if a matching or next timestamp exists
       
        ############################################################################
        count_time = glucose_start_time - pd.Timedelta(minutes=25)
        stretch_data_full=[]

        for i in range(30): 
            try:                        
                result = CGMData_df.loc[CGMData_df['Timestamp'] == count_time ,'Sensor Glucose (mg/dL)'].values[0]
                stretch_data_full.append(result)
            except:
                stretch_data_full.append(np.nan)
            count_time += pd.Timedelta(minutes=5)

        # Count NaN values in the list
        nan_count = sum(pd.isna(i) for i in stretch_data_full)

        if nan_count<5:
            CGM_Meal_data.append(stretch_data_full)

################################################################################
# Handle Missing data
for item in CGM_Meal_data:
    # # create an object for KNNImputer
        imputer = KNNImputer(n_neighbors=5)

        ## create an object for KNNImputer
        data_array = np.array(item).reshape(len(item), 1)
        new_data_array = imputer.fit_transform(data_array)
        CGM_Meal_data[CGM_Meal_data.index(item)]=new_data_array.flatten().tolist()

# print(CGM_Meal_data)
                
################################################################################
#populate meal_df
# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for row in CGM_Meal_data:
    meal_df1.loc[index] = row
    index += 1  # Increment the index for the next row 

##############################################################################################################################################     
# constructing no_meal_df1
##############################################################################################################################################

# take an empty list to store stretch data and for populating dataframe afterwards
CGM_no_meal_data_1=[]

###############################################################################
# Generate column names from TNM(+0) to TNM(+23) with 30 total columns
no_meal_df_column_names = [f'TNM({i})' for i in range(0, 120, 5)][:24]
# Create an empty DataFrame with these column names
no_meal_df1 = pd.DataFrame(columns=no_meal_df_column_names)

#################################################################################

#take an empty list to store no meal start times two hours after meal time
no_meal_start_times=[]

#Take all the no meal start time after two hours from meal start time and also check if there is any meal in that stretch
for meal_start_time in CGM_Meal_start_time:

    check_start_time = meal_start_time + pd.Timedelta(minutes=120)
    check_end_time = check_start_time + pd.Timedelta(minutes=120)

    #check if there is any meal in this time span
    check_list=[]
    for timestamp in InsulinData_df_cleaned['Timestamp']:
        if (timestamp >= check_start_time) and (timestamp<check_end_time):
            check_list.append(False)
            break
    
    if all(check_list) == True:
        no_meal_start_times.append(meal_start_time + pd.Timedelta(minutes=120))

#Getting no meal data for one stretch (upto 120 minutes from no meal start time)

for start_time in no_meal_start_times:
    if pd.notna(start_time):
        one_no_meal_stretch_data=[]
        count=0
        for i in range(24):
            try:
                result=CGMData_df.loc[CGMData_df['Timestamp'] == start_time + pd.Timedelta(minutes= count) ,'Sensor Glucose (mg/dL)'].values[0]
                one_no_meal_stretch_data.append(result)
            except:
                one_no_meal_stretch_data.append(np.nan)
            count +=5

        # Count NaN values in the list
        nan_count = sum(pd.isna(item) for item in one_no_meal_stretch_data)

        if nan_count < 5:
            CGM_no_meal_data_1.append(one_no_meal_stretch_data)

#####################################################################################
# Handle Missing data with interpolation
for item in CGM_no_meal_data_1:

    # Convert the list to a pandas Series
    data_series = pd.Series(item)

    # Fill NaN values using interpolation
    interpolated_series = data_series.interpolate(method='linear')

    # Convert back to list if needed
    interpolated_list = interpolated_series.tolist()

    CGM_no_meal_data_1[CGM_no_meal_data_1.index(item)]=interpolated_list


# print(CGM_no_meal_data_1)

################################################################################################################
# extracting no_meal_stretch data from overnight data
##################################################################################################################

#Resample on day and take the last meal of the day and make a new dataframe
new_InsulinData_df_cleaned = InsulinData_df_cleaned.set_index("Timestamp") 
day_end_meal_df = new_InsulinData_df_cleaned.resample('D').last()
day_end_meal_df.dropna(inplace=True)
day_end_meal_df['Timestamp']= pd.to_datetime(day_end_meal_df['Date'].astype(str) + ' ' + day_end_meal_df['Time'].astype(str), format=f"{DATE_FORMAT_1} {TIME_FORMAT}")
day_end_meal_df.reset_index(drop=True,inplace=True)


##################################################################################################

#calculate no meal stretch data from over night time
CGM_no_meal_data_2=[]

for i in  day_end_meal_df['Timestamp']:
    mask = CGMData_df['Timestamp']>= i
    over_night_start_time = CGMData_df[mask]['Timestamp'].min() + pd.Timedelta(minutes= 240)
    # print(over_night_start_time)

    stretch_start_time = over_night_start_time

    for i in range(5):

        check_list=[]
        for timestamp in InsulinData_df_cleaned['Timestamp']:
            if (timestamp >= stretch_start_time) and (timestamp< stretch_start_time + pd.Timedelta(minutes=120)):
                check_list.append(False)
                break
        
        if all(check_list)== True:

            one_no_meal_stretch_data=[]
            time_list=[]
            count=0
            for i in range(24):
                result_time = stretch_start_time+pd.Timedelta(minutes=count)
                time_list.append(result_time)
                count +=5

            # update next stretch start time
            stretch_start_time += pd.Timedelta(minutes=120)

            for i in time_list:
                try:
                    result = CGMData_df.loc[CGMData_df['Timestamp'] == i ,'Sensor Glucose (mg/dL)'].values[0]
                    one_no_meal_stretch_data.append(result)
                except:
                    one_no_meal_stretch_data.append(np.nan)

            # Count NaN values in the list
            nan_count = sum(pd.isna(item) for item in one_no_meal_stretch_data)

            if one_no_meal_stretch_data and nan_count<5 :
                CGM_no_meal_data_2.append(one_no_meal_stretch_data)

# Handle Missing data with interpolation
for item in CGM_no_meal_data_2:

    # Convert the list to a pandas Series
    data_series = pd.Series(item)

    # Fill NaN values using interpolation
    interpolated_series = data_series.interpolate(method='linear')

    # Convert back to list if needed
    interpolated_list = interpolated_series.tolist()

    CGM_no_meal_data_2[CGM_no_meal_data_2.index(item)]=interpolated_list   

# print(CGM_no_meal_data_2) 
############################################################################################ 

# concatenate two no meal data lists
CGM_no_meal_data=CGM_no_meal_data_1+CGM_no_meal_data_2

############################################################################################
#populate no_meal_df
# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for row in CGM_no_meal_data:
    no_meal_df1.loc[index] = row
    index += 1  # Increment the index for the next row 

no_meal_df1.dropna(inplace=True)                   


#########################################################################################################################################################################
# Extracting meal and no meal dataframes from second set of cgm and insulin datasets
##########################################################################################################################################################################

# adding time stamp values to dataframes and sort rows based on time stamp
Insulin_patient2_df['Timestamp']=pd.to_datetime(Insulin_patient2_df['Date']+ ' '+ Insulin_patient2_df['Time'])

CGM_patient2_df['Timestamp'] = pd.to_datetime(CGM_patient2_df['Date'] + ' ' + CGM_patient2_df['Time'])

Insulin_patient2_df.sort_values("Timestamp",ascending=True,inplace=True)
Insulin_patient2_df.reset_index(drop=True,inplace=True)
CGM_patient2_df.sort_values("Timestamp",ascending=True,inplace=True)
CGM_patient2_df.reset_index(drop=True,inplace=True)

###############################################################################################

# filter nan and 0 values  from Insulin dataframe based on meal column
mask = pd.notna(Insulin_patient2_df['BWZ Carb Input (grams)'])
Insulin_patient2_df_filtered = Insulin_patient2_df[mask & (Insulin_patient2_df['BWZ Carb Input (grams)']!= 0.0)].reset_index(drop=True)

###############################################################################################
# Extract meal times from filtered Insuline Df if difference between two meal time is less than 2 hours

# Initialize a list to store indices to drop
to_drop = []

# Loop over consecutive rows
for i in range(1, len(Insulin_patient2_df_filtered)):
    # Calculate time difference in minutes
    time_diff = (Insulin_patient2_df_filtered.iloc[i]['Timestamp'] - Insulin_patient2_df_filtered.iloc[i-1]['Timestamp']).total_seconds() / 60
    
    # If time difference is <= 30 mins, mark the earlier one for dropping
    if time_diff <= 120:
        to_drop.append(i-1)
        
# print(to_drop)
# Drop the rows based on the indices marked
Insulin_patient2_df_cleaned = Insulin_patient2_df_filtered.drop(to_drop).reset_index(drop=True)

#####################################################################################################################################################
# constructing meal_df2
################################################################################################################################################

# Initialize an empty list to store the filtered rows

CGM_Meal_start_time_2=[]

# ##############################################################################
# Generate column names from TM(-25) to TM(+120) with 30 total columns
meal_df_column_names = [f'TM({i})' for i in range(-25, 125, 5)][:30]
# Create an empty DataFrame with these column names
meal_df2 = pd.DataFrame(columns=meal_df_column_names)

#################################################################################

# take an empty list to store stretch data and for populating dataframe afterwards
CGM_Meal_data_2=[]

# Iterate over each timestamp in df_main
for meal_time in Insulin_patient2_df_cleaned['Timestamp']:
   
    # Find the first timestamp in CGMDf that is the same or greater than the meal_time
    glucose_start_df = CGM_patient2_df[(CGM_patient2_df['Timestamp'] >= meal_time) & (CGM_patient2_df['Timestamp']<= (meal_time + pd.Timedelta(minutes=15)))]
    glucose_start_time= glucose_start_df['Timestamp'].min()
    CGM_Meal_start_time_2.append(glucose_start_time)
    # print(glucose_start_time)
    
    if pd.notna(glucose_start_time):  # if a matching or next timestamp exists
              
        count_time = glucose_start_time - pd.Timedelta(minutes=25)

        stretch_data_full=[]

        for i in range(30): 
            try:                        
                result = CGM_patient2_df.loc[CGM_patient2_df['Timestamp'] == count_time ,'Sensor Glucose (mg/dL)'].values[0]
                stretch_data_full.append(result)
            except:
                stretch_data_full.append(np.nan)
            count_time += pd.Timedelta(minutes=5)

        # Count NaN values in the list
        nan_count = sum(pd.isna(i) for i in stretch_data_full)

        if nan_count<5:
            CGM_Meal_data_2.append(stretch_data_full)
# print(CGM_Meal_data)

################################################################################################
# Handle Missing data
for item in CGM_Meal_data_2:
    # # create an object for KNNImputer
        imputer = KNNImputer(n_neighbors=5)

        ## create an object for KNNImputer
        data_array = np.array(item).reshape(len(item), 1)
        new_data_array = imputer.fit_transform(data_array)
        CGM_Meal_data_2[CGM_Meal_data_2.index(item)]=new_data_array.flatten().tolist()

# print(CGM_Meal_data)
                
################################################################################
#populate meal_df2
# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for row in CGM_Meal_data_2:
    meal_df2.loc[index] = row
    index += 1  # Increment the index for the next row 

##############################################################################################################################################     
# constructing no_meal_df2
##############################################################################################################################################

# Construct no_meal_df
CGM_no_meal_data_3=[]
###############################################################################
# Generate column names from TNM(+0) to TNM(+23) with 30 total columns
no_meal_df_column_names = [f'TNM({i})' for i in range(0, 120, 5)][:24]
# Create an empty DataFrame with these column names
no_meal_df2 = pd.DataFrame(columns=no_meal_df_column_names)

#################################################################################

no_meal_start_times=[]

#Take all the no meal start time after two hours from meal start time and also check if there is any meal in that stretch
for meal_start_time in CGM_Meal_start_time_2:

    check_start_time = meal_start_time + pd.Timedelta(minutes=120)
    check_end_time = check_start_time + pd.Timedelta(minutes=120)

    #check if there is any meal in this time span
    check_list=[]
    for timestamp in Insulin_patient2_df_cleaned['Timestamp']:
        if (timestamp >= check_start_time) and (timestamp<check_end_time):
            check_list.append(False)
            break
    
    if all(check_list) == True:
        no_meal_start_times.append(meal_start_time + pd.Timedelta(minutes=120))

#Getting no meal data for one stretch (upto 120 minutes from no meal start time)

for start_time in no_meal_start_times:
    
    if pd.notna(start_time):
        one_no_meal_stretch_data=[]
        count=0
        for i in range(24):
            try:
                result=CGM_patient2_df.loc[CGM_patient2_df['Timestamp'] == start_time + pd.Timedelta(minutes= count) ,'Sensor Glucose (mg/dL)'].values[0]
                one_no_meal_stretch_data.append(result)
            except:
                one_no_meal_stretch_data.append(np.nan)
            count +=5

        # Count NaN values in the list
        nan_count = sum(pd.isna(item) for item in one_no_meal_stretch_data)

        if nan_count < 5:
            CGM_no_meal_data_3.append(one_no_meal_stretch_data)


# Handle Missing data with interpolation
for item in CGM_no_meal_data_3:

    # Convert the list to a pandas Series
    data_series = pd.Series(item)

    # Fill NaN values using interpolation
    interpolated_series = data_series.interpolate(method='linear')

    # Convert back to list if needed
    interpolated_list = interpolated_series.tolist()

    CGM_no_meal_data_3[CGM_no_meal_data_3.index(item)]=interpolated_list

# print(CGM_no_meal_data_3)

################################################################################################################
# extracting no_meal_stretch data from overnight data
##################################################################################################################

#Resample on day and take the last meal of the day and make a new dataframe
new_Insulin_patient2_df_cleaned = Insulin_patient2_df_cleaned.set_index("Timestamp",drop=True)
day_end_meal_df = new_Insulin_patient2_df_cleaned.resample('D').last()
day_end_meal_df.dropna(inplace=True)
day_end_meal_df['Timestamp']= pd.to_datetime(day_end_meal_df['Date'] + ' ' + day_end_meal_df['Time'])
day_end_meal_df.reset_index(drop=True,inplace=True)

################################################################################

#calculate no meal stretch data from over night time
CGM_no_meal_data_4=[]

for i in  day_end_meal_df['Timestamp']:
    mask = CGM_patient2_df['Timestamp']>= i
    over_night_start_time = CGM_patient2_df[mask]['Timestamp'].min() + pd.Timedelta(minutes= 240)
    # print(over_night_start_time)

    stretch_start_time = over_night_start_time

    for i in range(5):
        check_list=[]
        for timestamp in Insulin_patient2_df_cleaned['Timestamp']:
            if (timestamp >= stretch_start_time) and (timestamp< stretch_start_time + pd.Timedelta(minutes=120)):
                check_list.append(False)
                break

        if all(check_list)== True:

            one_no_meal_stretch_data=[]
            time_list=[]
            count=0
            for i in range(24):
                result_time = stretch_start_time+pd.Timedelta(minutes=count)
                time_list.append(result_time)
                count +=5
            # update next stretch start time
            stretch_start_time += pd.Timedelta(minutes=120)

            for i in time_list:
                try:
                    result = CGM_patient2_df.loc[CGM_patient2_df['Timestamp'] == i ,'Sensor Glucose (mg/dL)'].values[0]
                    one_no_meal_stretch_data.append(result)
                except:
                    one_no_meal_stretch_data.append(np.nan)


            # Count NaN values in the list
            nan_count = sum(pd.isna(item) for item in one_no_meal_stretch_data)

            if one_no_meal_stretch_data and nan_count<5 :
                CGM_no_meal_data_4.append(one_no_meal_stretch_data)

##################################################################################
# Handle Missing data with interpolation
for item in CGM_no_meal_data_4:

    # Convert the list to a pandas Series
    data_series = pd.Series(item)

    # Fill NaN values using interpolation
    interpolated_series = data_series.interpolate(method='linear')

    # Convert back to list if needed
    interpolated_list = interpolated_series.tolist()

    CGM_no_meal_data_4[CGM_no_meal_data_4.index(item)]=interpolated_list   
            

# print(CGM_no_meal_data_4) 

#######################################################################################

# concatenate two no meal data lists
CGM_no_meal_data=CGM_no_meal_data_3+CGM_no_meal_data_4

################################################################################
#populate no_meal_df
# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for row in CGM_no_meal_data:
    no_meal_df2.loc[index] = row
    index += 1  # Increment the index for the next row 

no_meal_df2.dropna(inplace=True)  

####################################################################################################################################
# Concatenating data frames to get final meal and no_meal dataframes
meal_df=pd.concat([meal_df1,meal_df2])
no_meal_df=pd.concat([no_meal_df1,no_meal_df2])

print(len(meal_df))
print(len(no_meal_df))
print(meal_df.head(5))
print(no_meal_df.head(5))

######################################################################################################################################
# extract features from meal and no meal df
######################################################################################################################################
#Calculate Fast Fourier Transform
def calculate_FFT(data_array):

    # Sampling frequency (example: 1 Hz, adjust based on your data)
    sampling_rate = 10.0

    # Perform FFT
    # fft_values = np.fft.fft(data_array)
    fft_values = rfft(data_array)
        
    # Get the magnitude spectrum (absolute values)
    magnitude = np.abs(fft_values)
        
    # Get the corresponding frequencies
    freqs = rfftfreq(len(data_array), d=1/sampling_rate)
        
    # # Find peaks in the magnitude spectrum
    peaks, _ = find_peaks(magnitude)
        
    # # Collect the peak magnitudes and corresponding frequencies
    peak_magnitudes = magnitude[peaks]
    peak_frequencies = freqs[peaks]


    if len(peak_magnitudes)>=3:
        # return (magnitude[1],freqs[1],magnitude[2],freqs[2])
        return(peak_frequencies[1],peak_magnitudes[1],peak_frequencies[2],peak_magnitudes[2])
    
    elif len(peak_magnitudes)==2:
        return(peak_frequencies[1],peak_magnitudes[1],0,0)
    else:
        return(0,0,0,0)
    
#########################################################################################################
# create function for feature extraction

def meal_feature_extractor(stretch_data):

    feature_list=[]
    #extract min value from first hour of stretch data 

    min_CGM = min(stretch_data[:12])
    min_CGM_index= stretch_data.index(min_CGM)
    
    #extract max CGM from after min CGM

    max_CGM = max(stretch_data[min_CGM_index:])
    # Find the index of the maximum value
    max_CGM_index = stretch_data.index(max_CGM)


    CGM_Diff_normalized = (max_CGM -  min_CGM)/min_CGM

    feature_list.append(CGM_Diff_normalized)
    ######################################################

    # Calculate the difference between the indices
    index_difference = max_CGM_index - min_CGM_index

    time_diff_min_max = index_difference * 5

    feature_list.append(time_diff_min_max)
    #########################################################

    for item in calculate_FFT(stretch_data):
        feature_list.append(item)

    ##########################################################
    
    #calculating derivatives 1st order

    # to_calculate_array = stretch_data[6:0]

    first_order_derivative_array= np.gradient(stretch_data)

    first_order_derivative = first_order_derivative_array.max()
    feature_list.append(first_order_derivative)

    second_order_derivative_array= np.gradient(first_order_derivative_array)

    second_order_derivative = second_order_derivative_array.max()
    feature_list.append(second_order_derivative)

    return feature_list
#####################################################################################################################

def no_meal_feature_extractor(stretch_data):

    feature_list=[]
    #extract min value from first hour of stretch data 

    min_CGM = min(stretch_data)
    min_CGM_index= stretch_data.index(min_CGM)
    
    #extract max CGM from after min CGM

    max_CGM = max(stretch_data)
    # Find the index of the maximum value
    max_CGM_index = stretch_data.index(max_CGM)


    CGM_Diff_normalized = (max_CGM -  min_CGM)/min_CGM

    feature_list.append(CGM_Diff_normalized)
    ######################################################

    # Calculate the difference between the indices
    index_difference = max_CGM_index - min_CGM_index

    time_diff_min_max = index_difference * 5

    feature_list.append(time_diff_min_max)
    #########################################################

    for item in calculate_FFT(stretch_data):
        feature_list.append(item)

    ##########################################################
    
    #calculating derivatives 1st order

    # to_calculate_array = stretch_data[6:0]

    first_order_derivative_array= np.gradient(stretch_data)

    first_order_derivative = first_order_derivative_array.max()
    feature_list.append(first_order_derivative)

    second_order_derivative_array= np.gradient(first_order_derivative_array)

    second_order_derivative = second_order_derivative_array.max()
    feature_list.append(second_order_derivative)

    return feature_list

###########################################################################
# create dataframes for meal_feaure and no meal features
feature_columns=['CGM_Max_Min_Diff','Time_diff_min_max (in minutes)',
                                   'fft_feq_1','fft_peak1','fft_feq_2','fft_peak2','gradient_1st','gradient_2nd']

meal_features_df=pd.DataFrame(columns=feature_columns )
no_meal_features_df=pd.DataFrame(columns=feature_columns )

#################################################################
# populating features df with meal data
frature_list=[]
for i in range(len(meal_df)):
    frature_list.append(meal_feature_extractor(meal_df.iloc[i].to_list()))

# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for item in frature_list:
    meal_features_df.loc[index] = item
    index += 1  # Increment the index for the next row 

meal_features_df['class_label']=1

##################################################################
# populating features df with meal data
frature_list=[]
for i in range(len(no_meal_df)):
    frature_list.append(no_meal_feature_extractor(no_meal_df.iloc[i].to_list()))

# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for item in frature_list:
    no_meal_features_df.loc[index] = item
    index += 1  # Increment the index for the next row 

no_meal_features_df['class_label']=0
##################################################################
#create final feature data frame to train model
features_df=pd.concat([meal_features_df,no_meal_features_df])

#######################################################################################################################################
#split the data into train and test set and scale the data
X=features_df.drop('class_label',axis=1)
y=features_df['class_label']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)

scaler=StandardScaler().fit(X_train.values)
X_train_scaled=scaler.transform(X_train.values)
X_test_scaled=scaler.transform(X_test.values)

#####################################################################

#train and test model

# model=RandomForestClassifier (random_state=70,n_estimators=180,max_depth=14,min_samples_split=7,criterion='gini')
model=RandomForestClassifier (random_state=100,n_estimators=500,max_depth=14,min_samples_split=7,criterion='gini')
model.fit(X_train_scaled,y_train)

#save model
dump(model, open('model.pkl', 'wb'))
# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))

training_score = model.score(X_train_scaled, y_train)
testing_score = model.score(X_test_scaled, y_test)

print(f"training_score:{training_score},testing_score:{testing_score}")
##########################################################################
