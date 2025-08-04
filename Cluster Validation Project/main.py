#!/usr/bin/env python3
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.impute import KNNImputer
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import warnings
import math 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import DBSCAN
from sklearn.cluster import BisectingKMeans

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

DATE_FORMAT = '%m/%d/%Y'
TIME_FORMAT = '%H:%M:%S'

#####################################################################################################################
#import Files

InsulinData_df = pd.read_csv("InsulinData.csv",usecols=['Date', 'Time','BWZ Carb Input (grams)'],low_memory=False)
CGMData_df= pd.read_csv("CGMData.csv",usecols=['Date', 'Time','Sensor Glucose (mg/dL)'],low_memory=False)

#####################################################################################################################
# create result_df and result_list

result_df=pd.DataFrame(columns=['SSE for Kmeans','SSE for DBSCAN','Entropy for KMeans','Entropy for DBSCAN','Purity for Kmeans','Purity for DBSCAN'])
result_list=[]

######################################################################################################################
# adding time stamp column to both CGM and Insulin dataframes and sort rows  based on time stamp and reset index

InsulinData_df['Timestamp'] = pd.to_datetime(InsulinData_df['Date'] + ' ' + InsulinData_df['Time'], format=f"{DATE_FORMAT} {TIME_FORMAT}")
InsulinData_df['Date'] = pd.to_datetime(InsulinData_df['Date'], format='%m/%d/%Y').dt.date
InsulinData_df['Time'] = pd.to_datetime(InsulinData_df['Time'], format='%H:%M:%S').dt.time

CGMData_df['Timestamp'] = pd.to_datetime(CGMData_df['Date'] + ' ' + CGMData_df['Time'], format=f"{DATE_FORMAT} {TIME_FORMAT}")
CGMData_df['Date'] = pd.to_datetime(CGMData_df['Date'], format='%m/%d/%Y').dt.date
CGMData_df['Time'] = pd.to_datetime(CGMData_df['Time'], format='%H:%M:%S').dt.time

InsulinData_df.sort_values('Timestamp',inplace=True)
CGMData_df.sort_values('Timestamp',inplace=True)
InsulinData_df.reset_index(drop=True, inplace=True)
CGMData_df.reset_index(drop=True, inplace=True)

#######################################################################################################################
#filter insulin dataframe for meal value rows

mask = pd.notna(InsulinData_df['BWZ Carb Input (grams)'])
InsulinData_df_filtered = InsulinData_df[mask & (InsulinData_df['BWZ Carb Input (grams)']!=0)].reset_index(drop=True)

########################################################################################################################
# Extract meal times from filtered Insuline Df where difference between two meal times is greater than 120 mins

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

#######################################################################################################################
# Construct meal_df 
########################################################################################################################

# Initialize an empty list to store the filtered rows

CGM_Meal_start_time=[]

# ##############################################################################
# Generate column names from TM(-25) to TM(+120) with 30 total columns
meal_df_column_names = [f'TM({i})' for i in range(-25, 125, 5)][:30]
# Create an empty DataFrame with these column names
meal_data_df = pd.DataFrame(columns=meal_df_column_names)

#############################################################################
CGM_Meal_data=[]
CGM_Meal_amount=[]

# Iterate over each timestamp in df_main
for meal_time in InsulinData_df_cleaned['Timestamp']:

    meal_amount = InsulinData_df_cleaned.loc[InsulinData_df_cleaned['Timestamp']== meal_time,'BWZ Carb Input (grams)'].values[0]
   
    # Find the first timestamp in CGMDf that is the same or greater than the meal_time
    glucose_start_df = CGMData_df[(CGMData_df['Timestamp'] >= meal_time) & (CGMData_df['Timestamp']<= (meal_time + pd.Timedelta(minutes=15)))]
    glucose_start_time= glucose_start_df['Timestamp'].min()
    CGM_Meal_start_time.append(glucose_start_time)
    # print(glucose_start_time)
    
    if pd.notna(glucose_start_time):  # if a matching or next timestamp exists

#################################################################################

        #Filling glucose sensor data with polynoial regression
        # CGMData_df[(CGMData_df['Timestamp'] >= glucose_start_time - pd.Timedelta(minutes=25)) & (CGMData_df['Timestamp']<= glucose_start_time + pd.Timedelta(minutes=120))]['Sensor Glucose (mg/dL)'].interpolate(method='polynomial', order=2, inplace=True)

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
            # new_series=pd.Series(stretch_data_full)
            # new_series=new_series.interpolate(method='polynomial', order=3)
            # stretch_data_full=new_series.tolist()
            CGM_Meal_data.append(stretch_data_full)
            CGM_Meal_amount.append(meal_amount)

# print(CGM_Meal_data)

##############################################################################
# Handle Missing data
for item in CGM_Meal_data:
    # # create an object for KNNImputer
        imputer = KNNImputer(n_neighbors=5)

        ## create an object for KNNImputer
        data_array = np.array(item).reshape(len(item), 1)
        new_data_array = imputer.fit_transform(data_array)
        CGM_Meal_data[CGM_Meal_data.index(item)]=new_data_array.flatten().tolist()

# print(CGM_Meal_data)
                
###############################################################################
#populate meal_df
# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for row in CGM_Meal_data:
    meal_data_df.loc[index] = row
    index += 1  # Increment the index for the next row 

###################################################################################################################################
# create a separate dataframe with meal data to create bins

meal_amount_df = pd.DataFrame(CGM_Meal_amount, columns=['BWZ Carb Input (grams)'])

max_meal_amount=meal_amount_df ['BWZ Carb Input (grams)'].max()
min_meal_amount=meal_amount_df ['BWZ Carb Input (grams)'].min()

meal_amount_df['bins'] = pd.cut(x=meal_amount_df['BWZ Carb Input (grams)'], bins=[2, 23, 43, 63, 83, 103,120],include_lowest=False,
                    labels=['3_23', '24_43', '44_63',
                            '64_83', '84_103','104_120'])


##################################################################################################################################
# Extract features to create features dataframe

#Function for calculating FFT
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

##################################################################################################

#Function for feature extraction from a stretch data

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

####################################################################################################
# create feature dataframe
feature_columns=['CGM_Max_Min_Diff','Time_diff_min_max (in minutes)',
                                   'fft_feq_1','fft_peak1','fft_feq_2','fft_peak2','gradient_1st','gradient_2nd']

meal_features_df=pd.DataFrame(columns=feature_columns )

# populating features df with meal data
feature_list=[]
for i in range(len(meal_data_df)):
    feature_list.append(meal_feature_extractor(meal_data_df.iloc[i].to_list()))

# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for item in feature_list:
    meal_features_df.loc[index] = item
    index += 1  # Increment the index for the next row 

#####################################################################################################
# concat meal_features_df and meal_amount_df
meal_features_df= pd.concat([meal_features_df,meal_amount_df],axis=1)

####################################################################################################################################
# Clustering using KMeans
####################################################################################################################################

model = KMeans(n_clusters=6, random_state=5,init='k-means++',max_iter=300)
model.fit(meal_features_df[['CGM_Max_Min_Diff','Time_diff_min_max (in minutes)',
                                   'fft_feq_1','fft_peak1','fft_feq_2','fft_peak2','gradient_1st','gradient_2nd']])
meal_features_df["kmeans_class"] = model.labels_
sse_Kmeans=model.inertia_
print(f"sse_kmeans:{sse_Kmeans}")
result_list.append(sse_Kmeans)

##########################################################################################################
#Construct Matrix for KMeans to calculate Entropy and Purity

kmeans_evaluation_matrix = pd.DataFrame(columns=['3_23', '24_43', '44_63',
                            '64_83', '84_103','104_120'])

bin_count_list=[]

labels=set(model.labels_)

for label in labels:
    new_meal_features_df=meal_features_df[meal_features_df["kmeans_class"]==label]

    # group based on bins
    grouped_df = new_meal_features_df.groupby('bins')
    group_counts = grouped_df.size()
    bin_count_list.append(group_counts.to_list())

print(bin_count_list)

##########################################################################
#Populate kmeans_evaluation_matrix
# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for item in bin_count_list:
    kmeans_evaluation_matrix.loc[index] = item
    index += 1  # Increment the index for the next row 

########################################################################################################################################
# Clustering using DBSCAN
#######################################################################################################################################

X_train_scaled = StandardScaler().fit_transform(meal_features_df[['CGM_Max_Min_Diff','Time_diff_min_max (in minutes)',
                                   'fft_feq_1','fft_peak1','fft_feq_2','fft_peak2','gradient_1st','gradient_2nd']].values)
dbscan = DBSCAN(eps=0.7, min_samples=8 ,p=2)
dbscan.fit(X_train_scaled)
labels = dbscan.labels_
meal_features_df["db_scan_class"] = dbscan.labels_

# Take the rows where label is -1
meal_features_df_1 = meal_features_df.loc[meal_features_df["db_scan_class"]==-1]
meal_features_df_1.head(2)
indices=meal_features_df.loc[meal_features_df["db_scan_class"]==-1].index

# apply bisecting kmeans for the rows where label is -1
bisect_means = BisectingKMeans(n_clusters=5, random_state=5).fit(meal_features_df_1[['CGM_Max_Min_Diff','Time_diff_min_max (in minutes)',
                                   'fft_feq_1','fft_peak1','fft_feq_2','fft_peak2','gradient_1st','gradient_2nd']])
bisec_kmeans_labels = bisect_means.labels_

# map the labels to 1,2,3,4,5 as 0 was already there with dbscan classes
mapping = {0:1, 1:2, 2:3, 3:4,4:5}
bisec_kmeans_labels = [mapping[i] for i in bisec_kmeans_labels]

# replacing the rows where dbscan label is -1 with new bisecting kmeans labels
count=0
for i in indices:
    meal_features_df.loc[i,"db_scan_class"]=bisec_kmeans_labels[count]
    count +=1

##################################################################################################
#Calculate sse for DBSCAN
##################################################################################################
# Initialize lists to store centroids and SSE for each cluster
sse_list = []
dbscan_labels=meal_features_df["db_scan_class"].unique()
# Calculate centroids and SSE for each cluster
for label in dbscan_labels:

    # Get points that belong to the current cluster
    cluster_points = meal_features_df[meal_features_df["db_scan_class"] == label]
    cluster_points=cluster_points.loc[:,['CGM_Max_Min_Diff','Time_diff_min_max (in minutes)',
                                    'fft_feq_1','fft_peak1','fft_feq_2','fft_peak2','gradient_1st','gradient_2nd']]
    centroid = np.mean(cluster_points, axis=0)
    
    cluster_points_sse=[]
    for i in range(len(cluster_points)):
        sse = np.sum((centroid - cluster_points.iloc[i])**2)
        cluster_points_sse.append(sse)
    sse_cluster=np.sum(cluster_points_sse)
    sse_list.append(sse_cluster)

    
sse_DBSCAN=np.sum(sse_list)
result_list.append(sse_DBSCAN)
print(f"sse_DBSCAN: {sse_DBSCAN}")

######################################################################################################
# Construct DBSCAN evaluation matrix to calculate entropy and purity

dbscan_evaluation_matrix = pd.DataFrame(columns=['3_23', '24_43', '44_63',
                            '64_83', '84_103','104_120'])

bin_count_list=[]

labels=set(meal_features_df["db_scan_class"])

for label in labels:
    new_meal_features_df=meal_features_df[meal_features_df["db_scan_class"]==label]

    # group based on bins
    grouped_df = new_meal_features_df.groupby('bins')
    group_counts = grouped_df.size()
    bin_count_list.append(group_counts.to_list())

print(bin_count_list)

##########################################################################
#Populate kmeans_evaluation_matrix
# Initialize an index counter
index = 0

# Append each list as a row in the DataFrame using .loc[]
for item in bin_count_list:
    dbscan_evaluation_matrix.loc[index] = item
    index += 1  # Increment the index for the next row 


######################################################################################################################################
# Function for calculating total entropy
#######################################################################################################################################
def calculate_total_entropy(input_matrix):
    class_entropies=[]
    #Entropy Calculation
    for i in range(len(input_matrix)):
        entropies =[]
        total_c1ass_element=np.sum(input_matrix.iloc[i],axis=0)
        for j in range(6):
            if input_matrix.iloc[i,j] != 0:
               entropy_bin = - ((input_matrix.iloc[i,j])/total_c1ass_element)* math.log((input_matrix.iloc[i,j])/total_c1ass_element)
            else:
               entropy_bin =0
            entropies.append(entropy_bin)
        class_entropies.append(np.sum(entropies))
    # print(class_entropies)
    ###############################################################################################################

    #  Calculate Total Entropy
    weighted_entropies=[]
    for i in range(6):
        total_c1ass_element=np.sum(input_matrix.iloc[i],axis=0)
        weighted_entropy=total_c1ass_element*class_entropies[i]
        weighted_entropies.append(weighted_entropy)

    total_sum = input_matrix.values.sum()
    total_entropy=(np.sum(weighted_entropies)/total_sum)*0.5 
    # print(f"total entropy : {total_entropy}")

    return total_entropy

#########################################################################################################################################
# Function for calculating total Purity
##########################################################################################################################################
# Calculate Purity
def calculate_total_purity(input_matrix):
    class_purity=[]
    #Entropy Calculation
    for i in range(len(input_matrix)):
        purities =[]
        total_c1ass_element=np.sum(input_matrix.iloc[i],axis=0)
        for j in range(6):
            purity_bin = ((input_matrix.iloc[i,j])/total_c1ass_element)       
            purities.append(purity_bin)

        class_purity.append(max(purities))
    # print(class_purity)
    ###############################################################################################################

    #  Calculate Total Entropy
    weighted_purities=[]
    for i in range(6):
        total_c1ass_element=np.sum(input_matrix.iloc[i],axis=0)
        weighted_purity=total_c1ass_element*class_purity[i]
        weighted_purities.append(weighted_purity)

    total_sum = input_matrix.values.sum()
    total_purity=(np.sum(weighted_purities)/total_sum)*2
    # print(f"total purity : {total_purity}")

    return total_purity

############################################################################################################################################
# Appending results in result list

entropy_kmeans=calculate_total_entropy(kmeans_evaluation_matrix)
result_list.append(entropy_kmeans)
entropy_dbscan=calculate_total_entropy(dbscan_evaluation_matrix)
result_list.append(entropy_dbscan)
purity_kmeans=calculate_total_purity(kmeans_evaluation_matrix)
result_list.append(purity_kmeans)
purity_DBSCAN=calculate_total_purity(dbscan_evaluation_matrix)
result_list.append(purity_DBSCAN)

######################################################################################
#Populate result_df with result list

index = 0
# Append each list as a row in the DataFrame using .loc[]
for item in [result_list]:
    result_df.loc[index] = item
    index += 1  # Increment the index for the next row 

#####################################################################################
# create result.csv from result dataframe

result_df.to_csv('Result.csv',header=False,index=False)
