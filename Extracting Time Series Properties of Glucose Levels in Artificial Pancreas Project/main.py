#!/usr/bin/env python3 
import pandas as pd
import numpy as np
import datetime as dt
import os

##################################################################################################################
# Reading csv files
current_working_directory = os.getcwd()

full_path_insulin= os.path.join(os.getcwd(),'InsulinData.csv')
Insulin_df = pd.read_csv(full_path_insulin,low_memory=False)

full_path_CGM = os.path.join(os.getcwd(),'CGMData.csv')
CGM_df = pd.read_csv(full_path_CGM, low_memory=False)

################################################################################################################

#Adding a timestamp column to Insulin_df
date_format = '%m/%d/%Y'
time_format = '%H:%M:%S'
Insulin_df['Timestamp']= pd.to_datetime(Insulin_df['Date'] + ' ' + Insulin_df['Time'], format=f"{date_format} {time_format}")

# Get the timestamp from when auto mode starts
filtered_Insulin_df=Insulin_df[Insulin_df['Alarm']=="AUTO MODE ACTIVE PLGM OFF"]
auto_mode_start = filtered_Insulin_df[['Timestamp']].min()

###################################################################################################################
# Keeping only the necessary column required for analysis
CGM_df=CGM_df[['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)']]

# Adding a time stamp column and changing the datatype of date and time column of CGM_df
date_format = '%m/%d/%Y'
time_format = '%H:%M:%S'
CGM_df['Timestamp']= pd.to_datetime(CGM_df['Date'] + ' ' + CGM_df['Time'], format=f"{date_format} {time_format}")

CGM_df['Date'] = pd.to_datetime(CGM_df['Date'], format='%m/%d/%Y').dt.normalize()
CGM_df['Time'] = pd.to_datetime(CGM_df['Time'], format='%H:%M:%S').dt.time

######################################################################################################################
# Dividing the CGM_df in auto and manual based on the timestamp got from Insulin_df
CGM_df_auto = CGM_df[CGM_df['Timestamp']> auto_mode_start['Timestamp']]
CGM_df_manual = CGM_df[CGM_df['Timestamp']<= auto_mode_start['Timestamp']]

print(f"length of CGM_df_auto: {len(CGM_df_auto)}")
print(f"length of CGM_df_manual: {len(CGM_df_manual)}")

# Dropping missing value rows
CGM_df_auto=CGM_df_auto.dropna(subset=['Sensor Glucose (mg/dL)'])
CGM_df_manual=CGM_df_manual.dropna(subset=['Sensor Glucose (mg/dL)'])

# Dividing the CGM_df_auto and CGM_df_manual in daytime and overnight
daytime_starttime = dt.datetime.strptime('06:00:00', '%H:%M:%S').time()
daytime_endtime = dt.datetime.strptime('23:59:59', '%H:%M:%S').time()
overnight_start_time= dt.datetime.strptime('00:00:00', '%H:%M:%S').time()


CGM_df_auto_daytime = CGM_df_auto[(CGM_df_auto['Time']>=daytime_starttime) & (CGM_df_auto['Time']<=daytime_endtime)]
CGM_df_auto_overnight=CGM_df_auto[(CGM_df_auto['Time']>=overnight_start_time) & (CGM_df_auto['Time']<=daytime_starttime)]
CGM_df_manual_daytime = CGM_df_manual[(CGM_df_manual['Time']>=daytime_starttime) & (CGM_df_manual['Time']<=daytime_endtime)]
CGM_df_manual_overnight=CGM_df_manual[(CGM_df_manual['Time']>=overnight_start_time) & (CGM_df_manual['Time']<=daytime_starttime)]

#############################################################################################################
# Defining three functions to measure percentage values of CGM Data

def measure_upper(df, upper_range):
    try:
        df_new = df[df['Sensor Glucose (mg/dL)']> upper_range]
        if len(df_new) >0 :
            df_new = (df_new.groupby(['Date']).count()/2.88)
            avg_percentage_value=df_new['Time'].median()
            return avg_percentage_value
        else:
            return 0
    except Exception as e:
        return str(e)

    

def measure_lower(df, lower_range):
    try:
        df_new = df[df['Sensor Glucose (mg/dL)']< lower_range]
        if len(df_new) >0 :
            df_new = (df_new.groupby(['Date']).count()/2.88)
            avg_percentage_value=df_new['Time'].median()
            return avg_percentage_value
        else:
            return 0
    except Exception as e:
        return str(e)
    
def measure_in_ranges(df, lower_range , upper_range):
    try: 
        df_new = df[(df['Sensor Glucose (mg/dL)']>=lower_range) & (df['Sensor Glucose (mg/dL)']<=upper_range)]
        if len(df_new) >0 :
            df_new = (df_new.groupby(['Date']).count()/2.88)
            avg_percentage_value=df_new['Time'].median()
            return avg_percentage_value
        else:
            return 0
    except Exception as e:
        return str(e)

manual_measures_list=[]
auto_measures_list=[]

##################################################################################################
# Deriving measures for manual Mode
####################################################################################################
# Overnight measurement for manual mode

# Percentage time in hyperglycemia (CGM > 180 mg/dL)
overnight_Percentage_time_in_hyperglycemia_manual = measure_upper(CGM_df_manual_overnight,180)
manual_measures_list.append(overnight_Percentage_time_in_hyperglycemia_manual)

# percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
overnight_Percentage_time_in_hyperglycemia_critical_manual= measure_upper(CGM_df_manual_overnight,250)
manual_measures_list.append(overnight_Percentage_time_in_hyperglycemia_critical_manual)

# percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
overnight_Percentage_time_in_ranges_manual = measure_in_ranges(CGM_df_manual_overnight, 70 , 180)
manual_measures_list.append(overnight_Percentage_time_in_ranges_manual)

# percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
overnight_Percentage_time_in_ranges_secondary_manual = measure_in_ranges(CGM_df_manual_overnight, 70 , 150)
manual_measures_list.append(overnight_Percentage_time_in_ranges_secondary_manual)

# percentage time in hypoglycemia level 1 (CGM < 70 mg/dL),
overnight_Percentage_time_in_hypoglycemia_level_1_manual = measure_lower(CGM_df_manual_overnight,70)
manual_measures_list.append(overnight_Percentage_time_in_hypoglycemia_level_1_manual)

# percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).
overnight_Percentage_time_in_hypoglycemia_level_2_manual = measure_lower(CGM_df_manual_overnight,54)
manual_measures_list.append(overnight_Percentage_time_in_hypoglycemia_level_2_manual)

###############################################################################################################
# Daytime measurement for manual mode

# Percentage time in hyperglycemia (CGM > 180 mg/dL)
daytime_Percentage_time_in_hyperglycemia_manual = measure_upper(CGM_df_manual_daytime,180)
manual_measures_list.append(daytime_Percentage_time_in_hyperglycemia_manual)

# percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
daytime_Percentage_time_in_hyperglycemia_critical_manual = measure_upper(CGM_df_manual_daytime,250)
manual_measures_list.append(daytime_Percentage_time_in_hyperglycemia_critical_manual)

# percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
daytime_Percentage_time_in_ranges_manual = measure_in_ranges(CGM_df_manual_daytime, 70 , 180)
manual_measures_list.append(daytime_Percentage_time_in_ranges_manual)

# percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
daytime_Percentage_time_in_ranges_secondary_manual = measure_in_ranges(CGM_df_manual_daytime, 70 , 150)
manual_measures_list.append(daytime_Percentage_time_in_ranges_secondary_manual)

# percentage time in hypoglycemia level 1 (CGM < 70 mg/dL),
daytime_Percentage_time_in_hypoglycemia_level_1_manual = measure_lower(CGM_df_manual_daytime,70)
manual_measures_list.append(daytime_Percentage_time_in_hypoglycemia_level_1_manual)

# percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).
daytime_Percentage_time_in_hypoglycemia_level_2_manual = measure_lower(CGM_df_manual_daytime,54)
manual_measures_list.append(daytime_Percentage_time_in_hypoglycemia_level_2_manual)

#####################################################################################################################

# Whole day measurement for manual mode
# Percentage time in hyperglycemia (CGM > 180 mg/dL)
whole_day_Percentage_time_in_hyperglycemia_manual = measure_upper(CGM_df_manual,180)
manual_measures_list.append(whole_day_Percentage_time_in_hyperglycemia_manual)

# percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
whole_day_Percentage_time_in_hyperglycemia_critical_manual = measure_upper(CGM_df_manual,250)
manual_measures_list.append(whole_day_Percentage_time_in_hyperglycemia_critical_manual)

# percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
whole_day_percentage_time_in_ranges_manual = measure_in_ranges(CGM_df_manual, 70 , 180)
manual_measures_list.append(whole_day_percentage_time_in_ranges_manual )

# percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
whole_day_percentage_time_in_ranges_secondary_manual = measure_in_ranges(CGM_df_manual, 70 , 150)
manual_measures_list.append(whole_day_percentage_time_in_ranges_secondary_manual)

# percentage time in hypoglycemia level 1 (CGM < 70 mg/dL),
whole_day_percentage_time_in_hypoglycemia_level_1_manual = measure_lower(CGM_df_manual,70)
manual_measures_list.append(whole_day_percentage_time_in_hypoglycemia_level_1_manual)

# percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).
whole_day_percentage_time_in_hypoglycemia_level_2_manual = measure_lower(CGM_df_manual,54)
manual_measures_list.append(whole_day_percentage_time_in_hypoglycemia_level_2_manual)


##################################################################################################
# Deriving measures for auto Mode
####################################################################################################

# Overnight measurement for auto mode
# Percentage time in hyperglycemia (CGM > 180 mg/dL)
overnight_Percentage_time_in_hyperglycemia_auto = measure_upper(CGM_df_auto_overnight,180)
auto_measures_list.append(overnight_Percentage_time_in_hyperglycemia_auto)

# percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
overnight_Percentage_time_in_hyperglycemia_critical_auto= measure_upper(CGM_df_auto_overnight,250)
auto_measures_list.append(overnight_Percentage_time_in_hyperglycemia_critical_auto)

# percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
overnight_Percentage_time_in_ranges_auto = measure_in_ranges(CGM_df_auto_overnight, 70 , 180)
auto_measures_list.append(overnight_Percentage_time_in_ranges_auto)

# percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
overnight_Percentage_time_in_ranges_secondary_auto = measure_in_ranges(CGM_df_auto_overnight, 70 , 150)
auto_measures_list.append(overnight_Percentage_time_in_ranges_secondary_auto)

# percentage time in hypoglycemia level 1 (CGM < 70 mg/dL),
overnight_Percentage_time_in_hypoglycemia_level_1_auto = measure_lower(CGM_df_auto_overnight,70)
auto_measures_list.append(overnight_Percentage_time_in_hypoglycemia_level_1_auto)

# percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).
overnight_Percentage_time_in_hypoglycemia_level_2_auto = measure_lower(CGM_df_auto_overnight,54)
auto_measures_list.append(overnight_Percentage_time_in_hypoglycemia_level_2_auto)

#########################################################################################################

# Daytime measurement for auto mode
# Percentage time in hyperglycemia (CGM > 180 mg/dL)
daytime_Percentage_time_in_hyperglycemia_auto = measure_upper(CGM_df_auto_daytime,180)
auto_measures_list.append(daytime_Percentage_time_in_hyperglycemia_auto)

# percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
daytime_Percentage_time_in_hyperglycemia_critical_auto = measure_upper(CGM_df_auto_daytime,250)
auto_measures_list.append(daytime_Percentage_time_in_hyperglycemia_critical_auto)

# percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
daytime_Percentage_time_in_ranges_auto = measure_in_ranges(CGM_df_auto_daytime, 70 , 180)
auto_measures_list.append(daytime_Percentage_time_in_ranges_auto)

# percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
daytime_Percentage_time_in_ranges_secondary_auto = measure_in_ranges(CGM_df_auto_daytime, 70 , 150)
auto_measures_list.append(daytime_Percentage_time_in_ranges_secondary_auto)

# percentage time in hypoglycemia level 1 (CGM < 70 mg/dL),
daytime_Percentage_time_in_hypoglycemia_level_1_auto = measure_lower(CGM_df_auto_daytime,70)
auto_measures_list.append(daytime_Percentage_time_in_hypoglycemia_level_1_auto)

# percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).
daytime_Percentage_time_in_hypoglycemia_level_2_auto = measure_lower(CGM_df_auto_daytime,54)
auto_measures_list.append(daytime_Percentage_time_in_hypoglycemia_level_2_auto)

############################################################################################################

# Whole day measurement for auto mode

# Percentage time in hyperglycemia (CGM > 180 mg/dL)
whole_day_Percentage_time_in_hyperglycemia_auto = measure_upper(CGM_df_auto,180)
auto_measures_list.append(whole_day_Percentage_time_in_hyperglycemia_auto)

# percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
whole_day_Percentage_time_in_hyperglycemia_critical_auto = measure_upper(CGM_df_auto,250)
auto_measures_list.append(whole_day_Percentage_time_in_hyperglycemia_critical_auto)

# percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
whole_day_percentage_time_in_ranges_auto = measure_in_ranges(CGM_df_auto, 70 , 180)
auto_measures_list.append(whole_day_percentage_time_in_ranges_auto )

# percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
whole_day_percentage_time_in_ranges_secondary_auto = measure_in_ranges(CGM_df_auto, 70 , 150)
auto_measures_list.append(whole_day_percentage_time_in_ranges_secondary_auto)

# percentage time in hypoglycemia level 1 (CGM < 70 mg/dL),
whole_day_percentage_time_in_hypoglycemia_level_1_auto = measure_lower(CGM_df_auto,70)
auto_measures_list.append(whole_day_percentage_time_in_hypoglycemia_level_1_auto)

# percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).
whole_day_percentage_time_in_hypoglycemia_level_2_auto = measure_lower(CGM_df_auto,54)
auto_measures_list.append(whole_day_percentage_time_in_hypoglycemia_level_2_auto)

#################################################################################################################

result_df=pd.DataFrame(columns=['Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)',
       'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
       'Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
       'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
       'Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
       'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)',
       'Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)',
       'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
       'Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
       'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
       'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
       'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)',
       'Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)',
       'Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
       'Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
       'Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
       'Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
       'Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'],
       index=['Manual Mode', 'Auto Mode'])
# Populating result_df with values

for i in range(1):
    for j in range(18):
        result_df.iloc[i,j] = manual_measures_list[j]

for i in range(1,2):
    for j in range(18):
        result_df.iloc[i,j] = auto_measures_list[j]

# generating Result.csv from dataframe

result_df.to_csv('Result.csv', index= False, header= False , mode='w')

