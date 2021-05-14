#!/usr/bin/env python
# coding: utf-8

# ## Phase 1: Part 1: Clean up of CCD files. This notebook contains code that cleans and merges the four individual ccd files for school directory, characteristics, farms and staff files into a singular file

# ### Loading necessary libraries

# In[221]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[222]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CCD


# ### 1. Cleaning and checking directory file

# In[223]:


ccd_directory = pandas.read_csv("ccd_DIRECTORY.csv")
ccd_directory.head()


# In[224]:


ccd_directory.columns


# In[225]:


ccd_directory.shape


# In[226]:


ccd_directory.describe()


# #### Checking for missing or null values

# In[227]:


sns.heatmap(ccd_directory.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[228]:


null_columns=ccd_directory.columns[ccd_directory.isnull().any()]
ccd_directory[null_columns].isnull().sum()


# We see alot of columns have null values but since we wont be using these columns in our analysis we can drop these and any other unneccessary columns from further analysis

# In[229]:


#ccd_directory.head().T


# ##### Dropping unnecessary columns 

# In[230]:


ccd_directory.drop(ccd_directory.columns[[1,2,7,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,32,33,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
                                         57,58,59,60,61,62,64]], axis=1, inplace=True)


# ##### The NCESSCH is a federally assigned school ID that is 12strings long with leading zeros, so lets correct that

# In[231]:


ccd_directory['NCESSCH'] = ccd_directory['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[232]:


#ccd_directory.head()


# ##### Checking for missing or null values

# In[233]:


sns.heatmap(ccd_directory.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# Now are remaining data contains no missing values

# In[234]:


ccd_directory['NCESSCH'].is_unique # checking to see if there are any schools that are duplicated, based on the ids all our schools are unique


# In[235]:


ccd_directory.shape


# In[236]:


count = ccd_directory['LEVEL'].value_counts()
print(count)


# ##### So we see there are schools of different levels included in the directory but we are only interested in high schools so we will need to filter those schools out, but we will do that later.

# In[237]:


#ccd_directory.describe()


# In[238]:


#ccd_directory.hist()


# Let's save a copy of this file

# In[239]:


ccd_directory.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CCD/Clean_ccd_directory.csv', index = False, header=True)


# ### 2. Cleaning and checking characteristics file

# In[240]:


ccd_character = pandas.read_csv("ccd_CHARACTER.csv")
ccd_character.head()


# In[241]:


count = ccd_character['TITLEI_STATUS'].value_counts()
print(count)


# Some values are missing and not reported we will have to remember to handle this later in our analysis

# In[242]:


ccd_character.columns


# In[243]:


ccd_character.shape


# In[244]:


ccd_character.describe()


# ##### Checking for missing or null values

# In[245]:


sns.heatmap(ccd_character.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[246]:


null_columns=ccd_character.columns[ccd_character.isnull().any()]
ccd_character[null_columns].isnull().sum()


# We see the Union column has null values but since we wont be using this column in our analysis we can drop this column and any other unnecessary columns from further analysis

# ##### Dropping unnecessary columns 

# In[247]:


ccd_character.drop(ccd_character.columns[[0,1,2,3,5,6,7,8,9,11,12,15,16,17,18,19]], axis=1, inplace=True)


# In[248]:


ccd_character.columns


# ##### The NCESSCH is a federally assigned school ID that is 12strings long with leading zeros, so lets correct that

# In[249]:


ccd_character['NCESSCH'] = ccd_character['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[250]:


ccd_character.head()


# In[251]:


#ccd_character['NCESSCH'].is_unique


# In[252]:


sns.heatmap(ccd_character.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# Now are remaining data contains no missing values

# In[253]:


ccd_character.shape


# In[254]:


ccd_character.describe()


# In[255]:


ccd_character.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CCD/Clean_ccd_character.csv', index = False, header=True)


# #### 3. Cleaning and checking farms file
# ##### Farms refers to students who are eligible for free and reduced priced meals at the school due to their families' low income status

# In[256]:


ccd_farms= pandas.read_csv("ccd_FARMS.csv")
ccd_farms.head()


# In[257]:


ccd_farms.columns


# In[258]:


ccd_farms.shape


# In[259]:


ccd_farms.describe()


# ##### Checking for missing or null values

# In[260]:


sns.heatmap(ccd_farms.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[261]:


null_columns=ccd_farms.columns[ccd_farms.isnull().any()]
ccd_farms[null_columns].isnull().sum()


# We see the Union and Student_count columns have null values but since we wont be using the Union column in our analysis we can drop this column and any other unnecessary columns from further analysis

# ##### Dropping unnecessary columns 

# In[262]:


ccd_farms.drop(ccd_farms.columns[[0,1,2,3,5,6,7,8,9,11]], axis=1, inplace=True)


# ##### The NCESSCH is a federally assigned school ID that is 12strings long with leading zeros, so lets correct that

# In[263]:


ccd_farms['NCESSCH'] = ccd_farms['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[264]:


ccd_farms.head()


# In[265]:


ccd_farms['NCESSCH'].is_unique #need a list of unique schools since it appears schools are duplicated


# We are interested in getting the schools that qualified for Free and reduced-price lunch (FARMS) and get the counts of those FARMS students

# In[266]:


ccd_farms_EUT=ccd_farms[ccd_farms['TOTAL_INDICATOR']=='Education Unit Total']


# In[267]:


ccd_farms_EUT.head()


# In[268]:


ccd_farms_EUT.shape


# In[269]:


ccd_farms_FRL=ccd_farms_EUT[ccd_farms_EUT['DATA_GROUP']=='Free and Reduced-price Lunch Table']


# In[270]:


ccd_farms_FRL.head()


# In[271]:


ccd_farms_FRL.shape


# Lets rename the column name so its clear what the counts reflect

# In[272]:


ccd_farms_FRL.rename(columns={'STUDENT_COUNT':'FARMS_COUNT'}, inplace=True)


# In[273]:


ccd_farms_FRL.head()


# ##### Checking for missing or null values

# In[274]:


sns.heatmap(ccd_farms_FRL.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[275]:


null_columns=ccd_farms_FRL.columns[ccd_farms_FRL.isnull().any()]
ccd_farms_FRL[null_columns].isnull().sum()


# In[276]:


ccd_farms_FRL['NCESSCH'].is_unique


# Lets drop any rows that have null counts reported for the Farms_count column

# In[277]:


ccd_farms_FRL_new = ccd_farms_FRL.dropna(axis = 0, how ='any') 


# In[278]:


print("Old data frame length:", len(ccd_farms_FRL)) 
print("New data frame length:", len(ccd_farms_FRL_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(ccd_farms_FRL)-len(ccd_farms_FRL_new))) 


# In[279]:


sns.heatmap(ccd_farms_FRL_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[280]:


ccd_farms_FRL_new.shape


# In[281]:


ccd_farms_FRL_new.hist()


# In[282]:


ccd_farms_FRL_new.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CCD/Clean_ccd_farms.csv', index = False, header=True)


# #### 4. Cleaning and checking staff file

# In[283]:


ccd_staff= pandas.read_csv("ccd_STAFF.csv")
ccd_staff.head()


# In[284]:


ccd_staff.columns


# In[285]:


ccd_staff.shape


# In[286]:


ccd_staff.describe()


# ##### Checking for missing or null values

# In[287]:


sns.heatmap(ccd_staff.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[288]:


null_columns=ccd_staff.columns[ccd_staff.isnull().any()]
ccd_staff[null_columns].isnull().sum()


# We see the Union column has null values but since we wont be using the Union column in our analysis we can drop this column and any other unnecessary columns from further analysis

# ##### Dropping unnecessary columns 

# In[289]:


ccd_staff.drop(ccd_staff.columns[[0,1,2,3,5,6,7,8,9,11,13,14]], axis=1, inplace=True)


# In[290]:


ccd_staff['NCESSCH'] = ccd_staff['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[291]:


ccd_staff['NCESSCH'].is_unique


# In[292]:


ccd_staff_new = ccd_staff.dropna(axis = 0, how ='any') 


# In[293]:


print("Old data frame length:", len(ccd_staff)) 
print("New data frame length:", len(ccd_staff_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(ccd_staff)-len(ccd_staff_new))) 


# In[294]:


sns.heatmap(ccd_staff_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[295]:


ccd_staff_new.head()


# In[296]:


ccd_staff_new.shape


# In[297]:


ccd_staff_new.hist()


# In[298]:


ccd_staff_new.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CCD/Clean_ccd_staff.csv', index = False, header=True)


# #### 5. Merge all files into master ccd file

# ##### Merging directory to characteristics file

# In[299]:


merged_dir_char = pandas.merge(left=ccd_directory,right=ccd_character, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_dir_char.shape


# In[300]:


merged_dir_char.columns


# In[301]:


merged_dir_char.drop([col for col in merged_dir_char.columns if col.endswith('_y')],axis=1,inplace=True)


# In[302]:


merged_dir_char.head()


# In[303]:


merged_dir_char.rename(columns = {'SCH_NAME_x':'SCH_NAME'}, inplace = True) 


# ##### Merging directory_characteristics file to staff file

# In[304]:


merged_dir_char_staff = pandas.merge(left=merged_dir_char,right=ccd_staff_new, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_dir_char_staff.shape


# In[305]:


merged_dir_char_staff.columns


# In[306]:


merged_dir_char_staff.drop([col for col in merged_dir_char_staff.columns if col.endswith('_y')],axis=1,inplace=True)


# In[307]:


merged_dir_char_staff.rename(columns = { 'SCH_NAME_x':'SCH_NAME'}, inplace = True) 


# ##### Merging directory_characteristics_staff file to farms file

# In[308]:


merged_dir_char_staff_farms = pandas.merge(left=merged_dir_char_staff,right=ccd_farms_FRL_new, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_dir_char_staff_farms.shape


# In[309]:


merged_dir_char_staff_farms.columns


# In[310]:


merged_dir_char_staff_farms.drop([col for col in merged_dir_char_staff_farms.columns if col.endswith('_y')],axis=1,inplace=True)


# In[311]:


merged_dir_char_staff_farms.rename(columns = {'SCH_NAME_x':'SCH_NAME' }, inplace = True) 


# In[312]:


merged_dir_char_staff_farms.columns


# In[313]:


sns.heatmap(merged_dir_char_staff_farms.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[314]:


null_columns=merged_dir_char_staff_farms.columns[merged_dir_char_staff_farms.isnull().any()]
merged_dir_char_staff_farms[null_columns].isnull().sum()


# In[315]:


merged_dir_char_staff_farms.head().T


# ##### Dropping unnecessary columns 

# In[316]:


merged_dir_char_staff_farms.drop(merged_dir_char_staff_farms.columns[[3,4,5,6,9,18,20,21]], axis=1, inplace=True)


# In[317]:


merged_dir_char_staff_farms.head()


# #### We are keeping only the open schools for the SY

# In[318]:


ccd_open_schools=merged_dir_char_staff_farms[merged_dir_char_staff_farms['SY_STATUS']==1]


# In[319]:


sns.heatmap(ccd_open_schools.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# #### We are keeping only the high schools in our dataset

# In[320]:


ccd_master_hs=ccd_open_schools[ccd_open_schools['LEVEL']=='High']


# In[321]:


ccd_master_hs.shape


# In[322]:


sns.heatmap(ccd_master_hs.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# #### Lets drop all the rows with null values

# In[323]:


ccd_master_hs_new = ccd_master_hs.dropna(axis = 0, how ='any') 


# In[324]:


print("Old data frame length:", len(ccd_master_hs)) 
print("New data frame length:", len(ccd_master_hs_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(ccd_master_hs)-len(ccd_master_hs_new))) 


# In[325]:


sns.heatmap(ccd_master_hs_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# Lets also address the issue with Title 1 before going further.

# In[326]:


count = ccd_master_hs_new['TITLEI_STATUS'].value_counts()
print(count)


# In[327]:


ccd_master_hs_new = ccd_master_hs_new[ccd_master_hs_new.TITLEI_STATUS != "MISSING"]


# In[328]:


ccd_master_hs_new.shape


# ##### Saving final copy of the merge file for later use

# In[329]:


ccd_master_hs_new.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CCD/Clean_ccd_master.csv', index = False, header=True)

