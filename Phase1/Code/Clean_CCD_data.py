#!/usr/bin/env python
# coding: utf-8

# ## Phase 1: Clean up of CCD data files

# In[1]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD


# #### Cleaning and checking directory file

# In[3]:


ccd_directory = pandas.read_csv("ccd_DIRECTORY.csv")
ccd_directory.head()


# In[4]:


ccd_directory.columns


# In[5]:


ccd_directory.shape


# In[6]:


ccd_directory.describe()


# In[7]:


sns.heatmap(ccd_directory.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[8]:


null_columns=ccd_directory.columns[ccd_directory.isnull().any()]
ccd_directory[null_columns].isnull().sum()


# In[9]:


ccd_directory.head().T


# In[10]:


ccd_directory.drop(ccd_directory.columns[[1,2,7,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,32,33,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
                                         57,58,59,60,61,62,64]], axis=1, inplace=True)


# In[11]:


ccd_directory['NCESSCH'] = ccd_directory['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[12]:


ccd_directory.head()


# In[13]:


sns.heatmap(ccd_directory.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[14]:


ccd_directory['NCESSCH'].is_unique


# In[15]:


ccd_directory.shape


# In[16]:


ccd_directory.describe()


# In[17]:


ccd_directory.hist()


# In[18]:


ccd_directory.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD/Clean_ccd_directory.csv', index = False, header=True)


# #### Cleaning and checking characteristics file

# In[19]:


ccd_character = pandas.read_csv("ccd_CHARACTER.csv")
ccd_character.head()


# In[20]:


ccd_character.columns


# In[21]:


ccd_character.shape


# In[22]:


ccd_character.describe()


# In[23]:


sns.heatmap(ccd_character.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[24]:


null_columns=ccd_character.columns[ccd_character.isnull().any()]
ccd_character[null_columns].isnull().sum()


# In[25]:


ccd_character.drop(ccd_character.columns[[1,2,6,7,8,12,16,17,18,19]], axis=1, inplace=True)


# In[26]:


ccd_character['NCESSCH'] = ccd_character['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[27]:


ccd_character.head()


# In[28]:


ccd_character['NCESSCH'].is_unique


# In[29]:


sns.heatmap(ccd_character.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[30]:


ccd_character.shape


# In[31]:


ccd_character.hist()


# In[32]:


ccd_character.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD/Clean_ccd_character.csv', index = False, header=True)


# #### Cleaning and checking farms file

# In[33]:


ccd_farms= pandas.read_csv("ccd_FARMS.csv")
ccd_farms.head()


# In[34]:


ccd_farms.columns


# In[35]:


ccd_farms.shape


# In[36]:


ccd_farms.describe()


# In[37]:


sns.heatmap(ccd_farms.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[38]:


null_columns=ccd_farms.columns[ccd_farms.isnull().any()]
ccd_farms[null_columns].isnull().sum()


# In[39]:


ccd_farms.drop(ccd_farms.columns[[1,2,6,7,8]], axis=1, inplace=True)


# In[40]:


ccd_farms['NCESSCH'] = ccd_farms['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[41]:


ccd_farms.head()


# In[42]:


ccd_farms['NCESSCH'].is_unique #need a list of unique schools


# In[43]:


ccd_farms_EUT=ccd_farms[ccd_farms['TOTAL_INDICATOR']=='Education Unit Total']


# In[44]:


ccd_farms_EUT.head()


# In[45]:


ccd_farms_EUT.shape


# In[46]:


ccd_farms_FRL=ccd_farms_EUT[ccd_farms_EUT['DATA_GROUP']=='Free and Reduced-price Lunch Table']


# In[47]:


ccd_farms_FRL.head()


# In[48]:


ccd_farms_FRL.shape


# In[49]:


ccd_farms_FRL.rename(columns={'STUDENT_COUNT':'FARMS_COUNT'}, inplace=True)


# In[50]:


ccd_farms_FRL.head()


# In[51]:


sns.heatmap(ccd_farms_FRL.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[52]:


null_columns=ccd_farms_FRL.columns[ccd_farms_FRL.isnull().any()]
ccd_farms_FRL[null_columns].isnull().sum()


# In[53]:


ccd_farms_FRL['NCESSCH'].is_unique


# In[54]:


ccd_farms_FRL_new = ccd_farms_FRL.dropna(axis = 0, how ='any') 


# In[55]:


print("Old data frame length:", len(ccd_farms_FRL)) 
print("New data frame length:", len(ccd_farms_FRL_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(ccd_farms_FRL)-len(ccd_farms_FRL_new))) 


# In[56]:


sns.heatmap(ccd_farms_FRL_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[57]:


ccd_farms_FRL_new.shape


# In[58]:


ccd_farms_FRL_new.hist()


# In[59]:


ccd_farms_FRL_new.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD/Clean_ccd_farms.csv', index = False, header=True)


# #### Cleaning and checking staff file

# In[60]:


ccd_staff= pandas.read_csv("ccd_STAFF.csv")
ccd_staff.head()


# In[61]:


ccd_staff.columns


# In[62]:


ccd_staff.shape


# In[63]:


ccd_staff.describe()


# In[64]:


sns.heatmap(ccd_staff.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[65]:


null_columns=ccd_staff.columns[ccd_staff.isnull().any()]
ccd_staff[null_columns].isnull().sum()


# In[66]:


ccd_staff.drop(ccd_staff.columns[[1,2,6,7,8,13,14]], axis=1, inplace=True)


# In[67]:


ccd_staff['NCESSCH'] = ccd_staff['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[68]:


ccd_staff['NCESSCH'].is_unique


# In[69]:


ccd_staff_new = ccd_staff.dropna(axis = 0, how ='any') 


# In[70]:


print("Old data frame length:", len(ccd_staff)) 
print("New data frame length:", len(ccd_staff_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(ccd_staff)-len(ccd_staff_new))) 


# In[71]:


sns.heatmap(ccd_staff_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[72]:


ccd_staff_new.head()


# In[73]:


ccd_staff_new.shape


# In[74]:


ccd_staff_new.hist()


# In[75]:


ccd_staff_new.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD/Clean_ccd_staff.csv', index = False, header=True)


# #### Merge all files into master ccd file

# In[182]:


merged_dir_char = pandas.merge(left=ccd_directory,right=ccd_character, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_dir_char.shape


# In[183]:


merged_dir_char.columns


# In[184]:


merged_dir_char.drop([col for col in merged_dir_char.columns if col.endswith('_y')],axis=1,inplace=True)


# In[185]:


merged_dir_char.head()


# In[186]:


merged_dir_char.rename(columns = {'SCHOOL_YEAR_x':'SCHOOL_YEAR', 'ST_x':'ST', 'SCH_NAME_x':'SCH_NAME', 'STATE_AGENCY_NO_x':'STATE_AGENCY_NO', 'ST_SCHID_x':'ST_SCHID', 'SCHID_x':'SCHID' }, inplace = True) 


# In[187]:


merged_dir_char_staff = pandas.merge(left=merged_dir_char,right=ccd_staff_new, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_dir_char_staff.shape


# In[188]:


merged_dir_char_staff.columns


# In[189]:


merged_dir_char_staff.drop([col for col in merged_dir_char_staff.columns if col.endswith('_y')],axis=1,inplace=True)


# In[190]:


merged_dir_char_staff.rename(columns = {'SCHOOL_YEAR_x':'SCHOOL_YEAR', 'ST_x':'ST', 'SCH_NAME_x':'SCH_NAME', 'STATE_AGENCY_NO_x':'STATE_AGENCY_NO', 'ST_SCHID_x':'ST_SCHID', 'SCHID_x':'SCHID' }, inplace = True) 


# In[191]:


merged_dir_char_staff_farms = pandas.merge(left=merged_dir_char_staff,right=ccd_farms_FRL_new, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_dir_char_staff_farms.shape


# In[192]:


merged_dir_char_staff_farms.columns


# In[193]:


merged_dir_char_staff_farms.drop([col for col in merged_dir_char_staff_farms.columns if col.endswith('_y')],axis=1,inplace=True)


# In[203]:


merged_dir_char_staff_farms.rename(columns = {'SCHOOL_YEAR_x':'SCHOOL_YEAR', 'ST_x':'ST', 'SCH_NAME_x':'SCH_NAME', 'STATE_AGENCY_NO_x':'STATE_AGENCY_NO', 'ST_SCHID_x':'ST_SCHID', 'SCHID_x':'SCHID' }, inplace = True) 


# In[204]:


merged_dir_char_staff_farms.columns


# In[205]:


sns.heatmap(merged_dir_char_staff_farms.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[206]:


null_columns=merged_dir_char_staff_farms.columns[merged_dir_char_staff_farms.isnull().any()]
merged_dir_char_staff_farms[null_columns].isnull().sum()


# In[208]:


merged_dir_char_staff_farms.head().T


# In[209]:


merged_dir_char_staff_farms.drop(merged_dir_char_staff_farms.columns[[3,4,5,6,18,19,21,22]], axis=1, inplace=True)


# In[210]:


merged_dir_char_staff_farms.head()


# #### Filtering for only open schools 

# In[211]:


ccd_open_schools=merged_dir_char_staff_farms[merged_dir_char_staff_farms['SY_STATUS']==1]


# In[212]:


sns.heatmap(ccd_open_schools.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[213]:


ccd_master_hs=ccd_open_schools[ccd_open_schools['LEVEL']=='High']


# In[215]:


ccd_master_hs.shape


# In[214]:


sns.heatmap(ccd_master_hs.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[218]:


ccd_master_hs_new = ccd_master_hs.dropna(axis = 0, how ='any') 


# In[219]:


print("Old data frame length:", len(ccd_master_hs)) 
print("New data frame length:", len(ccd_master_hs_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(ccd_master_hs)-len(ccd_master_hs_new))) 


# In[220]:


sns.heatmap(ccd_master_hs_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[225]:


ccd_master_hs_new.shape


# In[224]:


#ccd_master_hs_new.TITLEI_STATUS	.unique() 


# In[223]:


ccd_master_hs_new.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD/Clean_ccd_master.csv', index = False, header=True)


# In[ ]:




