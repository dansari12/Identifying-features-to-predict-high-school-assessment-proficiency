#!/usr/bin/env python
# coding: utf-8

# ### Phase 1: Part 6: Combine all clean_data files from the various sources and create a Master data file for reading proficiency

# In[38]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CCD


# In[40]:


ccd_master = pandas.read_csv("Clean_ccd_master.csv")
ccd_master['NCESSCH'] = ccd_master['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
ccd_master.head()


# In[41]:


ccd_master.shape


# In[42]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CRDC


# In[43]:


crdc_master_read = pandas.read_csv("Clean_crdc_master_read.csv")
crdc_master_read['NCESSCH'] = crdc_master_read['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
crdc_master_read.head()


# In[44]:


crdc_master_read.shape


# In[45]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/EDGE


# In[46]:


edge = pandas.read_csv("Clean_EDGE.csv")
edge['NCESSCH'] = edge['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
edge.head()


# In[47]:


edge.shape


# In[48]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/EDFacts


# In[49]:


Eng_prof = pandas.read_csv("edfacts_eng_merged_ccd.csv")
Eng_prof['NCESSCH'] = Eng_prof['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
Eng_prof.head()


# In[50]:


Eng_prof.shape


# ### Merge ccd and crdc file

# In[51]:


merged_ccd_crdc = pandas.merge(left=ccd_master,right=crdc_master_read, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc.shape


# In[52]:


merged_ccd_crdc.columns


# In[53]:


#merged_ccd_crdc.head().T


# ### Merge ccd_crdc file with edge file

# In[54]:


merged_ccd_crdc_edge = pandas.merge(left=merged_ccd_crdc,right=edge, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc_edge.shape


# In[55]:


merged_ccd_crdc_edge.columns


# In[56]:


#merged_ccd_crdc_edge.head().T


# ### Merge ccd_crdc_edge file with edfacts file

# In[57]:


merged_ccd_crdc_edge_engProf = pandas.merge(left=merged_ccd_crdc_edge,right=Eng_prof, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc_edge_engProf.shape


# In[58]:


merged_ccd_crdc_edge_engProf.columns


# #### Drop duplicate columns

# In[59]:


merged_ccd_crdc_edge_engProf.drop([col for col in merged_ccd_crdc_edge_engProf.columns if col.endswith('_y')],axis=1,inplace=True)
merged_ccd_crdc_edge_engProf.shape


# In[60]:


merged_ccd_crdc_edge_engProf.columns


# #### Resorting columns

# In[61]:


master_reading=merged_ccd_crdc_edge_engProf[['SCHOOL_YEAR_x', 'ST_x','NAME', 'NCESSCH', 'LEVEL', 'SCH_TYPE_TEXT_x', 'SCH_TYPE_x',
       'TITLEI_STATUS', 'TITLEI_STATUS_TEXT', 'TEACHERS',
       'FARMS_COUNT', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new',
       'Total_enroll_students', 
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT','SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers',
       'Total_SAT_ACT_students', 
       'SCH_IBENR_IND_new', 'Total_IB_students',
       'SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APOTHENR_IND_new','Total_AP_other_students', 'Total_students_tookAP', 
       'Income_Poverty_ratio', 'IPR_SE', 
       'ALL_RLA00NUMVALID_1718','ALL_RLA00PCTPROF_1718_new']]


# In[62]:


master_reading.shape


# In[63]:


master_reading.head()


# In[64]:


sns.heatmap(master_reading.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# #### Dropping rows with null values

# In[65]:


null_columns=master_reading.columns[master_reading.isnull().any()]
master_reading[null_columns].isnull().sum()


# In[66]:


master_reading_new = master_reading.dropna(axis = 0, how ='any') 


# In[67]:


print("Old data frame length:", len(master_reading)) 
print("New data frame length:", len(master_reading_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(master_reading)-len(master_reading_new))) 


# In[68]:


sns.heatmap(master_reading_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[69]:


master_reading_new.describe()


# In[70]:


master_reading_new.shape


# In[71]:


master_reading_new.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/MASTER/Master_reading.csv', index = False, header=True)

