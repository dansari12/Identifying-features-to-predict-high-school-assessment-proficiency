#!/usr/bin/env python
# coding: utf-8

# ### Combine all clean_data files from the various sources and create a Master data file for reading proficiency

# In[1]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pip install --user scikit-learn


# In[3]:


import sys
sys.path.append("/home/nbuser/.local/lib/python2.7/site-packages")


# In[4]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD


# In[5]:


ccd_master = pandas.read_csv("Clean_ccd_master.csv")
ccd_master['NCESSCH'] = ccd_master['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
ccd_master.head()


# In[6]:


ccd_master.shape


# In[7]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC


# In[8]:


crdc_master_read = pandas.read_csv("Clean_crdc_master_read.csv")
crdc_master_read['NCESSCH'] = crdc_master_read['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
crdc_master_read.head()


# In[9]:


crdc_master_read.shape


# In[10]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDGE


# In[11]:


edge = pandas.read_csv("Clean_EDGE.csv")
edge['NCESSCH'] = edge['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
edge.head()


# In[12]:


edge.shape


# In[13]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts


# In[14]:


Eng_prof = pandas.read_csv("edfacts_eng_merged_ccd.csv")
Eng_prof['NCESSCH'] = Eng_prof['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
Eng_prof.head()


# In[15]:


Eng_prof.shape


# ### Merge ccd and crdc file

# In[16]:


merged_ccd_crdc = pandas.merge(left=ccd_master,right=crdc_master_read, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc.shape


# In[17]:


merged_ccd_crdc.columns


# In[18]:


#merged_ccd_crdc.head().T


# In[19]:


merged_ccd_crdc_edge = pandas.merge(left=merged_ccd_crdc,right=edge, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc_edge.shape


# In[20]:


merged_ccd_crdc_edge.columns


# In[21]:


#merged_ccd_crdc_edge.head().T


# In[22]:


merged_ccd_crdc_edge_engProf = pandas.merge(left=merged_ccd_crdc_edge,right=Eng_prof, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc_edge_engProf.shape


# In[23]:


merged_ccd_crdc_edge_engProf.columns


# In[24]:


merged_ccd_crdc_edge_engProf.drop([col for col in merged_ccd_crdc_edge_engProf.columns if col.endswith('_y')],axis=1,inplace=True)
merged_ccd_crdc_edge_engProf.shape


# In[25]:


merged_ccd_crdc_edge_engProf.columns


# In[26]:


master_reading=merged_ccd_crdc_edge_engProf[['SCHOOL_YEAR_x', 'ST_x','NAME', 'NCESSCH', 'LEVEL', 'SCH_TYPE_TEXT_x', 'SCH_TYPE_x',
       'TITLEI_STATUS', 'TITLEI_STATUS_TEXT', 'MAGNET_TEXT', 'TEACHERS',
       'FARMS_COUNT', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new',
       'Total_enroll_students', 
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT','SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers',
       'Total_SAT_ACT_students', 
       'SCH_IBENR_IND_new', 'Total_IB_students',
       'SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APOTHENR_IND_new','Total_AP_other_students', 'Total_students_tookAP', 
       'Income_Poverty_ratio', 'IPR_SE', 
       'ALL_RLA00NUMVALID_1718','ALL_RLA00PCTPROF_1718_new']]


# In[27]:


master_reading.shape


# In[28]:


master_reading.head().T


# In[29]:


sns.heatmap(master_reading.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[30]:


null_columns=master_reading.columns[master_reading.isnull().any()]
master_reading[null_columns].isnull().sum()


# In[31]:


master_reading_new = master_reading.dropna(axis = 0, how ='any') 


# In[32]:


print("Old data frame length:", len(master_reading)) 
print("New data frame length:", len(master_reading_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(master_reading)-len(master_reading_new))) 


# In[33]:


sns.heatmap(master_reading_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[34]:


master_reading_new.describe()


# In[38]:


master_reading_new.shape


# In[37]:


master_reading_new.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/MASTER/Master_reading.csv', index = False, header=True)

