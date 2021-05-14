#!/usr/bin/env python
# coding: utf-8

# ### Phase 1: Part 5: Combine all clean_data files from the various data sources and create a Master data file for math proficiency

# In[1]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CCD


# In[3]:


ccd_master = pandas.read_csv("Clean_ccd_master.csv")
ccd_master['NCESSCH'] = ccd_master['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
ccd_master.head()


# In[4]:


ccd_master.shape


# In[5]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CRDC


# In[6]:


crdc_master_math = pandas.read_csv("Clean_crdc_master_math.csv")
crdc_master_math['NCESSCH'] = crdc_master_math['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
crdc_master_math.head()


# In[7]:


crdc_master_math.shape


# In[8]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/EDGE


# In[9]:


edge = pandas.read_csv("Clean_EDGE.csv")
edge['NCESSCH'] = edge['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
edge.head()


# In[10]:


edge.shape


# In[11]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/EDFacts


# In[12]:


Math_prof = pandas.read_csv("edfacts_math_merged_ccd.csv")
Math_prof['NCESSCH'] = Math_prof['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
Math_prof.head()


# In[13]:


Math_prof.shape


# ### Merge ccd and crdc file

# In[14]:


merged_ccd_crdc = pandas.merge(left=ccd_master,right=crdc_master_math, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc.shape


# In[15]:


merged_ccd_crdc.columns


# ### Merge ccd_crdc file with edge file

# In[16]:


merged_ccd_crdc_edge = pandas.merge(left=merged_ccd_crdc,right=edge, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc_edge.shape


# In[17]:


merged_ccd_crdc_edge.columns


# ### Merge ccd_crdc_edge file with edfacts file

# In[18]:


merged_ccd_crdc_edge_mathProf = pandas.merge(left=merged_ccd_crdc_edge,right=Math_prof, how='left', left_on='NCESSCH', right_on='NCESSCH')
merged_ccd_crdc_edge_mathProf.shape


# In[19]:


merged_ccd_crdc_edge_mathProf.columns


# #### Drop duplicate columns

# In[20]:


merged_ccd_crdc_edge_mathProf.drop([col for col in merged_ccd_crdc_edge_mathProf.columns if col.endswith('_y')],axis=1,inplace=True)
merged_ccd_crdc_edge_mathProf.shape


# #### Resorting columns

# In[21]:


master_math=merged_ccd_crdc_edge_mathProf[['SCHOOL_YEAR_x', 'ST_x', 'NAME', 'NCESSCH','LEVEL','SCH_TYPE_TEXT_x', 'SCH_TYPE_x', 
       'TITLEI_STATUS', 'TITLEI_STATUS_TEXT', 'TEACHERS',
       'FARMS_COUNT', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 
       'Total_enroll_students', 
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT','SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers',
       'Total_SAT_ACT_students', 
       'SCH_IBENR_IND_new', 'Total_IB_students',
       'SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new','Total_AP_math_students', 'Total_students_tookAP',
       'SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'Total_Alg1_enroll_students','Total_Alg1_pass_students', 
       'Income_Poverty_ratio', 'IPR_SE',
       'ALL_MTH00NUMVALID_1718', 'ALL_MTH00PCTPROF_1718_new']]


# In[22]:


master_math.head()


# In[23]:


sns.heatmap(master_math.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# #### Dropping rows with null values

# In[24]:


null_columns=master_math.columns[master_math.isnull().any()]
master_math[null_columns].isnull().sum()


# In[25]:


master_math_new = master_math.dropna(axis = 0, how ='any') 


# In[26]:


print("Old data frame length:", len(master_math)) 
print("New data frame length:", len(master_math_new))  
print("Number of rows with at least 1 NA value: ", 
      (len(master_math)-len(master_math_new))) 


# In[27]:


master_math_new.describe()


# In[28]:


master_math_new.shape


# In[29]:


master_math_new.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/MASTER/Master_math.csv', index = False, header=True)

