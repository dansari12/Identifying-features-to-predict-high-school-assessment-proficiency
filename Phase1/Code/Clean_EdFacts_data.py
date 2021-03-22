#!/usr/bin/env python
# coding: utf-8

# Phase 1: Clean up of EdFacts data files

# Clean up the Edfacts file for reading

# In[41]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts


# In[43]:


edfacts_eng = pandas.read_csv("rla-achievement-sch-sy2017-18.csv")
edfacts_eng.head()


# In[44]:


edfacts_eng= edfacts_eng[['STNAM','FIPST','LEAID','ST_LEAID','LEANM','NCESSCH','ST_SCHID','SCHNAM','DATE_CUR','ALL_RLA00NUMVALID_1718',
                                 'ALL_RLA00PCTPROF_1718']]
edfacts_eng.head()


# In[45]:


edfacts_eng.shape


# In[46]:


edfacts_eng.rename(columns={'NCESSCH':'NCESSCH_old'}, inplace=True)


# In[47]:


edfacts_eng.dtypes


# In[48]:


edfacts_eng.describe()


# In[49]:


sns.heatmap(edfacts_eng.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[50]:


edfacts_eng.hist()


# Joining ccd_directory and edfacts datasets; using inner join default

# In[51]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD


# In[52]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# In[53]:


edfacts_eng_merged_ccd = edfacts_eng.merge(ccd_directory, on='ST_SCHID')


# In[54]:


edfacts_eng_merged_ccd.head()


# In[55]:


edfacts_eng_merged_ccd.shape


# In[56]:


edfacts_eng_merged_ccd.describe()


# In[57]:


sns.heatmap(edfacts_eng_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[59]:


edfacts_eng_merged_ccd.columns


# In[37]:


#edfacts_eng_merged_ccd['comparison_column'] = np.where(edfacts_eng_merged_ccd["SCHNAM"] == edfacts_eng_merged_ccd["SCH_NAME"], True, False)


# In[38]:


#edfacts_eng_merged_ccd['comparison_column'].describe()


# In[60]:


edfacts_eng_merged_ccd.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts/edfacts_eng_merged_ccd.csv', index = False, header=True)


# Clean up the Edfacts file for math

# In[61]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts


# In[62]:


edfacts_math = pandas.read_csv("math-achievement-sch-sy2017-18.csv")
edfacts_math.head()


# In[63]:


edfacts_math= edfacts_math[['STNAM','FIPST','LEAID','ST_LEAID','LEANM','NCESSCH','ST_SCHID','SCHNAM','DATE_CUR','ALL_MTH00NUMVALID_1718',
                                 'ALL_MTH00PCTPROF_1718']]
edfacts_math.head()


# In[65]:


edfacts_math.shape


# In[66]:


edfacts_math.rename(columns={'NCESSCH':'NCESSCH_old'}, inplace=True)


# In[67]:


edfacts_math.dtypes


# In[68]:


edfacts_math.describe()


# In[69]:


sns.heatmap(edfacts_math.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[70]:


edfacts_math.hist()


# In[71]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD


# In[72]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# In[88]:


edfacts_math_merged_ccd = edfacts_math.merge(ccd_directory, on='ST_SCHID')


# In[89]:


edfacts_math_merged_ccd.head()


# In[90]:


edfacts_math_merged_ccd.shape  #one middle school and 3 DC youth programs did not find a match in the CCD file


# In[91]:


edfacts_math_merged_ccd.describe()


# In[92]:


sns.heatmap(edfacts_math_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[94]:


edfacts_math_merged_ccd.columns


# In[87]:


#edfacts_math_merged_ccd['comparison_column'] = np.where(edfacts_math_merged_ccd["SCHNAM"] == edfacts_math_merged_ccd["SCH_NAME"], True, False)


# In[93]:


#edfacts_math_merged_ccd['comparison_column'].describe()


# In[95]:


edfacts_math_merged_ccd.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts/edfacts_math_merged_ccd.csv', index = False, header=True)

