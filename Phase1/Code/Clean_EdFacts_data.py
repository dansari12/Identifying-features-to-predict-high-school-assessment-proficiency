#!/usr/bin/env python
# coding: utf-8

# ## Phase 1: Clean up of EdFacts data files

# ### Clean up the Edfacts file for reading

# In[2]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts


# In[4]:


edfacts_eng = pandas.read_csv("rla-achievement-sch-sy2017-18.csv")
edfacts_eng.head()


# In[5]:


edfacts_eng= edfacts_eng[['STNAM','FIPST','LEAID','ST_LEAID','LEANM','NCESSCH','ST_SCHID','SCHNAM','DATE_CUR','ALL_RLA00NUMVALID_1718',
                                 'ALL_RLA00PCTPROF_1718']]
edfacts_eng.head()


# In[6]:


edfacts_eng.shape


# In[7]:


edfacts_eng.rename(columns={'NCESSCH':'NCESSCH_old'}, inplace=True)


# In[8]:


edfacts_eng.dtypes


# In[9]:


edfacts_eng.describe()


# In[10]:


sns.heatmap(edfacts_eng.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[11]:


edfacts_eng.hist()


# Joining ccd_directory and edfacts datasets; using inner join default

# In[12]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD


# In[13]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# In[14]:


edfacts_eng_merged_ccd = edfacts_eng.merge(ccd_directory, on='ST_SCHID')


# In[15]:


edfacts_eng_merged_ccd.head()


# In[16]:


edfacts_eng_merged_ccd.shape


# In[17]:


edfacts_eng_merged_ccd.describe()


# In[18]:


sns.heatmap(edfacts_eng_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[56]:


edfacts_eng_merged_ccd.columns


# In[57]:


edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718_new'] = edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718'].replace(['9-Jun','14-Oct','19-Nov','15-19','20-24','20-29','21-39','24-29','25-29','30-34','30-39','35-39','40-44','40-49','40-59','45-49','50-59','50-54','55-59',
                                                                                                               '60-64','60-69','60-79','65-69','70-74','70-79','75-79','80-84','80-89','85-89','90-94','GE50','GE80','GE90','GE95','GE99','LE1','LE5','LE10','LE20',
                                                                                                              'LT50'],
                                                                                                              ['7.5','13','15','17','22','24.5','30','27','27','32','34.5','37','42','44.5','49.5','47','54.5','52','57','62','64.5','69.5','67','72','74.5','77','82',
                                                                                                               '84.5','87','92','50','80','90','95','99','1','5','10','20','49'])


# In[90]:


count_eng = edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718_new'].value_counts() 
print(count_eng) 


# In[63]:


edfacts_eng_merged_ccd.dtypes


# In[64]:


edfacts_eng_merged_ccd.shape


# In[65]:


edfacts_eng_merged_ccd_new=edfacts_eng_merged_ccd[edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718_new']!='PS' ]


# In[66]:


edfacts_eng_merged_ccd_new.shape


# In[67]:


edfacts_eng_merged_ccd_new[['ALL_RLA00PCTPROF_1718_new']] = edfacts_eng_merged_ccd_new[['ALL_RLA00PCTPROF_1718_new']].astype(float)


# In[70]:


edfacts_eng_merged_ccd_new.dtypes


# In[34]:


#edfacts_eng_merged_ccd.tail(20).T


# In[37]:


#edfacts_eng_merged_ccd['comparison_column'] = np.where(edfacts_eng_merged_ccd["SCHNAM"] == edfacts_eng_merged_ccd["SCH_NAME"], True, False)


# In[68]:


#edfacts_eng_merged_ccd['comparison_column'].describe()


# In[69]:


edfacts_eng_merged_ccd_new.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts/edfacts_eng_merged_ccd.csv', index = False, header=True)


# ### Clean up the Edfacts file for math

# In[72]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts


# In[73]:


edfacts_math = pandas.read_csv("math-achievement-sch-sy2017-18.csv")
edfacts_math.head()


# In[74]:


edfacts_math= edfacts_math[['STNAM','FIPST','LEAID','ST_LEAID','LEANM','NCESSCH','ST_SCHID','SCHNAM','DATE_CUR','ALL_MTH00NUMVALID_1718',
                                 'ALL_MTH00PCTPROF_1718']]
edfacts_math.head()


# In[75]:


edfacts_math.shape


# In[76]:


edfacts_math.rename(columns={'NCESSCH':'NCESSCH_old'}, inplace=True)


# In[77]:


edfacts_math.dtypes


# In[78]:


edfacts_math.describe()


# In[79]:


sns.heatmap(edfacts_math.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[80]:


edfacts_math.hist()


# In[81]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD


# In[82]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# In[83]:


edfacts_math_merged_ccd = edfacts_math.merge(ccd_directory, on='ST_SCHID')


# In[84]:


edfacts_math_merged_ccd.head()


# In[85]:


edfacts_math_merged_ccd.shape  #one middle school and 3 DC youth programs did not find a match in the CCD file


# In[86]:


edfacts_math_merged_ccd.describe()


# In[87]:


sns.heatmap(edfacts_math_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[88]:


edfacts_math_merged_ccd.columns


# In[89]:


edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718_new'] = edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718'].replace(['9-Jun','14-Oct','19-Nov','15-19','20-24','20-29','21-39','24-29','25-29','30-34','30-39','35-39','40-44','40-49','40-59','45-49','50-59','50-54','55-59',
                                                                                                               '60-64','60-69','60-79','65-69','70-74','70-79','75-79','80-84','80-89','85-89','90-94','GE50','GE80','GE90','GE95','GE99','LE1','LE5','LE10','LE20',
                                                                                                              'LT50'],
                                                                                                              ['7.5','13','15','17','22','24.5','30','27','27','32','34.5','37','42','44.5','49.5','47','54.5','52','57','62','64.5','69.5','67','72','74.5','77','82',
                                                                                                               '84.5','87','92','50','80','90','95','99','1','5','10','20','49'])


# In[91]:


count_math = edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718_new'].value_counts() 
print(count_math)


# In[92]:


edfacts_math_merged_ccd.dtypes


# In[93]:


edfacts_math_merged_ccd.shape


# In[94]:


edfacts_math_merged_ccd_new=edfacts_math_merged_ccd[edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718_new']!='PS' ]


# In[95]:


edfacts_math_merged_ccd_new.shape


# In[96]:


edfacts_math_merged_ccd_new[['ALL_MTH00PCTPROF_1718_new']] = edfacts_math_merged_ccd_new[['ALL_MTH00PCTPROF_1718_new']].astype(float)


# In[97]:


edfacts_math_merged_ccd_new.dtypes


# In[87]:


#edfacts_math_merged_ccd['comparison_column'] = np.where(edfacts_math_merged_ccd["SCHNAM"] == edfacts_math_merged_ccd["SCH_NAME"], True, False)


# In[93]:


#edfacts_math_merged_ccd['comparison_column'].describe()


# In[98]:


edfacts_math_merged_ccd_new.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/EDFacts/edfacts_math_merged_ccd.csv', index = False, header=True)


# In[ ]:




