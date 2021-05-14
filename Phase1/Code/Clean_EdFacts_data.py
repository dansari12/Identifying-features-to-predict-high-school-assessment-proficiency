#!/usr/bin/env python
# coding: utf-8

# ## Phase 1: Part 4: Clean up of EdFacts data files. This notebook contains code that cleans and preps the edfacts data files for reading and math proficiency scores.

# ### Clean up the Edfacts file for reading

# ### Loading necessary libraries

# In[41]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/EDFacts


# In[43]:


edfacts_eng = pandas.read_csv("rla-achievement-sch-sy2017-18.csv")
edfacts_eng.head()


# In[44]:


#selecting only relevant columns
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


# #### Checking for missing or null values

# In[49]:


sns.heatmap(edfacts_eng.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[50]:


edfacts_eng.hist()


# #### Joining ccd_directory and edfacts datasets.
# ##### The NCESSCH ID was not populated correctly in the raw edfacts data files. Some of the IDs are in Exponential formats and even are attempting to convert them to long strings the ID are not populating correctly. So instead, I will perform an inner join using the ST_SCHID and the ccd_directory file that contains the both the ST_SCHID and NCESSCH ID so that we can later merge all the files using just the NCESSCH id

# In[51]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CCD


# In[52]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# In[53]:


edfacts_eng_merged_ccd = edfacts_eng.merge(ccd_directory, on='ST_SCHID')


# #### Verifying that the merge was successful and that the correct school ID are reflect for each school. Lets look at the school names to confirm this

# In[54]:


edfacts_eng_merged_ccd.head().T


# #### School names match all is good

# In[55]:


edfacts_eng_merged_ccd.shape


# In[56]:


edfacts_eng_merged_ccd.describe()


# In[57]:


sns.heatmap(edfacts_eng_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[58]:


edfacts_eng_merged_ccd.columns


# #### Let's take a look at the percent proficiency scores to make sure they look okay

# In[59]:


edfacts_eng_merged_ccd.tail(20).T


# In[60]:


#edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718'].value_counts(ascending=True)


# #### The percents don't look right because the feds have applied suppression and converted some percents to ranges for privacy reasons. Refer to the EdFacts documentation folder for more info: /Phase1/Data Documentation/EdFacts/assessments-sy2017-18-public-file-documentation.docx
# 
# ###### I am going to replace the ranges with the median value for each range, so we have a continuous variable to work with. Also note, some of the ranges are showing up as dates, I was able to identify the ranges by look at the documentation.For any values that is less than or greater than, I replaced the percent with the next highest value(eg. LT50 replaced with 49). And for any value greater/less than or equal to, I replaced it with the equal to values (eg. LE50 replaced with 50).

# In[61]:


edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718_new'] = edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718'].replace(['9-Jun','14-Oct','19-Nov','15-19','20-24','20-29','21-39','24-29','25-29','30-34','30-39','35-39','40-44','40-49','40-59','45-49','50-59','50-54','55-59',
                                                                                                               '60-64','60-69','60-79','65-69','70-74','70-79','75-79','80-84','80-89','85-89','90-94','GE50','GE80','GE90','GE95','GE99','LE1','LE5','LE10','LE20',
                                                                                                              'LT50'],
                                                                                                              ['7.5','13','15','17','22','24.5','30','27','27','32','34.5','37','42','44.5','49.5','47','54.5','52','57','62','64.5','69.5','67','72','74.5','77','82',
                                                                                                               '84.5','87','92','50','80','90','95','99','1','5','10','20','49'])


# In[62]:


count_eng = edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718_new'].value_counts() 
print(count_eng) 


# In[63]:


edfacts_eng_merged_ccd.dtypes


# In[64]:


edfacts_eng_merged_ccd.shape


# #### Removing any schools were the percents were entirely suppressed.

# In[65]:


edfacts_eng_merged_ccd_new=edfacts_eng_merged_ccd[edfacts_eng_merged_ccd['ALL_RLA00PCTPROF_1718_new']!='PS' ]


# In[66]:


edfacts_eng_merged_ccd_new.shape


# #### Converting dtype to float for easier analysis later

# In[67]:


edfacts_eng_merged_ccd_new[['ALL_RLA00PCTPROF_1718_new']] = edfacts_eng_merged_ccd_new[['ALL_RLA00PCTPROF_1718_new']].astype(float)


# In[68]:


edfacts_eng_merged_ccd_new.dtypes


# In[69]:


#edfacts_eng_merged_ccd.tail(20).T


# In[70]:


#edfacts_eng_merged_ccd['comparison_column'] = np.where(edfacts_eng_merged_ccd["SCHNAM"] == edfacts_eng_merged_ccd["SCH_NAME"], True, False)


# In[71]:


#edfacts_eng_merged_ccd['comparison_column'].describe()


# In[72]:


edfacts_eng_merged_ccd_new.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/EDFacts/edfacts_eng_merged_ccd.csv', index = False, header=True)


# ### Clean up the Edfacts file for math

# In[75]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/EDFacts


# In[76]:


edfacts_math = pandas.read_csv("math-achievement-sch-sy2017-18.csv")
edfacts_math.head()


# In[77]:


edfacts_math= edfacts_math[['STNAM','FIPST','LEAID','ST_LEAID','LEANM','NCESSCH','ST_SCHID','SCHNAM','DATE_CUR','ALL_MTH00NUMVALID_1718',
                                 'ALL_MTH00PCTPROF_1718']]
edfacts_math.head()


# In[78]:


edfacts_math.shape


# In[79]:


edfacts_math.rename(columns={'NCESSCH':'NCESSCH_old'}, inplace=True)


# In[80]:


edfacts_math.dtypes


# In[81]:


edfacts_math.describe()


# In[82]:


sns.heatmap(edfacts_math.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[83]:


edfacts_math.hist()


# In[84]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CCD


# In[85]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# #### Joining ccd_directory and edfacts datasets.

# In[86]:


edfacts_math_merged_ccd = edfacts_math.merge(ccd_directory, on='ST_SCHID')


# In[87]:


edfacts_math_merged_ccd.head()


# In[88]:


edfacts_math_merged_ccd.shape  #one middle school and 3 DC youth programs did not find a match in the CCD file


# In[89]:


edfacts_math_merged_ccd.describe()


# In[90]:


sns.heatmap(edfacts_math_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[91]:


edfacts_math_merged_ccd.columns


# #### The percents don't look right because the feds have applied suppression and converted some percents to ranges for privacy reasons. Refer to the EdFacts documentation folder for more info: /Phase1/Data Documentation/EdFacts/assessments-sy2017-18-public-file-documentation.docx. Applied same transformations as those applied to the reading file.

# In[92]:


edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718_new'] = edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718'].replace(['9-Jun','14-Oct','19-Nov','15-19','20-24','20-29','21-39','24-29','25-29','30-34','30-39','35-39','40-44','40-49','40-59','45-49','50-59','50-54','55-59',
                                                                                                               '60-64','60-69','60-79','65-69','70-74','70-79','75-79','80-84','80-89','85-89','90-94','GE50','GE80','GE90','GE95','GE99','LE1','LE5','LE10','LE20',
                                                                                                              'LT50'],
                                                                                                              ['7.5','13','15','17','22','24.5','30','27','27','32','34.5','37','42','44.5','49.5','47','54.5','52','57','62','64.5','69.5','67','72','74.5','77','82',
                                                                                                               '84.5','87','92','50','80','90','95','99','1','5','10','20','49'])


# In[93]:


count_math = edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718_new'].value_counts() 
print(count_math)


# In[94]:


edfacts_math_merged_ccd.dtypes


# In[95]:


edfacts_math_merged_ccd.shape


# In[96]:


edfacts_math_merged_ccd_new=edfacts_math_merged_ccd[edfacts_math_merged_ccd['ALL_MTH00PCTPROF_1718_new']!='PS' ]


# In[97]:


edfacts_math_merged_ccd_new.shape


# In[98]:


edfacts_math_merged_ccd_new[['ALL_MTH00PCTPROF_1718_new']] = edfacts_math_merged_ccd_new[['ALL_MTH00PCTPROF_1718_new']].astype(float)


# In[99]:


edfacts_math_merged_ccd_new.dtypes


# In[100]:


#edfacts_math_merged_ccd['comparison_column'] = np.where(edfacts_math_merged_ccd["SCHNAM"] == edfacts_math_merged_ccd["SCH_NAME"], True, False)


# In[101]:


#edfacts_math_merged_ccd['comparison_column'].describe()


# #### Saving file for later use

# In[102]:


edfacts_math_merged_ccd_new.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/EDFacts/edfacts_math_merged_ccd.csv', index = False, header=True)

