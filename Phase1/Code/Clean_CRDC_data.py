#!/usr/bin/env python
# coding: utf-8

# ## Phase 1: Clean up of CRDC data files

# In[1]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC


# ### Cleaning school characteristics file

# In[3]:


Sch_char = pandas.read_csv("School Characteristics.csv",encoding='cp1252')
Sch_char.head()


# In[4]:


Sch_char['SCHID'] = Sch_char['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[5]:


Sch_char['LEAID'] = Sch_char['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[6]:


Sch_char.columns


# In[7]:


Sch_char.drop(Sch_char.columns[[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,31]], axis=1, inplace=True)


# In[8]:


Sch_char.shape


# In[9]:


#Sch_char.head()


# In[10]:


cols = ['LEAID', 'SCHID']
Sch_char['NCESSCH'] = Sch_char[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[11]:


Sch_char['NCESSCH'].is_unique


# In[12]:


Sch_char.rename(columns={'SCH_STATUS_SPED':'Special_ed_schl','SCH_STATUS_MAGNET':'Magnet_schl','SCH_STATUS_CHARTER':'Charter_Schl','SCH_STATUS_ALT':'Alternate_schl'}, inplace=True)


# In[13]:


Sch_char.head()


# In[14]:


count = Sch_char['Charter_Schl'].value_counts() 
print(count) 


# In[15]:


Sch_char['Special_ed_schl_new'] = Sch_char['Special_ed_schl'].replace(['Yes','No'],['1','0'])


# In[16]:


Sch_char['Magnet_schl_new'] = Sch_char['Magnet_schl'].replace(['Yes','No'],['1','0'])


# In[17]:


Sch_char['Charter_Schl_new'] = Sch_char['Charter_Schl'].replace(['Yes','No'],['1','0'])


# In[18]:


Sch_char['Alternate_schl_new'] = Sch_char['Alternate_schl'].replace(['Yes','No'],['1','0'])


# In[19]:


Sch_char[['Special_ed_schl_new', 'Magnet_schl_new','Charter_Schl_new','Alternate_schl_new']] = Sch_char[['Special_ed_schl_new', 'Magnet_schl_new','Charter_Schl_new','Alternate_schl_new']].astype(int)


# In[20]:


sns.heatmap(Sch_char.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[21]:


Sch_char.describe()


# In[22]:


Sch_char.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_schlcharacteristics.csv', index = False, header=True)


# ### Cleaning school expenditure file

# In[23]:


Sch_exp = pandas.read_csv("School Expenditures.csv", encoding='cp1252')
Sch_exp.tail()


# In[24]:


Sch_exp['SCHID'] = Sch_exp['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[25]:


Sch_exp['LEAID'] = Sch_exp['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[26]:


Sch_exp.columns


# In[27]:


Sch_exp.drop(Sch_exp.columns[[7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]], axis=1, inplace=True)


# In[28]:


Sch_exp.head()


# In[29]:


cols = ['LEAID', 'SCHID']
Sch_exp['NCESSCH'] = Sch_exp[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[30]:


Sch_exp.shape


# In[31]:


Sch_exp['NCESSCH'].is_unique


# In[32]:


Sch_exp.rename(columns={'SCH_FTE_TEACH_WOFED':'FTE_teachers_count','SCH_SAL_TEACH_WOFED':'SalaryforTeachers'}, inplace=True)


# In[33]:


Sch_exp.head()


# In[34]:


#Sch_exp['Teacher_salary_ratio'] = (Sch_exp['SalaryforTeachers'] / Sch_exp['FTE_teachers_count'])


# In[35]:


sns.heatmap(Sch_exp.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[36]:


Sch_exp.describe()


# #### Dropping columns with zero or less than zero Salary expenditures

# In[37]:


Sch_exp_clean= Sch_exp[Sch_exp.SalaryforTeachers > 0]


# In[38]:


Sch_exp_clean.shape


# In[39]:


Sch_exp_clean.describe()


# In[40]:


Sch_exp_clean.head()


# In[41]:


Sch_exp_clean.hist()


# In[42]:


Sch_exp_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_schlexpenses.csv', index = False, header=True)


# ### Cleaning school support file

# In[43]:


Sch_sup= pandas.read_csv("School Support.csv",encoding='cp1252')
Sch_sup.head()


# In[44]:


Sch_sup['SCHID'] = Sch_sup['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[45]:


Sch_sup['LEAID'] = Sch_sup['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[46]:


Sch_sup.columns


# In[47]:


Sch_sup.head()


# In[48]:


Sch_sup.drop(Sch_sup.columns[[7,11,12,13,14,15,16,17,18,19,20,21]], axis=1, inplace=True)


# In[49]:


Sch_sup.head()


# In[50]:


cols = ['LEAID', 'SCHID']
Sch_sup['NCESSCH'] = Sch_sup[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[51]:


Sch_sup.shape


# In[52]:


Sch_sup['NCESSCH'].is_unique


# In[53]:


sns.heatmap(Sch_sup.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[54]:


Sch_sup.describe()


# #### Filtering FTE count GT 1 and Cert count GT -5

# In[55]:


Sch_sup_FTEGT1= Sch_sup[Sch_sup.SCH_FTETEACH_TOT > 0]


# In[56]:


Sch_sup_clean= Sch_sup_FTEGT1[Sch_sup_FTEGT1.SCH_FTETEACH_CERT > -5]


# In[57]:


Sch_sup_clean.shape


# In[58]:


Sch_sup_clean.head()


# In[59]:


Sch_sup_clean.describe()


# In[60]:


Sch_sup_clean.hist()


# In[61]:


Sch_sup_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_schlsupport.csv', index = False, header=True)


# ### Cleaning SAT and ACT file

# In[62]:


SAT_ACT = pandas.read_csv("SAT and ACT.csv", encoding='cp1252')
SAT_ACT.head()


# In[63]:


SAT_ACT['LEAID'] = SAT_ACT['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[64]:


SAT_ACT['SCHID'] = SAT_ACT['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[65]:


SAT_ACT.columns


# In[66]:


SAT_ACT.shape


# In[67]:


SAT_ACT.drop(SAT_ACT.columns[[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27]], axis=1, inplace=True)


# In[68]:


SAT_ACT.head()


# In[69]:


cols = ['LEAID', 'SCHID']
SAT_ACT['NCESSCH'] = SAT_ACT[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# #### Adding total count of male and female participation on ACT and SAT

# In[70]:


SAT_ACT.rename(columns={'TOT_SATACT_M':'Male_part_count','TOT_SATACT_F':'Female_part_count'}, inplace=True)


# In[71]:


SAT_ACT['Total_SAT_ACT_students'] = (SAT_ACT['Male_part_count'] + SAT_ACT['Female_part_count'])


# In[72]:


SAT_ACT.head()


# In[73]:


SAT_ACT.describe()


# #### Keeping total counts GT 0

# In[74]:


SAT_ACT_clean= SAT_ACT[SAT_ACT.Total_SAT_ACT_students > 0]


# In[75]:


SAT_ACT_clean.shape


# In[76]:


SAT_ACT_clean.head()


# In[77]:


sns.heatmap(SAT_ACT.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[78]:


SAT_ACT_clean.describe()


# In[79]:


SAT_ACT_clean.hist()


# In[80]:


SAT_ACT_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_SAT_ACT.csv', index = False, header=True)


# ### Cleaning IB file

# In[81]:


IB= pandas.read_csv("International Baccalaureate.csv",encoding='cp1252')
IB.head()


# In[82]:


IB['SCHID'] = IB['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[83]:


IB['LEAID'] = IB['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[84]:


IB.columns


# In[85]:


IB.shape


# In[86]:


IB.drop(IB.columns[[7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28]], axis=1, inplace=True)


# In[87]:


IB.head()


# In[88]:


cols = ['LEAID', 'SCHID']
IB['NCESSCH'] = IB[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[89]:


IB.rename(columns={'TOT_IBENR_M':'Male_enroll_count','TOT_IBENR_F':'Female_enroll_count'}, inplace=True)


# In[90]:


IB['Total_IB_students'] = (IB['Male_enroll_count'] + IB['Female_enroll_count'])


# #### Keeping IB program indicator with Y/N

# In[91]:


IB_clean= IB[IB.SCH_IBENR_IND != '-9']


# In[92]:


IB_clean.shape


# In[93]:


IB_clean.dtypes


# In[94]:


IB_clean['SCH_IBENR_IND_new'] = IB_clean['SCH_IBENR_IND'].replace(['Yes','No'],['1','0'])


# In[95]:


IB_clean[['SCH_IBENR_IND_new']]=IB_clean[['SCH_IBENR_IND_new']].astype(int)


# In[96]:


IB_clean.head()


# In[97]:


sns.heatmap(IB_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[98]:


IB_clean.describe()


# In[99]:


IB_clean.hist()


# In[100]:


IB_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_IB.csv', index = False, header=True)


# ### Cleaning AP file

# In[101]:


AP = pandas.read_csv("Advanced Placement.csv",encoding='cp1252')
AP.head()


# In[102]:


AP['SCHID'] = AP['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[103]:


AP['LEAID'] = AP['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[104]:


AP.columns


# In[105]:


AP.shape


# In[106]:


AP=AP[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','SCH_APENR_IND','SCH_APCOURSES','SCH_APMATHENR_IND','TOT_APMATHENR_M','TOT_APMATHENR_F','SCH_APOTHENR_IND','TOT_APOTHENR_M','TOT_APOTHENR_F','TOT_APEXAM_ONEORMORE_M','TOT_APEXAM_ONEORMORE_F']]


# In[107]:


AP.shape


# In[108]:


AP.head()


# In[109]:


cols = ['LEAID', 'SCHID']
AP['NCESSCH'] = AP[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[110]:


AP.rename(columns={'TOT_APMATHENR_M':'Male_enroll_math_count','TOT_APMATHENR_F':'Female_enroll_math_count','TOT_APOTHENR_M':'Male_enroll_other_count','TOT_APOTHENR_F':'Female_enroll_other_count'}, inplace=True)


# In[111]:


AP['Total_AP_math_students'] = (AP['Male_enroll_math_count'] + AP['Female_enroll_math_count'])


# In[112]:


AP['Total_AP_other_students'] = (AP['Male_enroll_other_count'] + AP['Female_enroll_other_count'])


# In[113]:


AP['Total_students_tookAP'] = (AP['TOT_APEXAM_ONEORMORE_M'] + AP['TOT_APEXAM_ONEORMORE_F'])


# In[114]:


AP.columns


# In[115]:


AP_math=AP[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','SCH_APENR_IND', 'SCH_APCOURSES', 'SCH_APMATHENR_IND',
       'Total_AP_math_students','Total_students_tookAP']]


# In[116]:


AP_math_clean= AP_math[AP_math.SCH_APENR_IND.isin(['Yes','No'])]


# In[117]:


AP_math_clean.shape


# In[118]:


AP_math_clean.dtypes


# In[119]:


AP_math_clean['SCH_APENR_IND_new'] = AP_math_clean['SCH_APENR_IND'].replace(['Yes','No'],['1','0'])


# In[120]:


AP_math_clean[['SCH_APENR_IND_new']] = AP_math_clean[['SCH_APENR_IND_new']].astype(int)


# In[121]:


AP_math_clean['SCH_APMATHENR_IND_new'] = AP_math_clean['SCH_APMATHENR_IND'].replace(['Yes','No'],['1','0'])


# In[122]:


AP_math_clean[['SCH_APMATHENR_IND_new']] = AP_math_clean[['SCH_APMATHENR_IND_new']].astype(int)


# In[123]:


AP_math_clean.dtypes


# In[124]:


sns.heatmap(AP_math_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[125]:


AP_math_clean.describe()


# In[126]:


AP_math_clean.hist()


# In[127]:


AP_math_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_AP_math.csv', index = False, header=True)


# In[128]:


AP_other=AP[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','SCH_APENR_IND', 'SCH_APCOURSES', 'SCH_APOTHENR_IND',
       'Total_AP_other_students', 'Total_students_tookAP']]


# In[129]:


AP_other_clean= AP_other[AP_other.SCH_APENR_IND.isin(['Yes','No'])]


# In[130]:


AP_other_clean.shape


# In[131]:


AP_other_clean.dtypes


# In[132]:


count_other = AP_other_clean['SCH_APOTHENR_IND'].value_counts() 
print(count_other) 


# In[133]:


AP_other_clean['SCH_APENR_IND_new'] = AP_other_clean['SCH_APENR_IND'].replace(['Yes','No'],['1','0'])


# In[134]:


AP_other_clean[['SCH_APENR_IND_new']] = AP_other_clean[['SCH_APENR_IND_new']].astype(int)


# In[135]:


AP_other_clean['SCH_APOTHENR_IND_new'] = AP_other_clean['SCH_APOTHENR_IND'].replace(['Yes','No'],['1','0'])


# In[136]:


AP_other_clean[['SCH_APOTHENR_IND_new']] = AP_other_clean[['SCH_APOTHENR_IND_new']].astype(int)


# In[137]:


AP_other_clean.dtypes


# In[138]:


sns.heatmap(AP_other_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[139]:


AP_other_clean.describe()


# In[140]:


AP_other_clean.hist()


# In[141]:


AP_other_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_AP_other.csv', index = False, header=True)


# ### Cleaning Algebra 1 file

# In[142]:


Alg1 = pandas.read_csv("Algebra I.csv",encoding='cp1252')
Alg1.head()


# In[143]:


Alg1['SCHID'] = Alg1['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[144]:


Alg1['LEAID'] = Alg1['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[145]:


Alg1.columns


# In[146]:


Alg1.shape


# In[147]:


Alg1=Alg1[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','SCH_MATHCLASSES_ALG','SCH_MATHCERT_ALG','TOT_ALGENR_GS0910_M',
           'TOT_ALGENR_GS0910_F','TOT_ALGENR_GS1112_M','TOT_ALGENR_GS1112_F','TOT_ALGPASS_GS0910_M','TOT_ALGPASS_GS0910_F','TOT_ALGPASS_GS1112_M','TOT_ALGPASS_GS1112_F']]


# In[148]:


Alg1.shape


# In[149]:


Alg1.head()


# In[150]:


cols = ['LEAID', 'SCHID']
Alg1['NCESSCH'] = Alg1[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[151]:


Alg1.rename(columns={'TOT_ALGENR_GS0910_M':'Male_enroll_9to10_count','TOT_ALGENR_GS0910_F':'Female_enroll_9to10_count','TOT_ALGENR_GS1112_M':'Male_enroll_11to12_count',
                   'TOT_ALGENR_GS1112_F':'Female_enroll_11to12_count','TOT_ALGPASS_GS0910_M':'Male_pass_9to10_count','TOT_ALGPASS_GS0910_F':'Female_pass_9to10_count',
                  'TOT_ALGPASS_GS1112_M':'Male_pass_11to12_count','TOT_ALGPASS_GS1112_F':'Female_pass_11to12_count'}, inplace=True)


# In[152]:


Alg1.columns


# In[153]:


Alg1['Total_Alg1_enroll_students'] = (Alg1['Male_enroll_9to10_count'] + Alg1['Female_enroll_9to10_count'] + Alg1['Male_enroll_11to12_count'] + Alg1['Female_enroll_11to12_count'])


# In[154]:


Alg1['Total_Alg1_pass_students'] = (Alg1['Male_pass_9to10_count'] + Alg1['Female_pass_9to10_count'] + Alg1['Male_pass_11to12_count'] + Alg1['Female_pass_11to12_count'])


# In[155]:


Alg1=Alg1[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME',
       'COMBOKEY', 'SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'NCESSCH',
       'Total_Alg1_enroll_students', 'Total_Alg1_pass_students']]


# In[156]:


Alg1_clean= Alg1[Alg1.SCH_MATHCLASSES_ALG > 0]


# In[157]:


Alg1_clean.shape


# In[158]:


sns.heatmap(Alg1_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[159]:


Alg1_clean.describe()


# In[160]:


Alg1_clean.hist()


# In[161]:


Alg1_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_Alg1.csv', index = False, header=True)


# ### Cleaning Algebra 2 file

# In[162]:


Alg2 = pandas.read_csv("Algebra II.csv",encoding='cp1252')
Alg2.head()


# In[163]:


Alg2['SCHID'] = Alg2['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[164]:


Alg2['LEAID'] = Alg2['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[165]:


Alg2.columns


# In[166]:


Alg2.shape


# In[167]:


Alg2=Alg2[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','SCH_MATHCLASSES_ALG2', 'SCH_MATHCERT_ALG2','TOT_MATHENR_ALG2_M',
       'TOT_MATHENR_ALG2_F']]


# In[168]:


Alg2.shape


# In[169]:


Alg2.head()


# In[170]:


cols = ['LEAID', 'SCHID']
Alg2['NCESSCH'] = Alg2[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[171]:


Alg2.rename(columns={'TOT_MATHENR_ALG2_M':'Male_enroll_Alg2_count','TOT_MATHENR_ALG2_F':'Female_enroll_Alg2_count'}, inplace=True)


# In[172]:


Alg2.columns


# In[173]:


Alg2['Total_Alg2_enroll_students'] = (Alg2['Male_enroll_Alg2_count'] + Alg2['Female_enroll_Alg2_count'])


# In[174]:


Alg2=Alg2[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME',
       'COMBOKEY', 'SCH_MATHCLASSES_ALG2', 'SCH_MATHCERT_ALG2', 'NCESSCH',
       'Total_Alg2_enroll_students']]


# In[175]:


Alg2_clean= Alg2[Alg2.SCH_MATHCLASSES_ALG2 > 0]


# In[176]:


Alg2_clean.shape


# In[177]:


sns.heatmap(Alg2_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[178]:


Alg2_clean.describe()


# In[179]:


Alg2_clean.hist()


# In[180]:


Alg2_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_Alg2.csv', index = False, header=True)


# ### Cleaning Calculus file

# In[181]:


Calculus = pandas.read_csv("Calculus.csv",encoding='cp1252')
Calculus.head()


# In[182]:


Calculus['SCHID'] = Calculus['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[183]:


Calculus['LEAID'] = Calculus['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[184]:


Calculus.columns


# In[185]:


Calculus.shape


# In[186]:


Calculus=Calculus[[' "LEA_STATE"', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','SCH_MATHCLASSES_CALC','SCH_MATHCERT_CALC','TOT_MATHENR_CALC_M','TOT_MATHENR_CALC_F']]


# In[187]:


Calculus.shape


# In[188]:


cols = ['LEAID', 'SCHID']
Calculus['NCESSCH'] = Calculus[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[189]:


Calculus['Total_Calc_enroll_students'] = (Calculus['TOT_MATHENR_CALC_M'] + Calculus['TOT_MATHENR_CALC_M'])


# In[190]:


Calculus.columns


# In[191]:


Calculus=Calculus[[' "LEA_STATE"', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','SCH_MATHCLASSES_CALC','SCH_MATHCERT_CALC','Total_Calc_enroll_students']]


# In[192]:


Calculus_clean=Calculus[Calculus.SCH_MATHCLASSES_CALC > 0]


# In[193]:


Calculus_clean.shape


# In[194]:


sns.heatmap(Calculus_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[195]:


Calculus_clean.describe()


# In[196]:


Calculus_clean.hist()


# In[197]:


Calculus_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_calculus.csv', index = False,header=True)


# ### Cleaning Geometry file

# In[198]:


Geometry = pandas.read_csv("Geometry.csv",encoding='cp1252')
Geometry.head()


# In[199]:


Geometry['SCHID'] = Geometry['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[200]:


Geometry['LEAID'] = Geometry['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[201]:


Geometry.columns


# In[202]:


Geometry.shape


# In[203]:


Geometry=Geometry[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','SCH_MATHCERT_GEOM','SCH_MATHCLASSES_GEOM','TOT_MATHENR_GEOM_M','TOT_MATHENR_GEOM_F']]


# In[204]:


Geometry.shape


# In[205]:


cols = ['LEAID', 'SCHID']
Geometry['NCESSCH'] = Geometry[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[206]:


Geometry['Total_Geomty_enroll_students'] = (Geometry['TOT_MATHENR_GEOM_M'] + Geometry['TOT_MATHENR_GEOM_F'])


# In[207]:


Geometry.columns


# In[208]:


Geometry=Geometry[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','SCH_MATHCERT_GEOM','SCH_MATHCLASSES_GEOM','Total_Geomty_enroll_students']]


# In[209]:


Geometry_clean=Geometry[Geometry.SCH_MATHCLASSES_GEOM > 0]


# In[210]:


sns.heatmap(Geometry_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[211]:


Geometry_clean.describe()


# In[212]:


Geometry_clean.hist()


# In[213]:


Geometry_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_geometry.csv', index = False,header=True)


# ### Cleaning Enrollment file

# In[214]:


Enroll = pandas.read_csv("Enrollment.csv",encoding='cp1252')
Enroll.head()


# In[215]:


Enroll['SCHID'] = Enroll['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[216]:


Enroll['LEAID'] = Enroll['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[217]:


Enroll.columns


# In[218]:


Enroll.shape


# In[219]:


Enroll=Enroll[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','TOT_ENR_M','TOT_ENR_F']]


# In[220]:


Enroll.shape


# In[221]:


cols = ['LEAID', 'SCHID']
Enroll['NCESSCH'] = Enroll[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[222]:


Enroll['Total_enroll_students'] = (Enroll['TOT_ENR_M'] + Enroll['TOT_ENR_F'])


# In[223]:


Enroll.columns


# In[224]:


Enroll=Enroll[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','Total_enroll_students']]


# In[225]:


Enroll_clean=Enroll[Enroll.Total_enroll_students > 0]


# In[226]:


Enroll_clean.shape


# In[227]:


sns.heatmap(Enroll_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[228]:


Enroll_clean.describe()


# In[229]:


Enroll_clean.hist()


# In[230]:


Enroll_clean.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_enrollment.csv', index = False,header=True)


# #### Merge CRDC master with CCD directory to extract only high schools

# In[231]:


cd /Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CCD


# In[232]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# In[233]:


ccd_directory['NCESSCH'] = ccd_directory['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[234]:


ccd_directory.drop(ccd_directory.columns[[4,5,6,10,11]], axis=1, inplace=True)


# In[235]:


ccd_directory.shape


# In[236]:


Sch_char_merged_ccd = pandas.merge(left=Sch_char,right=ccd_directory, how='left', left_on='NCESSCH', right_on='NCESSCH')
Sch_char_merged_ccd.shape


# In[237]:


Sch_char_merged_ccd.head()


# In[238]:


sns.heatmap(Sch_char_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[239]:


null_columns=Sch_char_merged_ccd.columns[Sch_char_merged_ccd.isnull().any()]
Sch_char_merged_ccd[null_columns].isnull().sum()


# #### Keeping only high schools

# In[240]:


Sch_char_hs=Sch_char_merged_ccd[Sch_char_merged_ccd['LEVEL']=='High' ]


# In[241]:


sns.heatmap(Sch_char_hs.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[242]:


Sch_char_hs.shape


# In[243]:


Sch_char_hs.columns


# In[244]:


Sch_char_hs.head()


# In[317]:


Sch_char_hs.drop([col for col in Sch_char_hs.columns if col.endswith('_y')],axis=1,inplace=True)


# In[318]:


HS_Sch_char=Sch_char_hs[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME_x', 'SCHID_x','SCH_NAME_x','COMBOKEY','Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH','LEVEL']]


# In[319]:


HS_Sch_char.shape


# #### Merge remaining CRDC clean files

# ##### Merging with school enroll

# In[320]:


HS_Sch_char_merged_enroll = pandas.merge(left=HS_Sch_char,right=Enroll_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_merged_enroll.shape


# In[321]:


HS_Sch_char_merged_enroll.columns


# In[322]:


#HS_Sch_char_merged_exp.head()


# In[323]:


HS_Sch_char_merged_enroll.drop([col for col in HS_Sch_char_merged_enroll.columns if col.endswith('_y')],axis=1,inplace=True)


# In[324]:


HS_Sch_char_merged_enroll.columns


# In[325]:


HS_Sch_char_merged_enroll.shape


# In[388]:


HS_Sch_char_enroll=HS_Sch_char_merged_enroll[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students']]


# ##### Merging with school support

# In[389]:


HS_Sch_char_enroll_merged_sup = pandas.merge(left=HS_Sch_char_enroll,right=Sch_sup_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_merged_sup.shape


# In[390]:


HS_Sch_char_enroll_merged_sup.columns


# In[391]:


HS_Sch_char_enroll_merged_sup.head()


# In[450]:


HS_Sch_char_enroll_sup=HS_Sch_char_enroll_merged_sup[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new', 'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students','SCH_FTETEACH_TOT',
       'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT']]


# In[451]:


HS_Sch_char_enroll_sup.shape


# ##### Merging with school expenditures

# In[452]:


HS_Sch_char_enroll_sup_merged_exp = pandas.merge(left=HS_Sch_char_enroll_sup,right=Sch_exp_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_merged_exp.shape


# In[453]:


HS_Sch_char_enroll_sup_merged_exp.columns


# In[454]:


HS_Sch_char_enroll_sup_merged_exp.head()


# In[455]:


HS_Sch_char_enroll_sup_exp=HS_Sch_char_enroll_sup_merged_exp[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students','SCH_FTETEACH_TOT',
       'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT','FTE_teachers_count','SalaryforTeachers']]


# In[456]:


HS_Sch_char_enroll_sup_exp.shape


# ##### Merging with SAT_ACT

# In[457]:


HS_Sch_char_enroll_sup_exp_merged_SA = pandas.merge(left=HS_Sch_char_enroll_sup_exp,right=SAT_ACT_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_merged_SA.shape


# In[458]:


HS_Sch_char_enroll_sup_exp_merged_SA.columns


# In[459]:


HS_Sch_char_enroll_sup_exp_merged_SA.head()


# In[460]:


HS_Sch_char_enroll_sup_exp_SA=HS_Sch_char_enroll_sup_exp_merged_SA[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students','SCH_FTETEACH_TOT',
       'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT','FTE_teachers_count','SalaryforTeachers','Total_SAT_ACT_students']]


# In[461]:


HS_Sch_char_enroll_sup_exp_SA.shape


# ##### Merging with IB

# In[462]:


HS_Sch_char_enroll_sup_exp_SA_merged_IB = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA,right=IB_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_merged_IB.shape


# In[463]:


HS_Sch_char_enroll_sup_exp_SA_merged_IB.columns


# In[464]:


#HS_Sch_char_enroll_sup_exp_SA_merged_IB.head()


# In[465]:


HS_Sch_char_enroll_sup_exp_SA_IB=HS_Sch_char_enroll_sup_exp_SA_merged_IB[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL',
       'Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT','SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers',
       'Total_SAT_ACT_students','SCH_IBENR_IND_new','Total_IB_students']]


# In[466]:


HS_Sch_char_enroll_sup_exp_SA_IB.shape


# ##### Merging with AP other

# In[467]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB,right=AP_other_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other.shape


# In[468]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other.columns


# In[469]:


#HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other.head()


# In[470]:


HS_Sch_char_enroll_sup_exp_SA_IB_AP_other=HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APOTHENR_IND_new',
       'Total_AP_other_students', 'Total_students_tookAP']]


# In[471]:


HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.shape


# In[472]:


sns.heatmap(HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[473]:


null_columns=HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.columns[HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.isnull().any()]
HS_Sch_char_enroll_sup_exp_SA_IB_AP_other[null_columns].isnull().sum()


# In[474]:


crdc_master_read = HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.dropna(axis = 0, how ='any') 


# In[475]:


print("Old data frame length:", len(HS_Sch_char_enroll_sup_exp_SA_IB_AP_other)) 
print("New data frame length:", len(crdc_master_read))  
print("Number of rows with at least 1 NA value: ", 
      (len(HS_Sch_char_enroll_sup_exp_SA_IB_AP_other)-len(crdc_master_read))) 


# In[476]:


sns.heatmap(crdc_master_read.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[477]:


crdc_master_read.shape


# In[478]:


crdc_master_read.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_master_read.csv', index = False,header=True)


# ##### Merging with AP math

# In[479]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB,right=AP_math_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math.shape


# In[480]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math.columns


# In[481]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath=HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new',
       'Total_AP_math_students', 'Total_students_tookAP']]


# In[482]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath.shape


# ##### Merging with Alg1 

# In[483]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1 = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB_APmath,right=Alg1_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1.shape


# In[484]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1.columns


# In[485]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new',
       'Total_AP_math_students', 'Total_students_tookAP','SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'Total_Alg1_enroll_students',
       'Total_Alg1_pass_students']]


# In[486]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.shape


# ##### Merging with Alg2

# In[487]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_merged_Alg2= pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1,right=Alg2_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_merged_Alg2.shape


# In[488]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_merged_Alg2.columns


# In[489]:


sns.heatmap(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_merged_Alg2.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[490]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_merged_Alg2[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new',
       'Total_AP_math_students', 'Total_students_tookAP','SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'Total_Alg1_enroll_students','Total_Alg1_pass_students',
        'SCH_MATHCLASSES_ALG2','SCH_MATHCERT_ALG2', 'Total_Alg2_enroll_students']]


# In[491]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2.shape


# ##### Merging with Calculus

# In[492]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_merged_Cal= pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2,right=Calculus_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_merged_Cal.shape


# In[493]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_merged_Cal.columns


# In[494]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_merged_Cal[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new',
       'Total_AP_math_students', 'Total_students_tookAP','SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'Total_Alg1_enroll_students','Total_Alg1_pass_students',
        'SCH_MATHCLASSES_ALG2','SCH_MATHCERT_ALG2', 'Total_Alg2_enroll_students','SCH_MATHCLASSES_CALC','SCH_MATHCERT_CALC', 'Total_Calc_enroll_students']]


# In[495]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal.shape


# ##### Merging with Geometry

# In[496]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal_merged_Geo= pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal,right=Geometry_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal_merged_Geo.shape


# In[497]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal_merged_Geo.columns


# In[498]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal_Geo=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal_merged_Geo[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new',
       'Total_AP_math_students', 'Total_students_tookAP','SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'Total_Alg1_enroll_students','Total_Alg1_pass_students',
        'SCH_MATHCLASSES_ALG2','SCH_MATHCERT_ALG2', 'Total_Alg2_enroll_students','SCH_MATHCLASSES_CALC','SCH_MATHCERT_CALC', 'Total_Calc_enroll_students','SCH_MATHCERT_GEOM', 'SCH_MATHCLASSES_GEOM',
       'Total_Geomty_enroll_students']]


# In[499]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal_Geo.shape


# In[500]:


sns.heatmap(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1_2_Cal_Geo.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[501]:


sns.heatmap(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[502]:


null_columns=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.columns[HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.isnull().any()]
HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1[null_columns].isnull().sum()


# In[503]:


crdc_master_math = HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.dropna(axis = 0, how ='any') 


# In[504]:


print("Old data frame length:", len(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1)) 
print("New data frame length:", len(crdc_master_math))  
print("Number of rows with at least 1 NA value: ", 
      (len(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1)-len(crdc_master_math))) 


# In[505]:


crdc_master_math.shape


# In[506]:


crdc_master_math.dtypes


# In[507]:


crdc_master_math.to_csv (r'/Users/dansari/Documents/GitHub/Identifying-features-to-predict-high-school-assessment-proficiency/Phase1/Data/CRDC/Clean_crdc_master_math.csv', index = False,header=True)


# In[ ]:




