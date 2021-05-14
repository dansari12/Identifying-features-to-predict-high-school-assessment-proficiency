#!/usr/bin/env python
# coding: utf-8

# ## Phase 1: Clean up of CRDC data files. This notebook contains code that cleans and merges the 8 individual crdc files for school characteristics, school support, school expenditures, AP, IB, SAT_ACT, Algebra 1 and enrollment into a two files: one for reading and the other for math.

# ### Loading necessary libraries

# In[298]:


import pandas
pandas.__version__
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[299]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CRDC


# ### 1. Cleaning school characteristics file

# In[300]:


Sch_char = pandas.read_csv("School Characteristics.csv",encoding='cp1252')
Sch_char.head()


# In[301]:


Sch_char['SCHID'] = Sch_char['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[302]:


Sch_char['LEAID'] = Sch_char['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[303]:


Sch_char.columns


# #### Dropping unnecessary columns

# In[304]:


Sch_char.drop(Sch_char.columns[[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,31]], axis=1, inplace=True)


# In[305]:


Sch_char.shape


# In[306]:


#Sch_char.head()


# ##### Since we do not have NCESSCH ID we can impute it using the LEAID and SCHID. 
# ##### Note: Unique NCES public school ID is generated based on the (7-digit NCES agency ID (LEAID) + 5-digit NCES school ID (SCHID). See https://nces.ed.gov/ccd/data/txt/psu10play.txt for more info

# In[307]:


cols = ['LEAID', 'SCHID']
Sch_char['NCESSCH'] = Sch_char[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[308]:


Sch_char['NCESSCH'].is_unique


# #### Renaming columns

# In[309]:


Sch_char.rename(columns={'SCH_STATUS_SPED':'Special_ed_schl','SCH_STATUS_MAGNET':'Magnet_schl','SCH_STATUS_CHARTER':'Charter_Schl','SCH_STATUS_ALT':'Alternate_schl'}, inplace=True)


# In[310]:


Sch_char.head()


# In[311]:


count = Sch_char['Charter_Schl'].value_counts() 
print(count) 


# ##### Recoding string Y/N values to integers 1/0

# In[312]:


Sch_char['Special_ed_schl_new'] = Sch_char['Special_ed_schl'].replace(['Yes','No'],['1','0'])


# In[313]:


Sch_char['Magnet_schl_new'] = Sch_char['Magnet_schl'].replace(['Yes','No'],['1','0'])


# In[314]:


Sch_char['Charter_Schl_new'] = Sch_char['Charter_Schl'].replace(['Yes','No'],['1','0'])


# In[315]:


Sch_char['Alternate_schl_new'] = Sch_char['Alternate_schl'].replace(['Yes','No'],['1','0'])


# In[316]:


Sch_char[['Special_ed_schl_new', 'Magnet_schl_new','Charter_Schl_new','Alternate_schl_new']] = Sch_char[['Special_ed_schl_new', 'Magnet_schl_new','Charter_Schl_new','Alternate_schl_new']].astype(int)


# #### Checking for missing or null values

# In[317]:


sns.heatmap(Sch_char.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[318]:


Sch_char.describe()


# In[319]:


Sch_char.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_schlcharacteristics.csv', index = False, header=True)


# ### 2. Cleaning school expenditure file

# In[320]:


Sch_exp = pandas.read_csv("School Expenditures.csv", encoding='cp1252')
Sch_exp.tail()


# In[321]:


Sch_exp['SCHID'] = Sch_exp['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[322]:


Sch_exp['LEAID'] = Sch_exp['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[323]:


Sch_exp.columns


# In[324]:


Sch_exp.drop(Sch_exp.columns[[7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]], axis=1, inplace=True)


# In[325]:


Sch_exp.head()


# ##### Since we do not have NCESSCH ID we can impute it using the LEAID and SCHID. 

# In[326]:


cols = ['LEAID', 'SCHID']
Sch_exp['NCESSCH'] = Sch_exp[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[327]:


Sch_exp.shape


# In[328]:


Sch_exp['NCESSCH'].is_unique


# ##### Renaming columns

# In[329]:


Sch_exp.rename(columns={'SCH_FTE_TEACH_WOFED':'FTE_teachers_count','SCH_SAL_TEACH_WOFED':'SalaryforTeachers'}, inplace=True)


# In[330]:


Sch_exp.head()


# In[331]:


#Sch_exp['Teacher_salary_ratio'] = (Sch_exp['SalaryforTeachers'] / Sch_exp['FTE_teachers_count'])


# #### Checking for missing or null values

# In[332]:


sns.heatmap(Sch_exp.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[333]:


Sch_exp.describe()


# #### Dropping columns with less than zero FTE Teacher counts and Salary expenditures

# In[334]:


Sch_expTC= Sch_exp[Sch_exp.FTE_teachers_count > 0]


# In[335]:


Sch_exp_clean= Sch_expTC[Sch_expTC.SalaryforTeachers > 0]


# In[336]:


Sch_exp_clean.shape


# In[337]:


Sch_exp_clean.describe()


# In[338]:


Sch_exp_clean.head()


# In[339]:


Sch_exp_clean.hist()


# In[340]:


Sch_exp_clean.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_schlexpenses.csv', index = False, header=True)


# ### 3. Cleaning school support file

# In[341]:


Sch_sup= pandas.read_csv("School Support.csv",encoding='cp1252')
Sch_sup.head()


# In[342]:


Sch_sup['SCHID'] = Sch_sup['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[343]:


Sch_sup['LEAID'] = Sch_sup['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[344]:


Sch_sup.columns


# In[345]:


Sch_sup.head()


# ##### Dropping irrelevant columns

# In[346]:


Sch_sup.drop(Sch_sup.columns[[7,11,12,13,14,15,16,17,18,19,20,21]], axis=1, inplace=True)


# In[347]:


Sch_sup.head()


# ##### Since we do not have NCESSCH ID we can impute it using the LEAID and SCHID. 

# In[348]:


cols = ['LEAID', 'SCHID']
Sch_sup['NCESSCH'] = Sch_sup[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[349]:


Sch_sup.shape


# In[350]:


Sch_sup['NCESSCH'].is_unique


# #### Checking for missing or null values

# In[351]:


sns.heatmap(Sch_sup.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[352]:


Sch_sup.describe()


# #### Filtering FTE count greater than 1 and Cert count greater than -5

# In[353]:


Sch_sup_FTEGT1= Sch_sup[Sch_sup.SCH_FTETEACH_TOT > 1]


# In[354]:


Sch_sup_clean= Sch_sup_FTEGT1[Sch_sup_FTEGT1.SCH_FTETEACH_CERT > -5]


# In[355]:


Sch_sup_clean.describe()


# In[356]:


Sch_sup_clean.shape


# In[357]:


Sch_sup_clean.head()


# In[358]:


Sch_sup_clean.describe()


# In[359]:


Sch_sup_clean.hist()


# In[360]:


Sch_sup_clean.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_schlsupport.csv', index = False, header=True)


# ### 4. Cleaning SAT and ACT file

# In[361]:


SAT_ACT = pandas.read_csv("SAT and ACT.csv", encoding='cp1252')
SAT_ACT.head()


# In[362]:


SAT_ACT['LEAID'] = SAT_ACT['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[363]:


SAT_ACT['SCHID'] = SAT_ACT['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[364]:


SAT_ACT.columns


# In[365]:


SAT_ACT.shape


# In[366]:


SAT_ACT.drop(SAT_ACT.columns[[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27]], axis=1, inplace=True)


# In[367]:


SAT_ACT.head()


# In[368]:


cols = ['LEAID', 'SCHID']
SAT_ACT['NCESSCH'] = SAT_ACT[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# #### Adding total count of male and female participation on ACT and SAT

# In[369]:


SAT_ACT.rename(columns={'TOT_SATACT_M':'Male_part_count','TOT_SATACT_F':'Female_part_count'}, inplace=True)


# In[370]:


SAT_ACT.describe()


# In[371]:


SAT_ACTGT0= SAT_ACT.loc[SAT_ACT['Male_part_count'] > 0]


# In[372]:


SAT_ACTGT0.describe()


# In[373]:


SAT_ACTGT0['Total_SAT_ACT_students'] = (SAT_ACTGT0['Male_part_count'] + SAT_ACTGT0['Female_part_count'])


# In[374]:


SAT_ACTGT0.describe()


# In[375]:


SAT_ACTGT0.shape


# #### Keeping total counts greater than 0

# In[376]:


SAT_ACT_clean= SAT_ACTGT0[SAT_ACTGT0.Total_SAT_ACT_students > 0]


# In[377]:


SAT_ACT_clean.shape


# In[378]:


SAT_ACT_clean.head()


# #### Checking for missing or null values

# In[379]:


sns.heatmap(SAT_ACT.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[380]:


SAT_ACT_clean.describe()


# In[381]:


SAT_ACT_clean.hist()


# In[382]:


SAT_ACT_clean.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_SAT_ACT.csv', index = False, header=True)


# ### 5. Cleaning IB file

# In[383]:


IB= pandas.read_csv("International Baccalaureate.csv",encoding='cp1252')
IB.head()


# In[384]:


IB['SCHID'] = IB['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[385]:


IB['LEAID'] = IB['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[386]:


IB.columns


# In[387]:


IB.shape


# In[388]:


IB.drop(IB.columns[[7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28]], axis=1, inplace=True)


# In[389]:


IB.head()


# In[390]:


cols = ['LEAID', 'SCHID']
IB['NCESSCH'] = IB[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[391]:


IB.rename(columns={'TOT_IBENR_M':'Male_enroll_count','TOT_IBENR_F':'Female_enroll_count'}, inplace=True)


# In[392]:


IB.describe()


# #### Recoding missing values as zero so that total counts can be calculated later

# In[393]:


IB['Male_enroll_count'] = IB['Male_enroll_count'].replace(-9,0)


# In[394]:


IB['Female_enroll_count'] = IB['Female_enroll_count'].replace(-9,0)


# In[395]:


IB.describe()


# In[396]:


IB['Total_IB_students'] = (IB['Male_enroll_count'] + IB['Female_enroll_count'])


# In[397]:


IB.describe()


# In[398]:


IB.shape


# #### Keeping IB program indicator with Y/N

# In[399]:


IB_clean= IB[IB.SCH_IBENR_IND != '-9']


# In[400]:


IB_clean.shape


# In[401]:


IB_clean.dtypes


# ##### Recoding string Y/N values to integers 1/0

# In[402]:


IB_clean['SCH_IBENR_IND_new'] = IB_clean['SCH_IBENR_IND'].replace(['Yes','No'],['1','0'])


# In[403]:


IB_clean[['SCH_IBENR_IND_new']]=IB_clean[['SCH_IBENR_IND_new']].astype(int)


# In[404]:


IB_clean.head()


# #### Checking for missing or null values

# In[405]:


sns.heatmap(IB_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[406]:


IB_clean.describe()


# ##### Filtering out negative values

# In[407]:


IB_clean1= IB_clean[IB_clean.SCH_IBENR_IND != '-6']


# In[408]:


IB_clean2= IB_clean1[IB_clean1.SCH_IBENR_IND != '-5']


# In[409]:


IB_clean2.describe()


# In[410]:


IB_clean2.hist()


# In[411]:


IB_clean2.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_IB.csv', index = False, header=True)


# ### 6. Cleaning AP file

# In[412]:


AP = pandas.read_csv("Advanced Placement.csv",encoding='cp1252')
AP.head()


# In[413]:


AP['SCHID'] = AP['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[414]:


AP['LEAID'] = AP['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[415]:


AP.columns


# In[416]:


AP.shape


# In[417]:


AP=AP[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','SCH_APENR_IND','SCH_APCOURSES','SCH_APMATHENR_IND','TOT_APMATHENR_M','TOT_APMATHENR_F','SCH_APOTHENR_IND','TOT_APOTHENR_M','TOT_APOTHENR_F','TOT_APEXAM_ONEORMORE_M','TOT_APEXAM_ONEORMORE_F']]


# In[418]:


AP.shape


# In[419]:


AP.head()


# In[420]:


cols = ['LEAID', 'SCHID']
AP['NCESSCH'] = AP[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[421]:


AP.rename(columns={'TOT_APMATHENR_M':'Male_enroll_math_count','TOT_APMATHENR_F':'Female_enroll_math_count','TOT_APOTHENR_M':'Male_enroll_other_count','TOT_APOTHENR_F':'Female_enroll_other_count'}, inplace=True)


# In[422]:


AP.describe()


# In[423]:


AP.shape


# In[424]:


AP= AP[AP.SCH_APENR_IND.isin(['Yes','No'])]


# In[425]:


AP.shape


# In[426]:


AP.describe()


# ##### If AP enrollment indicator is a No, then the corresponding columns for courses and student counts are marked a -9. So lets replace -9 with 0 counts for schools that don't have any AP enrollment indicators

# In[427]:


AP['SCH_APCOURSES'] = AP['SCH_APCOURSES'].replace(-9,0)


# In[428]:


AP['Male_enroll_math_count'] = AP['Male_enroll_math_count'].replace(-9,0)


# In[429]:


AP['Female_enroll_math_count'] = AP['Female_enroll_math_count'].replace(-9,0)


# In[430]:


AP['Male_enroll_other_count'] = AP['Male_enroll_other_count'].replace(-9,0)


# In[431]:


AP['Female_enroll_other_count'] = AP['Female_enroll_other_count'].replace(-9,0)


# In[432]:


AP['TOT_APEXAM_ONEORMORE_M'] = AP['TOT_APEXAM_ONEORMORE_M'].replace(-9,0)


# In[433]:


AP['TOT_APEXAM_ONEORMORE_F'] = AP['TOT_APEXAM_ONEORMORE_F'].replace(-9,0)


# Total counts of M and F

# In[434]:


AP['Total_AP_math_students'] = (AP['Male_enroll_math_count'] + AP['Female_enroll_math_count'])


# In[435]:


AP['Total_AP_other_students'] = (AP['Male_enroll_other_count'] + AP['Female_enroll_other_count'])


# In[436]:


AP['Total_students_tookAP'] = (AP['TOT_APEXAM_ONEORMORE_M'] + AP['TOT_APEXAM_ONEORMORE_F'])


# In[437]:


AP.columns


# In[438]:


AP_math=AP[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','SCH_APENR_IND', 'SCH_APCOURSES', 'SCH_APMATHENR_IND',
       'Total_AP_math_students','Total_students_tookAP']]


# In[439]:


AP_math.describe()


# #### Filtering out any ligering negative values that indicate missing not N/As

# In[440]:


AP_math_clean= AP_math.loc[AP_math['SCH_APCOURSES'] > -1]


# In[441]:


AP_math_clean= AP_math_clean.loc[AP_math_clean['Total_AP_math_students'] > -1]


# In[442]:


AP_math_clean= AP_math_clean.loc[AP_math_clean['Total_students_tookAP'] > -1]


# In[443]:


AP_math_clean.describe()


# In[444]:


AP_math_clean.shape


# In[445]:


AP_math_clean.dtypes


# ##### Recoding string Y/N values to integers 1/0

# In[446]:


AP_math_clean['SCH_APENR_IND_new'] = AP_math_clean['SCH_APENR_IND'].replace(['Yes','No'],['1','0'])


# In[447]:


AP_math_clean[['SCH_APENR_IND_new']] = AP_math_clean[['SCH_APENR_IND_new']].astype(int)


# In[448]:


AP_math_clean['SCH_APMATHENR_IND_new'] = AP_math_clean['SCH_APMATHENR_IND'].replace(['Yes','No','-9'],['1','0','0'])


# In[449]:


AP_math_clean[['SCH_APMATHENR_IND_new']] = AP_math_clean[['SCH_APMATHENR_IND_new']].astype(int)


# In[450]:


AP_math_clean.dtypes


# #### Checking for missing or null values

# In[451]:


sns.heatmap(AP_math_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[452]:


AP_math_clean.describe()


# In[453]:


AP_math_clean.hist()


# In[454]:


AP_math_clean.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_AP_math.csv', index = False, header=True)


# In[455]:


AP_other=AP[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','SCH_APENR_IND', 'SCH_APCOURSES', 'SCH_APOTHENR_IND',
       'Total_AP_other_students', 'Total_students_tookAP']]


# In[456]:


AP_other_clean= AP_other[AP_other.SCH_APENR_IND.isin(['Yes','No'])]


# In[457]:


AP_other_clean.shape


# In[458]:


AP_other_clean.dtypes


# In[459]:


count_other = AP_other_clean['SCH_APOTHENR_IND'].value_counts() 
print(count_other) 


# ##### Recoding string Y/N values to integers 1/0

# In[460]:


AP_other_clean['SCH_APENR_IND_new'] = AP_other_clean['SCH_APENR_IND'].replace(['Yes','No'],['1','0'])


# In[461]:


AP_other_clean[['SCH_APENR_IND_new']] = AP_other_clean[['SCH_APENR_IND_new']].astype(int)


# In[462]:


AP_other_clean['SCH_APOTHENR_IND_new'] = AP_other_clean['SCH_APOTHENR_IND'].replace(['Yes','No','-9'],['1','0','0'])


# In[463]:


AP_other_clean[['SCH_APOTHENR_IND_new']] = AP_other_clean[['SCH_APOTHENR_IND_new']].astype(int)


# In[464]:


AP_other_clean.dtypes


# In[465]:


sns.heatmap(AP_other_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[466]:


AP_other_clean.describe()


# In[467]:


AP_other_clean1= AP_other_clean.loc[AP_other_clean['SCH_APCOURSES'] > -1]


# In[468]:


AP_other_clean2= AP_other_clean1.loc[AP_other_clean1['Total_students_tookAP'] > -1]


# In[469]:


AP_other_clean2.describe()


# In[470]:


AP_other_clean.hist()


# In[471]:


AP_other_clean.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_AP_other.csv', index = False, header=True)


# ### 7. Cleaning Algebra 1 file

# In[472]:


Alg1 = pandas.read_csv("Algebra I.csv",encoding='cp1252')
Alg1.head()


# In[473]:


Alg1['SCHID'] = Alg1['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[474]:


Alg1['LEAID'] = Alg1['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[475]:


Alg1.columns


# In[476]:


Alg1.shape


# In[477]:


Alg1=Alg1[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','SCH_MATHCLASSES_ALG','SCH_MATHCERT_ALG','TOT_ALGENR_GS0910_M',
           'TOT_ALGENR_GS0910_F','TOT_ALGENR_GS1112_M','TOT_ALGENR_GS1112_F','TOT_ALGPASS_GS0910_M','TOT_ALGPASS_GS0910_F','TOT_ALGPASS_GS1112_M','TOT_ALGPASS_GS1112_F']]


# In[478]:


Alg1.shape


# In[479]:


Alg1.head()


# In[480]:


cols = ['LEAID', 'SCHID']
Alg1['NCESSCH'] = Alg1[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[481]:


Alg1.rename(columns={'TOT_ALGENR_GS0910_M':'Male_enroll_9to10_count','TOT_ALGENR_GS0910_F':'Female_enroll_9to10_count','TOT_ALGENR_GS1112_M':'Male_enroll_11to12_count',
                   'TOT_ALGENR_GS1112_F':'Female_enroll_11to12_count','TOT_ALGPASS_GS0910_M':'Male_pass_9to10_count','TOT_ALGPASS_GS0910_F':'Female_pass_9to10_count',
                  'TOT_ALGPASS_GS1112_M':'Male_pass_11to12_count','TOT_ALGPASS_GS1112_F':'Female_pass_11to12_count'}, inplace=True)


# In[482]:


Alg1.columns


# In[483]:


Alg1.describe()


# ##### Lets replace -9 with 0 counts for enrollment counts so we can total values later

# In[484]:


Alg1['Male_enroll_9to10_count'] = Alg1['Male_enroll_9to10_count'].replace(-9,0)


# In[485]:


Alg1['Female_enroll_9to10_count'] = Alg1['Female_enroll_9to10_count'].replace(-9,0)


# In[486]:


Alg1['Male_enroll_11to12_count'] = Alg1['Male_enroll_11to12_count'].replace(-9,0)


# In[487]:


Alg1['Female_enroll_11to12_count'] = Alg1['Female_enroll_11to12_count'].replace(-9,0)


# In[488]:


Alg1['Male_pass_9to10_count'] = Alg1['Male_pass_9to10_count'].replace(-9,0)


# In[489]:


Alg1['Female_pass_9to10_count'] = Alg1['Female_pass_9to10_count'].replace(-9,0)


# In[490]:


Alg1['Male_pass_11to12_count'] = Alg1['Male_pass_11to12_count'].replace(-9,0)


# In[491]:


Alg1['Female_pass_11to12_count'] = Alg1['Female_pass_11to12_count'].replace(-9,0)


# Total counts of M and F

# In[492]:


Alg1['Total_Alg1_enroll_students'] = (Alg1['Male_enroll_9to10_count'] + Alg1['Female_enroll_9to10_count'] + Alg1['Male_enroll_11to12_count'] + Alg1['Female_enroll_11to12_count'])


# In[493]:


Alg1['Total_Alg1_pass_students'] = (Alg1['Male_pass_9to10_count'] + Alg1['Female_pass_9to10_count'] + Alg1['Male_pass_11to12_count'] + Alg1['Female_pass_11to12_count'])


# In[494]:


Alg1=Alg1[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME',
       'COMBOKEY', 'SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'NCESSCH',
       'Total_Alg1_enroll_students', 'Total_Alg1_pass_students']]


# In[495]:


Alg1_clean= Alg1[Alg1.SCH_MATHCLASSES_ALG > 0]


# In[496]:


Alg1_clean.shape


# In[497]:


Alg1_clean.describe()


# #### Checking for missing or null values

# In[498]:


sns.heatmap(Alg1_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[499]:


Alg1_clean.describe()


# In[500]:


Alg1_clean.hist()


# In[501]:


Alg1_clean.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_Alg1.csv', index = False, header=True)


# ### 8. Cleaning Enrollment file

# In[502]:


Enroll = pandas.read_csv("Enrollment.csv",encoding='cp1252')
Enroll.head()


# In[503]:


Enroll['SCHID'] = Enroll['SCHID'].apply(lambda x: '{0:0>5}'.format(x))


# In[504]:


Enroll['LEAID'] = Enroll['LEAID'].apply(lambda x: '{0:0>7}'.format(x))


# In[505]:


Enroll.columns


# In[506]:


Enroll.shape


# In[507]:


Enroll=Enroll[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','TOT_ENR_M','TOT_ENR_F']]


# In[508]:


Enroll.shape


# In[509]:


cols = ['LEAID', 'SCHID']
Enroll['NCESSCH'] = Enroll[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# In[510]:


Enroll['Total_enroll_students'] = (Enroll['TOT_ENR_M'] + Enroll['TOT_ENR_F'])


# In[511]:


Enroll.columns


# In[512]:


Enroll=Enroll[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME', 'SCHID', 'SCH_NAME','COMBOKEY','NCESSCH','Total_enroll_students']]


# #### Excluding schools with 0 enrollment counts

# In[513]:


Enroll_clean=Enroll[Enroll.Total_enroll_students > 0]


# In[514]:


Enroll_clean.shape


# #### Checking for missing or null values

# In[515]:


sns.heatmap(Enroll_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[516]:


Enroll_clean.describe()


# In[517]:


Enroll_clean.hist()


# In[518]:


Enroll_clean.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_enrollment.csv', index = False,header=True)


# #### 9. Merge CRDC school characteristics with CCD directory to extract only high schools

# In[519]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/CCD


# In[520]:


ccd_directory= pandas.read_csv("Clean_ccd_directory.csv")
ccd_directory.head()


# In[521]:


ccd_directory['NCESSCH'] = ccd_directory['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))


# In[522]:


ccd_directory.drop(ccd_directory.columns[[4,5,6,10,11]], axis=1, inplace=True)


# In[523]:


ccd_directory.shape


# In[524]:


Sch_char_merged_ccd = pandas.merge(left=Sch_char,right=ccd_directory, how='left', left_on='NCESSCH', right_on='NCESSCH')
Sch_char_merged_ccd.shape


# In[525]:


Sch_char_merged_ccd.head()


# In[526]:


sns.heatmap(Sch_char_merged_ccd.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[527]:


null_columns=Sch_char_merged_ccd.columns[Sch_char_merged_ccd.isnull().any()]
Sch_char_merged_ccd[null_columns].isnull().sum()


# #### Keeping only high schools

# In[528]:


Sch_char_hs=Sch_char_merged_ccd[Sch_char_merged_ccd['LEVEL']=='High' ]


# In[529]:


sns.heatmap(Sch_char_hs.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[530]:


Sch_char_hs.shape


# In[531]:


Sch_char_hs.columns


# In[532]:


Sch_char_hs.head()


# In[533]:


Sch_char_hs.drop([col for col in Sch_char_hs.columns if col.endswith('_y')],axis=1,inplace=True)


# In[534]:


HS_Sch_char=Sch_char_hs[['LEA_STATE', 'LEA_STATE_NAME', 'LEAID', 'LEA_NAME_x', 'SCHID_x','SCH_NAME_x','COMBOKEY','Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH','LEVEL']]


# In[535]:


HS_Sch_char.shape


# #### 10. Merge remaining CRDC clean files

# ##### Merging with school enroll

# In[536]:


HS_Sch_char_merged_enroll = pandas.merge(left=HS_Sch_char,right=Enroll_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_merged_enroll.shape


# In[537]:


HS_Sch_char_merged_enroll.columns


# In[538]:


#HS_Sch_char_merged_exp.head()


# In[539]:


HS_Sch_char_merged_enroll.drop([col for col in HS_Sch_char_merged_enroll.columns if col.endswith('_y')],axis=1,inplace=True)


# In[540]:


HS_Sch_char_merged_enroll.columns


# In[541]:


HS_Sch_char_merged_enroll.shape


# In[542]:


HS_Sch_char_enroll=HS_Sch_char_merged_enroll[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students']]


# ##### Merging with school support

# In[543]:


HS_Sch_char_enroll_merged_sup = pandas.merge(left=HS_Sch_char_enroll,right=Sch_sup_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_merged_sup.shape


# In[544]:


HS_Sch_char_enroll_merged_sup.columns


# In[545]:


HS_Sch_char_enroll_merged_sup.head()


# In[546]:


HS_Sch_char_enroll_sup=HS_Sch_char_enroll_merged_sup[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new', 'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students','SCH_FTETEACH_TOT',
       'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT']]


# In[547]:


HS_Sch_char_enroll_sup.shape


# ##### Merging with school expenditures

# In[548]:


HS_Sch_char_enroll_sup_merged_exp = pandas.merge(left=HS_Sch_char_enroll_sup,right=Sch_exp_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_merged_exp.shape


# In[549]:


HS_Sch_char_enroll_sup_merged_exp.columns


# In[550]:


HS_Sch_char_enroll_sup_merged_exp.head()


# In[551]:


HS_Sch_char_enroll_sup_exp=HS_Sch_char_enroll_sup_merged_exp[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students','SCH_FTETEACH_TOT',
       'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT','FTE_teachers_count','SalaryforTeachers']]


# In[552]:


HS_Sch_char_enroll_sup_exp.shape


# ##### Merging with SAT_ACT

# In[553]:


HS_Sch_char_enroll_sup_exp_merged_SA = pandas.merge(left=HS_Sch_char_enroll_sup_exp,right=SAT_ACT_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_merged_SA.shape


# In[554]:


HS_Sch_char_enroll_sup_exp_merged_SA.columns


# In[555]:


HS_Sch_char_enroll_sup_exp_merged_SA.head()


# In[556]:


HS_Sch_char_enroll_sup_exp_SA=HS_Sch_char_enroll_sup_exp_merged_SA[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL', 'Total_enroll_students','SCH_FTETEACH_TOT',
       'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT','FTE_teachers_count','SalaryforTeachers','Total_SAT_ACT_students']]


# In[557]:


HS_Sch_char_enroll_sup_exp_SA.shape


# ##### Merging with IB

# In[558]:


HS_Sch_char_enroll_sup_exp_SA_merged_IB = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA,right=IB_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_merged_IB.shape


# In[559]:


HS_Sch_char_enroll_sup_exp_SA_merged_IB.columns


# In[560]:


#HS_Sch_char_enroll_sup_exp_SA_merged_IB.head()


# In[561]:


HS_Sch_char_enroll_sup_exp_SA_IB=HS_Sch_char_enroll_sup_exp_SA_merged_IB[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL',
       'Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT','SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers',
       'Total_SAT_ACT_students','SCH_IBENR_IND_new','Total_IB_students']]


# In[562]:


HS_Sch_char_enroll_sup_exp_SA_IB.shape


# #### For AP there are columns that are specific to math and reading so lets only include the relevant columns for each subject area

# ##### Merging with AP other

# In[563]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB,right=AP_other_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other.shape


# In[564]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other.columns


# In[565]:


#HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other.head()


# In[566]:


HS_Sch_char_enroll_sup_exp_SA_IB_AP_other=HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_other[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new','Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APOTHENR_IND_new',
       'Total_AP_other_students', 'Total_students_tookAP']]


# In[567]:


HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.shape


# #### Checking for missing or null values

# In[568]:


sns.heatmap(HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[569]:


null_columns=HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.columns[HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.isnull().any()]
HS_Sch_char_enroll_sup_exp_SA_IB_AP_other[null_columns].isnull().sum()


# In[570]:


crdc_master_read = HS_Sch_char_enroll_sup_exp_SA_IB_AP_other.dropna(axis = 0, how ='any') 


# In[571]:


print("Old data frame length:", len(HS_Sch_char_enroll_sup_exp_SA_IB_AP_other)) 
print("New data frame length:", len(crdc_master_read))  
print("Number of rows with at least 1 NA value: ", 
      (len(HS_Sch_char_enroll_sup_exp_SA_IB_AP_other)-len(crdc_master_read))) 


# #### Checking for missing or null values

# In[572]:


sns.heatmap(crdc_master_read.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[573]:


crdc_master_read.shape


# #### All relevant features necessary for the reading dataset are included so lets save this as crdc_master file for the reading

# In[574]:


crdc_master_read.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_master_read.csv', index = False,header=True)


# ##### Merging with AP math

# In[575]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB,right=AP_math_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math.shape


# In[576]:


HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math.columns


# In[577]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath=HS_Sch_char_enroll_sup_exp_SA_IB_merged_AP_math[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new',
       'Total_AP_math_students', 'Total_students_tookAP']]


# In[578]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath.shape


# ##### Merging with Alg1 

# In[579]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1 = pandas.merge(left=HS_Sch_char_enroll_sup_exp_SA_IB_APmath,right=Alg1_clean, how='left', left_on='NCESSCH', right_on='NCESSCH')
HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1.shape


# In[580]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1.columns


# In[581]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_merged_Alg1[['LEA_STATE_x', 'LEA_STATE_NAME_x', 'LEAID_x', 'LEA_NAME_x', 'SCHID_x',
       'SCH_NAME_x', 'COMBOKEY_x', 'Special_ed_schl_new',
       'Magnet_schl_new', 'Charter_Schl_new', 'Alternate_schl_new', 'NCESSCH', 'LEVEL','Total_enroll_students', 'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT',
       'SCH_FTETEACH_NOTCERT', 'FTE_teachers_count', 'SalaryforTeachers','Total_SAT_ACT_students', 'SCH_IBENR_IND_new', 'Total_IB_students','SCH_APENR_IND_new', 'SCH_APCOURSES', 'SCH_APMATHENR_IND_new',
       'Total_AP_math_students', 'Total_students_tookAP','SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG', 'Total_Alg1_enroll_students',
       'Total_Alg1_pass_students']]


# In[582]:


HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.shape


# #### Checking for missing or null values

# In[583]:


sns.heatmap(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[584]:


null_columns=HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.columns[HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.isnull().any()]
HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1[null_columns].isnull().sum()


# In[585]:


crdc_master_math = HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1.dropna(axis = 0, how ='any') 


# In[586]:


print("Old data frame length:", len(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1)) 
print("New data frame length:", len(crdc_master_math))  
print("Number of rows with at least 1 NA value: ", 
      (len(HS_Sch_char_enroll_sup_exp_SA_IB_APmath_Alg1)-len(crdc_master_math))) 


# In[587]:


crdc_master_math.shape


# In[588]:


crdc_master_math.dtypes


# #### All relevant features necessary for the math dataset are included so lets save this as crdc_master file for the math

# In[589]:


crdc_master_math.to_csv (r'/Users/dansa/Documents/GitHub/Phase1/Data/CRDC/Clean_crdc_master_math.csv', index = False,header=True)

