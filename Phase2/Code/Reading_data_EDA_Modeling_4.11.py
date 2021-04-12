#!/usr/bin/env python
# coding: utf-8

# ### Phase 2: Conducting EDA and model construction using the master_reading.csv file that contains all necessary features and target variable

# In[1]:


import sys
print(sys.base_prefix)


# In[12]:


#pip install --user scikit-learn


# In[13]:


#!{sys.executable} -m pip install xgboost


# In[15]:


#!{sys.executable} -m pip install lightgbm


# #### Importing all necessary libraries

# In[19]:


import pandas
pandas.__version__
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV, learning_curve, cross_val_predict
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.pipeline import Pipeline
import warnings
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/MASTER


# #### Loading the data and reformatting the school id column

# In[21]:


master_reading_new = pandas.read_csv("master_reading.csv")
master_reading_new['NCESSCH'] = master_reading_new['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
master_reading_new.head()


# In[22]:


master_reading_new.shape


# #### Inspecting the data file

# In[23]:


master_reading_new.columns


# Create a data frame with only the needed columns for further analysis

# In[24]:


reading=pd.DataFrame(master_reading_new, columns=[ 'NCESSCH','SCH_TYPE_x', 
       'TITLEI_STATUS','TEACHERS', 'FARMS_COUNT', 'Total_enroll_students',
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT',
       'FTE_teachers_count', 'SalaryforTeachers', 'Total_SAT_ACT_students',
       'SCH_IBENR_IND_new', 'Total_IB_students', 'SCH_APENR_IND_new',
       'SCH_APCOURSES', 'SCH_APOTHENR_IND_new', 'Total_AP_other_students',
       'Total_students_tookAP', 'Income_Poverty_ratio','ALL_RLA00PCTPROF_1718_new'])


# In[25]:


reading.head()


# In[26]:


reading.rename(columns={'NCESSCH':'School_ID', 'SCH_TYPE_x':'School_type','FARMS_COUNT':'No.FARMS_students',
                       'SCH_FTETEACH_TOT':'FTE_teachcount','SCH_FTETEACH_CERT':'Certified_FTE_teachers','SCH_FTETEACH_NOTCERT':
                       'Noncertified_FTE_teachers','Total_SAT_ACT_students':'Students_participate_SAT_ACT','SCH_IBENR_IND_new':'IB_Indicator','SCH_APENR_IND_new':'AP_Indicator',
                        'SCH_APCOURSES':'No.ofAP_courses_offer','SCH_APOTHENR_IND_new':'Students_enroll_inotherAP?','ALL_RLA00PCTPROF_1718_new':'Percent_Reading_Proficient'}, inplace=True)


# In[27]:


reading.describe()


# In[28]:


print(reading.info())


# We have 14,741 entries and no null values in any column. There are 21 columns, but we can drop the school_id and we'll want to split off the Percent_Reading_Proficient.
# The object type features should be strings.
# 
# Let's take a quick look at some of the data.

# In[30]:


reading.hist(bins=50, figsize=(20,15))
plt.show()


# We can see that some features have most of their instances at or near zero and relatively few instances at higher values, in some cases much higher. Other features cluster close to zero and have long tails. We also see the percent_reading_proficient is almost normally distributed.

# In[31]:


#reading['TITLEI_STATUS'].value_counts()


# In[33]:


sns.set_style('darkgrid')
_plt = sns.countplot(x='TITLEI_STATUS', data=reading)
_plt.set_title('School Title I status')
_plt.set_xticklabels(['Title I  schoolwide school','Not a Title I school','Title I schoolwide eligible- Title I targeted assistance program','Title I schoolwide eligible school-No program','Title I targeted assistance eligible school– No program','Title I targeted assistance school'])
_plt.set_xticklabels(_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('/Users/dansa/Documents/Title1_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# In[53]:


# Plot Histogram
sns.distplot(reading['Percent_Reading_Proficient'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(reading['Percent_Reading_Proficient'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Proficiency distribution')

fig = plt.figure()
res = stats.probplot(reading['Percent_Reading_Proficient'], plot=plt)
plt.show()

print("Skewness: %f" % reading['Percent_Reading_Proficient'].skew())
print("Kurtosis: %f" % reading['Percent_Reading_Proficient'].kurt())


# In[63]:


sns.set_style('darkgrid')
Type_plt = sns.countplot(x='School_type', data=reading)
Type_plt.set_title('School_types')
Type_plt.set_xticklabels(Type_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
Type_plt.set_xticklabels(["1-Regular School", "2-Special Education School", "3-Career and Technical School", "4-Alternative Education School"])
plt.savefig('/Users/dansa/Documents/Type_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# In[61]:


reading['School_type'].value_counts()


# ##### There are very few alternative, charter and magnet high schools in the dataset
# 
# Given that the number of alt, charter, magnet and sped schools are so small in the dataset we might want to exclude that columns in our model tuning later

# In[60]:


reading['Pct_certified_teachers']=(reading['Certified_FTE_teachers']/reading['FTE_teachcount']*100) # lets find the percent of certified teachers


# In[62]:


reading['Pct_noncertified_teachers']=(reading['Noncertified_FTE_teachers']/reading['FTE_teachcount']*100) # lets find the percent of noncertified teachers


# In[63]:


reading['Salary_perFTE_teacher'] = reading['SalaryforTeachers']/reading['FTE_teachers_count'] # Lets find the salary per FTE in each school


# In[64]:


reading['IPR_estimate'] = reading['Income_Poverty_ratio'] #Income poverty ratio is reported as a percent 


# In[65]:


#reading['Percent_Reading_Proficient'] = reading['Percent_Reading_Proficient']/100


# In[67]:


reading_clean=reading.drop(['School_ID','Certified_FTE_teachers', 'Noncertified_FTE_teachers','FTE_teachcount','FTE_teachers_count','SalaryforTeachers','Income_Poverty_ratio' ], axis=1)


# In[68]:


reading_clean.info()


# In[69]:


reading_clean.dtypes


# In[70]:


reading_clean.describe()


# In[74]:


sns.heatmap(reading_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[75]:


reading_clean.shape


# ### 3. Looking for Correlations and Visualizing
# We should calculate data correlations and plot a scatter matrix.
# 
# For training the ML models, we'll want to separate the Percent_Reading_Proficient from the rest of the data. But for investigating correlations, we'll want to include the target.

# In[72]:


reading_clean=reading_clean[['School_type', 'TITLEI_STATUS', 'TEACHERS', 'No.FARMS_students',
       'Total_enroll_students', 'Students_participate_SAT_ACT', 'IB_Indicator',
       'Total_IB_students', 'AP_Indicator', 'No.ofAP_courses_offer',
       'Students_enroll_inotherAP?', 'Total_AP_other_students',
       'Total_students_tookAP',
       'Pct_certified_teachers', 'Pct_noncertified_teachers',
       'Salary_perFTE_teacher', 'IPR_estimate','Percent_Reading_Proficient']]


# In[73]:


correlation_matrix = reading_clean.corr()


# In[74]:


correlation_matrix['Percent_Reading_Proficient'].sort_values(ascending=False)


# It seems like a few features (IPR_estimate, No.ofAP_courses_offer, Total_students_tookAP) have a weak to moderate positive correlation to the target (Percent_Reading_Proficient), and a couple are somewhat negatively correlated (School_type).
# 
# IPR_estimate is the Neighborhood Income Poverty Ratio.
# No.ofAP_courses_offer is the count of AP courses offered at the school.
# Total_students_tookAP is the number of students who took an AP course.
# School_type is refers to whether the school is a "1-Regular School, 2-Special Education School, 3-Career and Technical School and 4-Alternative Education School"
# 
# We can look at a heatmap of the correlations of all numeric features to visualize which features are correlated.

# In[93]:


# correlation matrix heatmap
plt.figure(figsize=(28,15))
corr_heatmap = sns.heatmap(correlation_matrix, annot=True, linewidths=0.2, center=0, cmap="RdYlGn")
corr_heatmap.set_title('Correlation Heatmap')
plt.savefig('/Users/dansa/Documents/corr_heatmap.png', dpi=300, bbox_inches='tight')


# In[90]:


#test
corr_pairs = {}
feats = correlation_matrix.columns
for x in feats:
    for y in feats:
        if x != y and np.abs(correlation_matrix[x][y]) >= 0.7:  # which pairs are strongely correlated?
            if (y, x) not in corr_pairs.keys():
                corr_pairs[(x, y)] = correlation_matrix[x][y]


# In[91]:


corr_pairs


# In[92]:


weaker_label = []
for pair in corr_pairs:
    if np.abs(correlation_matrix[pair[0]]['Percent_Reading_Proficient']) < np.abs(correlation_matrix[pair[1]]['Percent_Reading_Proficient']):
        weaker_label.append(pair[0])
    else:
        weaker_label.append(pair[1])


# In[93]:


poss_redundant_feats = set(weaker_label)
poss_redundant_feats


# In[111]:


cov_matrix = reading_clean.cov()
plt.figure(figsize=(28,15))
covar_heatmap = sns.heatmap(data=cov_matrix, cmap='coolwarm', linewidth=0.2) 
covar_heatmap.set_title('Covariance Heatmap')
plt.savefig('/Users/dansa/Documents/covar_heatmap.png', dpi=300, bbox_inches='tight')


# In[91]:


attrs = ['IPR_estimate','No.ofAP_courses_offer','Total_students_tookAP','Percent_Reading_Proficient']


# In[92]:


sns.set(style='ticks', color_codes=True)
_ = sns.pairplot(data=correlation_matrix[attrs], height=3, aspect=1, kind='scatter', plot_kws={'alpha':0.9})


# In[108]:


#from pandas.plotting import scatter_matrix
#attributes = ["Percent_Reading_Proficient", "IPR_estimate", "No.ofAP_courses_offer","Total_students_tookAP","Total_AP_other_students"]
#scatter_matrix(reading_clean[attributes], figsize=(12, 8))


# In[109]:


sns.jointplot(x="IPR_estimate", y="Percent_Reading_Proficient", data=reading_clean)


# ### ML prep
# #### Separate labels
# Let's separate out the target from the predicting features.

# In[103]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(reading_clean, test_size=0.2, random_state=0)


# In[105]:


df_X_train = train.drop(['Percent_Reading_Proficient'], axis=1)
df_y_train = train['Percent_Reading_Proficient'].copy()

df_X_test = test.drop(['Percent_Reading_Proficient'], axis=1)
df_y_test = test['Percent_Reading_Proficient'].copy()


# In[106]:


df_X_train.info()


# #### Transform Categorical Features
# Since these categorical features don't appear to have an inherent ordering, let's try encoding them as one-hot vectors for better ML performance.

# In[107]:


train_data_onehot = pd.get_dummies(df_X_train, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])
train_data_onehot.head()

test_data_onehot = pd.get_dummies(df_X_test, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])


# In[108]:


train_data_onehot.info()


# #### Scale Features
# We can check out the statistics for each feature, do they need to be normalized?

# In[112]:


train_data_onehot.describe()


# We'll probably want to scale these features using normalization or standardization.

# In[113]:


sc= StandardScaler()
train_scaler = sc.fit(train_data_onehot)
test_scaler = sc.fit(test_data_onehot)


# In[114]:


train_data_standardized = train_scaler.transform(train_data_onehot)
test_data_standardized = test_scaler.transform(test_data_onehot)


# In[115]:


print(train_data_standardized)
print(train_data_standardized.mean(axis=0))


# In[116]:


print(train_data_standardized.std(axis=0))


# In[117]:


train_data_std = pd.DataFrame(train_data_standardized, columns=train_data_onehot.columns)
test_data_std = pd.DataFrame(test_data_standardized, columns=test_data_onehot.columns)


# In[118]:


train_data_std.describe()


# That should work better, the standard deviation for each feature is 1 and the mean is ~0.

# ### Using a multiple linear regression model

# In[119]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_data_std, df_y_train)


# In[120]:


y_pred = regressor.predict(test_data_std)


# In[121]:


from termcolor import colored as cl


# In[122]:


print(cl('R-Squared :', attrs = ['bold']), regressor.score(test_data_std, df_y_test))


# In[123]:


# Method 2 Using Statsmodels
import statsmodels.api as sm
from scipy import stats


# In[124]:


y = list(df_y_train)


# In[125]:


X2 = sm.add_constant(train_data_std) 
# Define the model
est = sm.OLS(y, X2)
# Fit the model
est2 = est.fit()
print("summary()\n",est2.summary())
print("pvalues\n",est2.pvalues)
print("tvalues\n",est2.tvalues)
print("rsquared\n",est2.rsquared)
print("rsquared_adj\n",est2.rsquared_adj)


# In[126]:


sns.distplot(y_pred, hist = False, color = 'r', label = 'Predicted Values')
sns.distplot(df_y_test, hist = False, color = 'b', label = 'Actual Values')
plt.title('Actual vs Predicted Values', fontsize = 16)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend(loc = 'upper left', fontsize = 13)
plt.savefig('ap.png')


# In[127]:


df = pandas.DataFrame({'Actual':  df_y_test, 'Predicted': y_pred})
df


# In[128]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error( df_y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error( df_y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error( df_y_test, y_pred)))


# In[ ]:




