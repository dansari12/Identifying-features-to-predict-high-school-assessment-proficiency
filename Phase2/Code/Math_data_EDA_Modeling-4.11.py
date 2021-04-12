#!/usr/bin/env python
# coding: utf-8

# ### Phase 2: Conduct EDA and model construction using the master_math.csv file that contains all relevant features and target variable

# #### Importing all necessary libraries

# In[134]:


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


# In[135]:


#pip install --user scikit-learn


# In[136]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/MASTER


# #### Loading the data and reformatting the school id column

# In[137]:


master_math_new = pandas.read_csv("master_math.csv")
master_math_new['NCESSCH'] = master_math_new['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
master_math_new.head()


# In[138]:


master_math_new.shape


# #### Inspecting the data file

# In[139]:


master_math_new.columns


# Create a data frame with only the needed columns for further analysis

# In[140]:


math=pd.DataFrame(master_math_new, columns=[ 'NCESSCH', 'SCH_TYPE_x', 
       'TITLEI_STATUS','TEACHERS', 'FARMS_COUNT', 'Total_enroll_students',
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT',
       'FTE_teachers_count', 'SalaryforTeachers', 'Total_SAT_ACT_students',
       'SCH_IBENR_IND_new', 'Total_IB_students', 'SCH_APENR_IND_new',
       'SCH_APCOURSES', 'SCH_APMATHENR_IND_new','Total_AP_math_students',
       'Total_students_tookAP', 'SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG',
       'Total_Alg1_enroll_students', 'Total_Alg1_pass_students',
       'Income_Poverty_ratio','ALL_MTH00PCTPROF_1718_new'])


# In[141]:


math.head()


# In[142]:


math.rename(columns={'NCESSCH':'School_ID', 'SCH_TYPE_x':'School_type','FARMS_COUNT':'No.FARMS_students',
                       'SCH_FTETEACH_TOT':'FTE_teachcount','SCH_FTETEACH_CERT':'Certified_FTE_teachers','SCH_FTETEACH_NOTCERT':
                       'Noncertified_FTE_teachers','Total_SAT_ACT_students':'Students_participate_SAT_ACT','SCH_IBENR_IND_new':'IB_Indicator',
                       'SCH_APENR_IND_new':'AP_Indicator','SCH_APCOURSES':'No.ofAP_courses_offer','SCH_APMATHENR_IND_new':'Students_enroll_inAPMath?',
                       'SCH_MATHCLASSES_ALG':'No.ofAlg1classes','SCH_MATHCERT_ALG':'Alg1_taught_by_certmathteahcers',
                       'ALL_MTH00PCTPROF_1718_new':'Percent_Math_Proficient'}, inplace=True)


# In[143]:


math.describe()


# In[144]:


print(math.info())


# We have 14,008 entries and no null values in any column. There are 25 columns, but we can drop the school_id and we'll want to split off the Percent_Math_Proficient.
# The object type features should be strings.
# 
# Let's take a quick look at some of the data.

# In[145]:


math.hist(bins=50, figsize=(20,15))
plt.show()


# We can see that some features have most of their instances at or near zero and relatively few instances at higher values, in some cases much higher. Other features cluster close to zero and have long tails. We also see the percent_math_proficient is almost normally distributed.

# In[147]:


sns.set_style('darkgrid')
_plt = sns.countplot(x='TITLEI_STATUS', data=math)
_plt.set_title('School Title I status')
_plt.set_xticklabels(['Title I  schoolwide school','Not a Title I school','Title I schoolwide eligible- Title I targeted assistance program','Title I schoolwide eligible school-No program','Title I targeted assistance eligible school– No program','Title I targeted assistance school'])
_plt.set_xticklabels(_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('/Users/dansa/Documents/Title1_M_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# In[148]:


sns.set_style('darkgrid')
cert=math['Certified_FTE_teachers']
_plt = sns.distplot(cert)
_plt.set_title('Certified teachers')
_plt.set_xticklabels(_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('/Users/dansa/Documents/Certified_FTE_teachers_M_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# In[149]:


# Plot Histogram
sns.distplot(math['Percent_Math_Proficient'] , fit=norm);
# weights = np.ones_like(np.array(math['Percent_Math_Proficient']))/float(len(np.array(math['Percent_Math_Proficient'])))
# plt.hist(math['Percent_Math_Proficient'], weights=weights, bins = 100)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(math['Percent_Math_Proficient'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Proficiency distribution')

fig = plt.figure()
res = stats.probplot(math['Percent_Math_Proficient'], plot=plt)
plt.show()

print("Skewness: %f" % math['Percent_Math_Proficient'].skew())
print("Kurtosis: %f" % math['Percent_Math_Proficient'].kurt())


# In[150]:


sns.set_style('darkgrid')
Type_plt = sns.countplot(x='School_type', data=math)
Type_plt.set_title('School_types')
Type_plt.set_xticklabels(Type_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
Type_plt.set_xticklabels(["1-Regular School", "2-Special Education School", "3-Career and Technical School", "4-Alternative Education School"])
plt.savefig('/Users/dansa/Documents/Type_M_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# In[151]:


math['Pct_certified_teachers']=(math['Certified_FTE_teachers']/math['FTE_teachcount']*100) # lets find the percent of certified teachers


# In[152]:


math['Pct_noncertified_teachers']=(math['Noncertified_FTE_teachers']/math['FTE_teachcount']*100) # lets find the percent of noncertified teachers


# In[153]:


math['Salary_perFTE_teacher'] = math['SalaryforTeachers']/math['FTE_teachers_count'] # Lets find the salary per FTE in each school


# In[154]:


math['IPR_estimate'] = math['Income_Poverty_ratio'] #Income poverty ratio is reported as a percent 


# In[155]:


#math['Percent_Reading_Proficient'] = math['Percent_Reading_Proficient']/100


# In[156]:


math_clean=math.drop(['School_ID','Certified_FTE_teachers', 'Noncertified_FTE_teachers','FTE_teachcount','FTE_teachers_count','SalaryforTeachers','Income_Poverty_ratio' ], axis=1)


# In[157]:


math_clean.info()


# In[158]:


math_clean.describe()


# In[159]:


sns.heatmap(math_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[160]:


math_clean.shape


# ### 3. Looking for Correlations and Visualizing
# We should calculate data correlations and plot a scatter matrix.
# 
# For training the ML models, we'll want to separate the Percent_Math_Proficient from the rest of the data. But for investigating correlations, we'll want to include the target.

# In[161]:


math_clean=math_clean[['School_type', 'TITLEI_STATUS', 'TEACHERS', 'No.FARMS_students',
       'Total_enroll_students', 'Students_participate_SAT_ACT', 'IB_Indicator',
       'Total_IB_students', 'AP_Indicator', 'No.ofAP_courses_offer',
       'Students_enroll_inAPMath?', 'Total_AP_math_students','Total_students_tookAP', 
       'No.ofAlg1classes','Alg1_taught_by_certmathteahcers', 'Total_Alg1_enroll_students','Total_Alg1_pass_students',
       'Pct_certified_teachers', 'Pct_noncertified_teachers',
       'Salary_perFTE_teacher', 'IPR_estimate','Percent_Math_Proficient']]


# In[162]:


correlation_matrix = math_clean.corr()


# In[163]:


correlation_matrix['Percent_Math_Proficient'].sort_values(ascending=False)


# It seems like a few features (IPR_estimate, No.ofAP_courses_offer, Total_AP_math_students have a weak to moderate positive correlation to the target (Percent_Math_Proficient), and a couple are somewhat negatively correlated (School_type).
# 
# IPR_estimate is the Neighborhood Income Poverty Ratio.
# No.ofAP_courses_offer is the count of AP courses offered at the school.
# Total_AP_math_students is the number of students who took an AP math course.
# School_type is refers to whether the school is a "1-Regular School, 2-Special Education School, 3-Career and Technical School and 4-Alternative Education School"
# 
# We can look at a heatmap of the correlations of all numeric features to visualize which features are correlated.

# In[164]:


# correlation matrix heatmap
plt.figure(figsize=(28,15))
corr_heatmap = sns.heatmap(correlation_matrix, annot=True, linewidths=0.2, center=0, cmap="RdYlGn")
corr_heatmap.set_title('Correlation Heatmap')
plt.savefig('/Users/dansa/Documents/corr_heatmap.png', dpi=300, bbox_inches='tight')


# In[165]:


#test
corr_pairs = {}
feats = correlation_matrix.columns
for x in feats:
    for y in feats:
        if x != y and np.abs(correlation_matrix[x][y]) >= 0.7:  # which pairs are strongely correlated?
            if (y, x) not in corr_pairs.keys():
                corr_pairs[(x, y)] = correlation_matrix[x][y]


# In[166]:


corr_pairs


# In[167]:


weaker_label = []
for pair in corr_pairs:
    if np.abs(correlation_matrix[pair[0]]['Percent_Math_Proficient']) < np.abs(correlation_matrix[pair[1]]['Percent_Math_Proficient']):
        weaker_label.append(pair[0])
    else:
        weaker_label.append(pair[1])


# In[168]:


poss_redundant_feats = set(weaker_label)
poss_redundant_feats


# In[169]:


cov_matrix = math_clean.cov()
plt.figure(figsize=(28,15))
covar_heatmap = sns.heatmap(data=cov_matrix, cmap='coolwarm', linewidth=0.2) 
covar_heatmap.set_title('Covariance Heatmap')
plt.savefig('/Users/dansa/Documents/covar_heatmap.png', dpi=300, bbox_inches='tight')


# In[170]:


attrs = ['IPR_estimate','Total_AP_math_students','No.ofAP_courses_offer','Percent_Math_Proficient']


# In[171]:


sns.set(style='ticks', color_codes=True)
_ = sns.pairplot(data=correlation_matrix[attrs], height=3, aspect=1, kind='scatter', plot_kws={'alpha':0.9})


# In[172]:


sns.jointplot(x="IPR_estimate", y="Percent_Math_Proficient", data=math_clean)


# ### ML prep
# #### Separate labels
# Let's separate out the target from the predicting features.

# In[173]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(math_clean, test_size=0.2, random_state=0)


# In[174]:


train.shape


# In[175]:


test.shape


# In[176]:


math_clean.shape


# In[177]:


df_X_train = train.drop(['Percent_Math_Proficient'], axis=1)
df_y_train = train['Percent_Math_Proficient'].copy()

df_X_test = test.drop(['Percent_Math_Proficient'], axis=1)
df_y_test = test['Percent_Math_Proficient'].copy()


# In[178]:


df_X_train.info()


# #### Transform Categorical Features
# Since these categorical features don't appear to have an inherent ordering, let's try encoding them as one-hot vectors for better ML performance.

# In[179]:


train_data_onehot = pd.get_dummies(df_X_train, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])
train_data_onehot.head()

# one-hot the testing set as well
test_data_onehot = pd.get_dummies(df_X_test, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])


# In[180]:


train_data_onehot.info()


# #### Scale Features
# We can check out the statistics for each feature, do they need to be normalized?

# In[181]:


train_data_onehot.describe()


# In[182]:


sc= StandardScaler()
train_scaler = sc.fit(train_data_onehot)
test_scaler = sc.fit(test_data_onehot)
#print(train_scaler.mean_)
#print(train_scaler.scale_)


# In[183]:


train_data_standardized = train_scaler.transform(train_data_onehot)
test_data_standardized = test_scaler.transform(test_data_onehot)


# In[184]:


print(train_data_standardized)
print(train_data_standardized.mean(axis=0))


# In[185]:


print(train_data_standardized.std(axis=0))


# In[186]:


train_data_std = pd.DataFrame(train_data_standardized, columns=train_data_onehot.columns)
test_data_std = pd.DataFrame(test_data_standardized, columns=test_data_onehot.columns)


# In[187]:


train_data_std.describe()


# That should work better, the standard deviation for each feature is 1 and the mean is ~0.

# ### Using a multiple linear regression model

# In[188]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_data_std, df_y_train)


# In[189]:


y_pred = regressor.predict(test_data_std)


# In[190]:


from termcolor import colored as cl


# In[191]:


print(cl('R-Squared :', attrs = ['bold']), regressor.score(test_data_std, df_y_test))


# In[192]:


# Method 2 Using Statsmodels
import statsmodels.api as sm
from scipy import stats


# In[193]:


y = list(df_y_train)


# In[194]:


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


# In[195]:


sns.distplot(y_pred, hist = False, color = 'r', label = 'Predicted Values')
sns.distplot(df_y_test, hist = False, color = 'b', label = 'Actual Values')
plt.title('Actual vs Predicted Values', fontsize = 16)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend(loc = 'upper left', fontsize = 13)
plt.savefig('ap.png')


# In[196]:


df = pandas.DataFrame({'Actual':  df_y_test, 'Predicted': y_pred})
df


# In[197]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(df_y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(df_y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df_y_test, y_pred)))


# In[ ]:




