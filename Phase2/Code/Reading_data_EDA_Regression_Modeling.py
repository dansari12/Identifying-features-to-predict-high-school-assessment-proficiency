#!/usr/bin/env python
# coding: utf-8

# ### Phase 2: Conducting EDA and  Regression model construction using the master_reading.csv file that contains all necessary features and target variable

# In[179]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/MASTER


# In[180]:


#pip install --user scikit-learn


# #### Importing all necessary libraries

# In[181]:


import pandas
pandas.__version__
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Loading the data and reformatting the school id column

# In[182]:


master_reading_new = pandas.read_csv("master_reading.csv")
master_reading_new['NCESSCH'] = master_reading_new['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
master_reading_new.head()


# In[183]:


master_reading_new.shape


# #### Inspecting the data file

# In[184]:


master_reading_new.columns


# Create a data frame with only the needed columns for further analysis

# In[185]:


reading=pd.DataFrame(master_reading_new, columns=[ 'NCESSCH','SCH_TYPE_x', 
       'TITLEI_STATUS','TEACHERS', 'FARMS_COUNT', 'Total_enroll_students',
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT',
       'FTE_teachers_count', 'SalaryforTeachers', 'Total_SAT_ACT_students',
       'SCH_IBENR_IND_new', 'Total_IB_students', 'SCH_APENR_IND_new',
       'SCH_APCOURSES', 'SCH_APOTHENR_IND_new', 'Total_AP_other_students',
       'Total_students_tookAP', 'Income_Poverty_ratio','ALL_RLA00PCTPROF_1718_new'])


# In[186]:


reading.head()


# ##### Rename columns

# In[187]:


reading.rename(columns={'NCESSCH':'School_ID', 'SCH_TYPE_x':'School_type','FARMS_COUNT':'No.FARMS_students',
                       'SCH_FTETEACH_TOT':'FTE_teachcount','SCH_FTETEACH_CERT':'Certified_FTE_teachers','SCH_FTETEACH_NOTCERT':
                       'Noncertified_FTE_teachers','Total_SAT_ACT_students':'Students_participate_SAT_ACT','SCH_IBENR_IND_new':'IB_Indicator','SCH_APENR_IND_new':'AP_Indicator',
                        'SCH_APCOURSES':'No.ofAP_courses_offer','SCH_APOTHENR_IND_new':'Students_enroll_inotherAP?','ALL_RLA00PCTPROF_1718_new':'Percent_Reading_Proficient'}, inplace=True)


# In[188]:


reading.head().T


# In[189]:


reading.describe()


# In[190]:


print(reading.info())


# We have 14,509 entries and no null values in any column. There are 21 columns, but we can drop the school_id and we'll want to split off the Percent_Reading_Proficient.
# The object type features should be strings.
# 
# Let's take a quick look at some of the data.

# In[191]:


reading.hist(bins=50, figsize=(20,15))
plt.show()


# We can see that some features have most of their instances at or near zero and relatively few instances at higher values, in some cases much higher. Other features cluster close to zero and have long tails. We also see the percent_reading_proficient is almost normally distributed.

# In[192]:


reading['TITLEI_STATUS'].value_counts()


# In[193]:


sns.set_style('darkgrid')
_plt = sns.countplot(x='TITLEI_STATUS', data=reading)
_plt.set_title('School Title I status')
_plt.set_xticklabels(['Title I  schoolwide school','Not a Title I school','Title I schoolwide eligible- Title I targeted assistance program','Title I schoolwide eligible school-No program','Title I targeted assistance eligible school– No program','Title I targeted assistance school'])
_plt.set_xticklabels(_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('/Users/dansa/Documents/Title1_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# Most of the high schools appear to be Title 1 schoolwide or not Title 1 funded schools

# In[194]:


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


# In[195]:


# sns.set_style('darkgrid')
# Type_plt = sns.countplot(x='School_type', data=reading)
# Type_plt.set_title('School_types')
# Type_plt.set_xticklabels(Type_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
# Type_plt.set_xticklabels(["1-Regular School", "2-Special Education School", "3-Career and Technical School", "4-Alternative Education School"])
# plt.savefig('/Users/dansa/Documents/Type_dist.png', dpi=300, bbox_inches='tight')
# plt.show()


# Lets find the percent of certified and noncertified teachers

# In[196]:


reading['Pct_certified_teachers']=(reading['Certified_FTE_teachers']/reading['FTE_teachcount']*100) # lets find the percent of certified teachers


# In[197]:


reading['Pct_noncertified_teachers']=(reading['Noncertified_FTE_teachers']/reading['FTE_teachcount']*100) # lets find the percent of noncertified teachers


# Lets find the salary per FTE teachers

# In[198]:


reading['Salary_perFTE_teacher'] = reading['SalaryforTeachers']/reading['FTE_teachers_count'] # Lets find the salary per FTE in each school


# In[199]:


reading['IPR_estimate'] = reading['Income_Poverty_ratio'] #Income poverty ratio is reported as a percent 


# Lets drop the unwanted columns

# In[200]:


reading_clean=reading.drop(['School_ID','Certified_FTE_teachers', 'Noncertified_FTE_teachers','FTE_teachcount','FTE_teachers_count','SalaryforTeachers','Income_Poverty_ratio' ], axis=1)


# In[201]:


reading_clean.info()


# Change school type from int to float

# In[202]:


reading_clean['School_type'] = reading_clean['School_type'].astype(float)


# In[203]:


reading_clean.describe()


# Check for missing or null values

# In[204]:


sns.heatmap(reading_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[205]:


reading_clean.shape


# ### 3. Looking for Correlations and Visualizing
# We should calculate data correlations and plot a scatter matrix.
# 
# For training the ML models, we'll want to separate the Percent_Reading_Proficient from the rest of the data. But for investigating correlations, we'll want to include the target.

# In[206]:


reading_clean=reading_clean[['School_type', 'TITLEI_STATUS', 'TEACHERS', 'No.FARMS_students',
       'Total_enroll_students', 'Students_participate_SAT_ACT', 'IB_Indicator',
       'Total_IB_students', 'AP_Indicator', 'No.ofAP_courses_offer',
       'Students_enroll_inotherAP?', 'Total_AP_other_students',
       'Total_students_tookAP',
       'Pct_certified_teachers', 'Pct_noncertified_teachers',
       'Salary_perFTE_teacher', 'IPR_estimate','Percent_Reading_Proficient']]


# In[207]:


correlation_matrix = reading_clean.corr()


# In[208]:


correlation_matrix['Percent_Reading_Proficient'].sort_values(ascending=False)


# It seems like a few features (IPR_estimate, No.ofAP_courses_offer, Total_students_tookAP) have a moderate to weak positive correlation to the target (Percent_Reading_Proficient), and a couple are somewhat negatively correlated (School_type).
# 
# IPR_estimate is the Neighborhood Income Poverty Ratio.
# No.ofAP_courses_offer is the count of AP courses offered at the school.
# Total_students_tookAP is the number of students who took an AP course.
# School_type is refers to whether the school is a "1-Regular School, 2-Special Education School, 3-Career and Technical School and 4-Alternative Education School"
# 
# We can look at a heatmap of the correlations of all numeric features to visualize which features are correlated.

# In[209]:


# correlation matrix heatmap
plt.figure(figsize=(28,15))
corr_heatmap = sns.heatmap(correlation_matrix, annot=True, linewidths=0.2, center=0, cmap="RdYlGn")
corr_heatmap.set_title('Correlation Heatmap')
plt.savefig('/Users/dansa/Documents/corr_heatmap.png', dpi=300, bbox_inches='tight')


# In[210]:


#test
corr_pairs = {}
feats = correlation_matrix.columns
for x in feats:
    for y in feats:
        if x != y and np.abs(correlation_matrix[x][y]) >= 0.7:  # which pairs are strongely correlated?
            if (y, x) not in corr_pairs.keys():
                corr_pairs[(x, y)] = correlation_matrix[x][y]


# In[211]:


corr_pairs


# In[212]:


weaker_label = []
for pair in corr_pairs:
    if np.abs(correlation_matrix[pair[0]]['Percent_Reading_Proficient']) < np.abs(correlation_matrix[pair[1]]['Percent_Reading_Proficient']):
        weaker_label.append(pair[0])
    else:
        weaker_label.append(pair[1])


# In[213]:


poss_redundant_feats = set(weaker_label)
poss_redundant_feats


# In[214]:


cov_matrix = reading_clean.cov()
plt.figure(figsize=(28,15))
covar_heatmap = sns.heatmap(data=cov_matrix, cmap='coolwarm', linewidth=0.2) 
covar_heatmap.set_title('Covariance Heatmap')
plt.savefig('/Users/dansa/Documents/covar_heatmap.png', dpi=300, bbox_inches='tight')


# In[215]:


attrs = ['IPR_estimate','No.ofAP_courses_offer','Total_students_tookAP','Percent_Reading_Proficient']


# In[216]:


sns.set(style='ticks', color_codes=True)
_ = sns.pairplot(data=correlation_matrix[attrs], height=3, aspect=1, kind='scatter', plot_kws={'alpha':0.9})


# In[217]:


#from pandas.plotting import scatter_matrix
#attributes = ["Percent_Reading_Proficient", "IPR_estimate", "No.ofAP_courses_offer","Total_students_tookAP","Total_AP_other_students"]
#scatter_matrix(reading_clean[attributes], figsize=(12, 8))


# In[218]:


sns.jointplot(x="IPR_estimate", y="Percent_Reading_Proficient", data=reading_clean)


# ### ML prep
# #### Separate labels
# Let's separate out the target from the predicting features.

# In[219]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(reading_clean, test_size=0.2, random_state=0)


# In[220]:


df_X_train = train.drop(['Percent_Reading_Proficient'], axis=1)
df_y_train = train['Percent_Reading_Proficient'].copy()

df_X_test = test.drop(['Percent_Reading_Proficient'], axis=1)
df_y_test = test['Percent_Reading_Proficient'].copy()


# In[221]:


df_X_train.info()


# #### Transform Categorical Features
# Since these categorical features don't appear to have an inherent ordering, let's try encoding them as one-hot vectors for better ML performance.

# In[222]:


train_data_onehot = pd.get_dummies(df_X_train, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])
train_data_onehot.head()

test_data_onehot = pd.get_dummies(df_X_test, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])


# In[223]:


#train_data_onehot.info()


# #### Scale Features
# We can check out the statistics for each feature, do they need to be normalized?

# In[224]:


train_data_onehot.describe()


# We'll probably want to scale these features using normalization or standardization.

# In[225]:


sc= StandardScaler()


# In[226]:


train_data_standardized = sc.fit_transform(train_data_onehot)
test_data_standardized = sc.transform(test_data_onehot)


# In[227]:


print(train_data_standardized)
print(train_data_standardized.mean(axis=0))


# In[228]:


print(train_data_standardized.std(axis=0))


# In[229]:


train_data_std = pd.DataFrame(train_data_standardized, columns=train_data_onehot.columns)
test_data_std = pd.DataFrame(test_data_standardized, columns=test_data_onehot.columns)


# In[230]:


#train_data_std.describe()


# That should work better, the standard deviation for each feature is 1 and the mean is ~0.

# ### Because are target variable is continuous we will utilize regression models to train and predict the proficiency for reading

# ### Using a multiple linear regression model

# In[231]:


#Linear Regression
lr = linear_model.LinearRegression()
lr.fit(train_data_std, df_y_train)
y_pred = np.round(lr.predict(test_data_std))
meansqr=[]
meanabs=[]
r2=[]
rmse=[]
meansqr.append(mean_squared_error(df_y_test, y_pred))
meanabs.append(abs(df_y_test-y_pred).mean())
rmse.append(np.sqrt(mean_squared_error(df_y_test, y_pred)))
r2.append(r2_score(df_y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(df_y_test, y_pred))
print("Mean absolute error: %.2f"% mean_absolute_error(df_y_test, y_pred))
print('Root Mean Squared Error:%.2f'% np.sqrt(mean_squared_error( df_y_test, y_pred)))
print("r2 score: %.2f"% r2_score(df_y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,df_y_test)


# In[232]:


df = pandas.DataFrame({'Actual':  df_y_test, 'Predicted': y_pred})
df.head()


# In[233]:


sns.distplot(y_pred, hist = False, color = 'r', label = 'Predicted Values')
sns.distplot(df_y_test, hist = False, color = 'b', label = 'Actual Values')
plt.title('Actual vs Predicted Values', fontsize = 16)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend(loc = 'upper left', fontsize = 13)
plt.savefig('ap.png')


# ### Using a Random Forest regression model

# In[234]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_data_std, df_y_train);
y_pred =np.round(rf.predict(test_data_std))
meansqr.append(mean_squared_error(df_y_test, y_pred))
meanabs.append(abs(df_y_test-y_pred).mean())
rmse.append(np.sqrt(mean_squared_error(df_y_test, y_pred)))
r2.append(r2_score(df_y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(df_y_test, y_pred))
print("Mean absolute error: %.2f"% mean_absolute_error(df_y_test, y_pred))
print('Root Mean Squared Error:%.2f'% np.sqrt(mean_squared_error( df_y_test, y_pred)))
print("r2 score: %.2f"% r2_score(df_y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,df_y_test)


# In[235]:


df2 = pandas.DataFrame({'Actual':  df_y_test, 'Predicted': y_pred})
df2.head()


# In[236]:


sns.distplot(y_pred, hist = False, color = 'r', label = 'Predicted Values')
sns.distplot(df_y_test, hist = False, color = 'b', label = 'Actual Values')
plt.title('Actual vs Predicted Values', fontsize = 16)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend(loc = 'upper left', fontsize = 13)
plt.savefig('ap.png')


# ### Using a SVM regression model

# In[237]:


#svm regressor
from sklearn.svm import SVR
svr=SVR(kernel="linear",epsilon=1.0,degree=3)
svr.fit(train_data_std, df_y_train)
y_pred=svr.predict(test_data_std)
meansqr.append(mean_squared_error(df_y_test, y_pred))
meanabs.append(abs(df_y_test-y_pred).mean())
rmse.append(np.sqrt(mean_squared_error(df_y_test, y_pred)))
r2.append(r2_score(df_y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(df_y_test, y_pred))
print("Mean absolute error: %.2f"% mean_absolute_error(df_y_test, y_pred))
print('Root Mean Squared Error:%.2f'% np.sqrt(mean_squared_error( df_y_test, y_pred)))
print("r2 score: %.2f"% r2_score(df_y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,df_y_test)


# In[238]:


df3 = pandas.DataFrame({'Actual':  df_y_test, 'Predicted': y_pred})
df3.head()


# In[239]:


sns.distplot(y_pred, hist = False, color = 'r', label = 'Predicted Values')
sns.distplot(df_y_test, hist = False, color = 'b', label = 'Actual Values')
plt.title('Actual vs Predicted Values', fontsize = 16)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend(loc = 'upper left', fontsize = 13)
plt.savefig('ap.png')


# ### Using a Knn model

# In[240]:


#knearest neighbhors
from sklearn import neighbors
n_neighbors=5
knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
knn.fit(train_data_std, df_y_train)
y_pred=knn.predict(test_data_std)
meansqr.append(mean_squared_error(df_y_test, y_pred))
meanabs.append(abs(df_y_test-y_pred).mean())
rmse.append(np.sqrt(mean_squared_error(df_y_test, y_pred)))
r2.append(r2_score(df_y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(df_y_test, y_pred))
print("Mean absolute error: %.2f"% mean_absolute_error(df_y_test, y_pred))
print('Root Mean Squared Error:%.2f'% np.sqrt(mean_squared_error( df_y_test, y_pred)))
print("r2 score: %.2f"% r2_score(df_y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,df_y_test)


# In[241]:


df4 = pandas.DataFrame({'Actual':  df_y_test, 'Predicted': y_pred})
df4.head()


# In[242]:


sns.distplot(y_pred, hist = False, color = 'r', label = 'Predicted Values')
sns.distplot(df_y_test, hist = False, color = 'b', label = 'Actual Values')
plt.title('Actual vs Predicted Values', fontsize = 16)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend(loc = 'upper left', fontsize = 13)
plt.savefig('ap.png')


# Let's plot the key metrics

# In[243]:


objects=('LinearReg','RandForReg','SVMregg','Knn')
plt.bar(np.arange(len(meansqr)),meansqr)
plt.xticks(np.arange(len(meansqr)), objects)
plt.title('Mean Square Error values for the different models')
plt.xlabel('Model Names', fontsize = 12)
plt.ylabel('Mean square error', fontsize = 12)
plt.show()
plt.bar(np.arange(len(meanabs)),meanabs)
plt.xticks(np.arange(len(meanabs)), objects)
plt.title('Mean Absolute Error for the different models')
plt.xlabel('Model Names', fontsize = 12)
plt.ylabel('Mean absolute error', fontsize = 12)
plt.show()
plt.bar(np.arange(len(rmse)),rmse)
plt.xticks(np.arange(len(rmse)), objects)
plt.title('Root mean squared error for the different models')
plt.xlabel('Model Names', fontsize = 12)
plt.ylabel('Root mean square error', fontsize = 12)
plt.show()
plt.bar(np.arange(len(r2)),r2)
plt.xticks(np.arange(len(r2)), objects)
plt.title('r^2 score for the different models')
plt.xlabel('Model Names', fontsize = 12)
plt.ylabel('r^2 score', fontsize = 12)
plt.show()


# In[244]:


#r2


# In[245]:


from collections import OrderedDict
# Create a dictionary to store the train and test scores, best alpha values
scores = {'Regression models': ['Linear','Random Forest','SVM','Knn'],
         'RMSE': [20.63, 18.90, 20.77, 20.97],
         'R2': [0.22, 0.35, 0.21, 0.20]}
scores = OrderedDict(scores)

# Create a dataframe from the dictionary
reg_models_scores = pd.DataFrame.from_dict(scores)
reg_models_scores


# R_2 score or R squared coefficient is a statistical measure which indicates the percentage of the variance in the dependent variable (i.e. proficiency score) that the independent variables (school level features) explain collectively. R-squared measures the strength of the relationship between the model and the dependent variable on a 0 – 100% scale. If the r^2 score is 0 that implies a bad model and r^2 is equal to 1 for an ideal model. 
# 
# Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far the data points are from the line and is a measure of how spread out these residuals are.
# 
# Across all the different regression models we got the lowest rmse of 18.90 and the highest r^2 score of around 0.35 with the random forest regression model which implies that only 35% of the variance in proficiency scores for reading was accounted for by the selected features.
# 

# In[ ]:




