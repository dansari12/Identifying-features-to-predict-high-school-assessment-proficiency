#!/usr/bin/env python
# coding: utf-8

# ### Phase 3: Conduct EDA and classification model construction using the master_reading.csv file that contains all necessary features and target variable

# #### Importing all necessary libraries

# In[2]:


import pandas
pandas.__version__

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV, learning_curve, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[3]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/MASTER


# #### Loading the data and reformatting the school id column

# In[26]:


master_reading_new = pandas.read_csv("master_reading.csv")
master_reading_new['NCESSCH'] = master_reading_new['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
master_reading_new.head()


# In[27]:


master_reading_new.shape


# #### Inspecting the data file

# In[28]:


master_reading_new.columns


# Create a data frame with only the needed columns for further analysis

# In[29]:


reading=pd.DataFrame(master_reading_new, columns=[ 'NCESSCH','NAME','SCH_TYPE_x', 
       'TITLEI_STATUS','TEACHERS', 'FARMS_COUNT', 'Total_enroll_students',
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT',
       'FTE_teachers_count', 'SalaryforTeachers', 'Total_SAT_ACT_students',
       'SCH_IBENR_IND_new', 'Total_IB_students', 'SCH_APENR_IND_new',
       'SCH_APCOURSES', 'SCH_APOTHENR_IND_new', 'Total_AP_other_students',
       'Total_students_tookAP', 'Income_Poverty_ratio','ALL_RLA00PCTPROF_1718_new'])


# In[30]:


reading.head()


# ##### Rename columns

# In[31]:


reading.rename(columns={'NCESSCH':'School_ID', 'SCH_TYPE_x':'School_type','FARMS_COUNT':'No.FARMS_students',
                       'SCH_FTETEACH_TOT':'FTE_teachcount','SCH_FTETEACH_CERT':'Certified_FTE_teachers','SCH_FTETEACH_NOTCERT':
                       'Noncertified_FTE_teachers','Total_SAT_ACT_students':'Students_participate_SAT_ACT','SCH_IBENR_IND_new':'IB_Indicator','SCH_APENR_IND_new':'AP_Indicator',
                        'SCH_APCOURSES':'No.ofAP_courses_offer','SCH_APOTHENR_IND_new':'Students_enroll_inotherAP?','ALL_RLA00PCTPROF_1718_new':'Percent_Reading_Proficient'}, inplace=True)


# In[32]:


reading.describe().T


# ##### IB and Total_students_tookAP has some missing values, lets clean than up

# In[33]:


counts = reading['IB_Indicator'].value_counts().to_dict()
counts = reading['Total_students_tookAP'].value_counts().to_dict()
#print (counts)


# In[34]:


reading.shape


# In[35]:


reading=reading[reading['Total_students_tookAP']!=-10]


# In[36]:


reading=reading[reading['IB_Indicator']!=-6]


# In[37]:


counts = reading['IB_Indicator'].value_counts().to_dict()
#print (counts)


# ##### Let take a closer look at the dataframe and datatypes

# In[38]:


print(reading.info())


# We have 14,507 entries and no null values in any column. There are 22 columns, but we can drop the school_id and name and we'll want to split off the Percent_Reading_Proficient.
# The object type features should be strings.
# 
# Let's take a quick look at some of the data.

# In[39]:


reading.hist(bins=50, figsize=(20,15))
plt.show()


# We can see that some features have most of their instances at or near zero and relatively few instances at higher values, in some cases much higher. Other features cluster close to zero and have long tails. We also see the percent_reading_proficient is almost normally distributed.

# In[40]:


reading_EDA=reading[['NAME','Income_Poverty_ratio','Percent_Reading_Proficient']]


# In[41]:


highest_proficiency=reading_EDA.sort_values(by=['Percent_Reading_Proficient'], inplace=True, ascending=False)


# In[42]:


reading_EDA.head()


# In[43]:


reading_high = reading_EDA.head(5)
reading_high.shape


# In[44]:


plt.style.use('ggplot')

plt.barh(reading_high.NAME, reading_high.Percent_Reading_Proficient, color='green')
plt.ylabel("School Names")
plt.xlabel("Percent Reading Proficient")
plt.title("Top 5 high schools with highest percent reading proficiency")
plt.xlim(0.0, 100.0)

plt.show()


# In[45]:


lowest_proficiency=reading_EDA.sort_values(by=['Percent_Reading_Proficient'], inplace=True, ascending=True)


# In[46]:


reading_EDA.head()


# In[47]:


reading_low = reading_EDA.head(5)
reading_low.shape


# In[48]:


plt.style.use('ggplot')

plt.barh(reading_low.NAME, reading_low.Percent_Reading_Proficient, color='green')
plt.ylabel("School Names")
plt.xlabel("Percent Reading Proficient")
plt.title("Top 10 high schools with highest percent reading proficiency")
plt.xlim(0.0, 100.0)

plt.show()


# In[49]:


sns.set_style('darkgrid')
_plt = sns.countplot(x='TITLEI_STATUS', data=reading)
_plt.set_title('School Title I status')
_plt.set_xticklabels(['Title I  schoolwide school','Not a Title I school','Title I schoolwide eligible- Title I targeted assistance program','Title I schoolwide eligible school-No program','Title I targeted assistance eligible school– No program','Title I targeted assistance school'])
_plt.set_xticklabels(_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('/Users/dansa/Documents/Title1_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# Most of the high schools appear to be Title 1 schoolwide or not Title 1 funded schools

# ##### Let's look at the distribution of the proficiency percentages

# In[50]:


from scipy.stats import norm
# Plot Histogram
sns.distplot(reading['Percent_Reading_Proficient'] , bins=20, kde=False, fit=stats.norm);

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


# #### Lets find the percent of certified and non-certified teachers

# In[51]:


reading['Pct_certified_teachers']=(reading['Certified_FTE_teachers']/reading['FTE_teachcount']*100)


# In[52]:


reading['Pct_noncertified_teachers']=(reading['Noncertified_FTE_teachers']/reading['FTE_teachcount']*100) 


# #### Lets find the salary per FTE in each school

# In[53]:


reading['Salary_perFTE_teacher'] = reading['SalaryforTeachers']/reading['FTE_teachers_count'] 


# In[54]:


reading['IPR_estimate'] = reading['Income_Poverty_ratio'] #Income poverty ratio is reported as a percent 


# ##### Lets drop the unwanted columns

# In[55]:


reading_clean=reading.drop(['School_ID','NAME','Certified_FTE_teachers', 'Noncertified_FTE_teachers','FTE_teachcount','FTE_teachers_count','SalaryforTeachers','Income_Poverty_ratio' ], axis=1)


# In[56]:


reading_clean.info()


# ##### Change school type from int to float

# In[57]:


reading_clean['School_type'] = reading_clean['School_type'].astype(float)


# In[58]:


reading_clean.describe()


# ##### Check for missing or null values

# In[44]:


sns.heatmap(reading_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[45]:


reading_clean.shape

#### Let's create Labels for Reading Proficiency based on the percent distribution of the schools
# In[62]:


reading_clean[['Percent_Reading_Proficient']].describe()


# In[63]:


reading_clean['Percent_Reading_Proficient'].plot(kind='hist')


# In[64]:


boxplot = reading_clean.boxplot(column=['Percent_Reading_Proficient'])
boxplot.plot()

plt.show()


# In[65]:


mu = 200
sigma = 25
n_bins = 5


fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(reading_clean.Percent_Reading_Proficient, n_bins, density=True, histtype='step',
                           cumulative=True, label='Empirical')

# Add a line showing the expected distribution.
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]

ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

# Overlay a reversed cumulative histogram.
ax.hist(reading_clean.Percent_Reading_Proficient, bins=bins, density=True, histtype='step', cumulative=-1,
        label='Reversed emp.')

# tidy up the figure
ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative step histograms')
ax.set_xlabel('Percent Math Proficiency')
ax.set_ylabel('Likelihood of occurrence')

plt.show()


# In[66]:


# getting data of the histogram
count, bins_count = np.histogram(reading_clean.Percent_Reading_Proficient, bins=10)
  
# finding the Probability Distribution Function of the histogram using count values
pdf = count / sum(count)
  
# using numpy np.cumsum to calculate the Cumulative Distribution Function
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
  
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()


# In[67]:


fig, ax = plt.subplots()
reading_clean['Percent_Reading_Proficient'].hist(bins=30, color='#A9C5D3', 
                             edgecolor='black', grid=False)
ax.set_title('Percent of Schools with at or above Reading Proficiency Histogram', fontsize=12)
ax.set_xlabel('Percent of Reading Proficiency', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
#Reference: https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b


# In[68]:


quantile_list = [0, .25, .5, .75, 1.]
#quantile_list = [0, .20, .80, 1.]
quantiles = reading_clean['Percent_Reading_Proficient'].quantile(quantile_list)
quantiles


# In[69]:


fig, ax = plt.subplots()
reading_clean['Percent_Reading_Proficient'].hist(bins=30, color='#A9C5D3', 
                             edgecolor='black', grid=False)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)

ax.set_title('Percentages of Reading Proficiency across all High Schools', fontsize=12)
ax.set_xlabel('Percent of Reading Proficiency', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)


# In[70]:


pd.qcut(reading_clean['Percent_Reading_Proficient'], q=4, precision = 0).value_counts()


# In[71]:


quantile_labels = ['Low', 'Moderate','High','Very High']
#quantile_labels = ['Lower 20', 'Middle','Upper 20']
quantile_numeric = [1,2,3,4]
reading_clean['Percent_Reading_Proficient_quantile_range'] = pd.qcut(
                                            reading_clean['Percent_Reading_Proficient'], 
                                            q=quantile_list)
reading_clean['Percent_Reading_Proficient_quantile_label'] = pd.qcut(
                                            reading_clean['Percent_Reading_Proficient'], 
                                            q=quantile_list,       
                                            labels=quantile_labels)
reading_clean['Percent_Reading_Proficient_quantile_encoded'] = pd.qcut(
                                            reading_clean['Percent_Reading_Proficient'], 
                                            q=quantile_list,
                                            labels=quantile_numeric,
                                            precision=0)

reading_clean.head()


# In[74]:


reading_clean['Percent_Reading_Proficient_quantile_label'].value_counts(ascending=True)


# In[75]:


reading_clean['Percent_Reading_Proficient_quantile_encoded'] = reading_clean['Percent_Reading_Proficient_quantile_encoded'].astype(float)


# ### Looking for Correlations and Visualizing
# We should calculate data correlations and plot a scatter matrix.
# 
# For training the ML models, we'll want to separate the Percent_Reading_Proficient from the rest of the data. But for investigating correlations, we'll want to include the target.

# In[77]:


reading_clean1=reading_clean[['School_type', 'TITLEI_STATUS', 'TEACHERS', 'No.FARMS_students',
       'Total_enroll_students', 'Students_participate_SAT_ACT', 'IB_Indicator',
       'Total_IB_students', 'AP_Indicator', 'No.ofAP_courses_offer',
       'Students_enroll_inotherAP?', 'Total_AP_other_students',
       'Total_students_tookAP',
       'Pct_certified_teachers', 'Pct_noncertified_teachers',
       'Salary_perFTE_teacher', 'IPR_estimate','Percent_Reading_Proficient_quantile_encoded']]


# In[78]:


reading_clean1.head()


# In[79]:


correlation_matrix = reading_clean1.corr()


# In[80]:


correlation_matrix['Percent_Reading_Proficient_quantile_encoded'].sort_values(ascending=False)


# It seems like a few features (IPR_estimate, No.ofAP_courses_offer, Total_students_tookAP) have a weak to moderate positive correlation to the target (Percent_Reading_Proficient), and a couple are somewhat negatively correlated (School_type).
# 
# * IPR_estimate is the Neighborhood Income Poverty Ratio.
# * No.ofAP_courses_offer is the count of AP courses offered at the school.
# * Total_students_tookAP is the number of students who took an AP course.
# * School_type is refers to whether the school is a "1-Regular School, 2-Special Education School, 3-Career and Technical School and 4-Alternative Education School"
# 
# We can look at a heatmap of the correlations of all numeric features to visualize which features are correlated.

# In[81]:


# correlation matrix heatmap
plt.figure(figsize=(28,15))
corr_heatmap = sns.heatmap(correlation_matrix, annot=True, linewidths=0.2, center=0, cmap="RdYlGn")
corr_heatmap.set_title('Correlation Heatmap')
plt.savefig('/Users/dansa/Documents/corr_heatmap.png', dpi=300, bbox_inches='tight')


# In[84]:


#test
corr_pairs = {}
feats = correlation_matrix.columns
for x in feats:
    for y in feats:
        if x != y and np.abs(correlation_matrix[x][y]) >= 0.7:  # which pairs are strongely correlated?
            if (y, x) not in corr_pairs.keys():
                corr_pairs[(x, y)] = correlation_matrix[x][y]


# In[85]:


corr_pairs


# In[86]:


attrs = ['IPR_estimate','No.ofAP_courses_offer','Total_students_tookAP','Percent_Reading_Proficient_quantile_encoded']


# In[87]:


sns.set(style='ticks', color_codes=True)
_ = sns.pairplot(data=correlation_matrix[attrs], height=3, aspect=1, kind='scatter', plot_kws={'alpha':0.9})


# In[88]:


sns.jointplot(x="IPR_estimate", y="Percent_Reading_Proficient_quantile_encoded", data=reading_clean1)


# In[89]:


sns.pairplot(reading_clean1, hue = 'Percent_Reading_Proficient_quantile_encoded',vars = ['IPR_estimate','No.ofAP_courses_offer','Total_students_tookAP','Total_AP_other_students'] )


# ### ML prep
# #### Separate labels
# Let's separate out the target from the predicting features.

# In[90]:


X = reading_clean1.drop('Percent_Reading_Proficient_quantile_encoded', axis=1)
y = reading_clean1.Percent_Reading_Proficient_quantile_encoded


# In[91]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[92]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Transform Categorical Features
# Since these categorical features don't appear to have an inherent ordering, let's try encoding them as one-hot vectors for better ML performance.

# In[93]:


train_data_onehot = pd.get_dummies(X_train, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])
train_data_onehot.head()

test_data_onehot = pd.get_dummies(X_test, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])


# In[94]:


test_data_onehot.head()


# #### Scale Features
# We can check out the statistics for each feature, do they need to be normalized?

# In[95]:


train_data_onehot.describe()


# We'll probably want to scale these features using normalization or standardization.

# In[96]:


sc= StandardScaler()

X_train_scaled = sc.fit_transform(train_data_onehot)

X_test_scaled = sc.transform(test_data_onehot)


# In[97]:


print(X_train_scaled)
print(X_train_scaled.mean(axis=0))


# In[98]:


print(X_train_scaled.std(axis=0))


# In[99]:


X_train_std = pd.DataFrame(X_train_scaled, columns=train_data_onehot.columns)
X_test_std = pd.DataFrame(X_test_scaled, columns=test_data_onehot.columns)


# In[100]:


#X_train_std.describe()


# That should work better, the standard deviation for each feature is 1 and the mean is ~0.

# ### Classification models

# #### Logistic Regression

# In[101]:


lr=LogisticRegression()


# In[102]:


lr.fit(X_train_std,y_train)
lr_pred=lr.predict(X_test_std)


# In[103]:


print("Predicted Levels: ",list(lr_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[104]:


intercept = lr.intercept_


# In[105]:


coefficients = lr.coef_


# In[106]:


coef_list = list(coefficients[0,:])


# In[107]:


coef_df = pd.DataFrame({'Feature': list(X_train_std.columns),

'Coefficient': coef_list})

print(coef_df)


# In[108]:


predicted_prob = lr.predict_proba(X_test_std)[:,1]


# In[109]:


cm = pd.DataFrame(confusion_matrix(y_test, lr_pred))

cm['Total'] = np.sum(cm, axis=1)

cm = cm.append(np.sum(cm, axis=0), ignore_index=True)

cm.columns = ['low','mod','high', 'very high', 'Total']

cm = cm.set_index([['low','mod','high', 'very high', 'Total']])

print(cm)


# In[110]:


print(classification_report(y_test, lr_pred))


# #### SVC

# In[111]:


from sklearn import svm
svc = svm.SVC(kernel='linear')


# In[112]:


svc.fit(X_train_std,y_train)
svc_pred=svc.predict(X_test_std)


# In[113]:


print("Predicted Levels: ",list(svc_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[115]:


def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()

f_importances(abs(svc.coef_[0]), X_train_std.columns, top=10)
#Reference: https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn


# In[116]:


print(classification_report(y_test, svc_pred))


# #### Knn Classifier

# In[117]:


knn=KNeighborsClassifier()


# In[118]:


knn.fit(X_train_std,y_train)
knn_pred=knn.predict(X_test_std)


# In[119]:


print("Predicted Levels: ",list(knn_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[120]:


print(classification_report(y_test, knn_pred))


# #### DecisionTreeClassifier

# In[121]:


dt=DecisionTreeClassifier()


# In[122]:


dt.fit(X_train_std,y_train)
dt_pred=dt.predict(X_test_std)


# In[123]:


print("Predicted Levels: ",list(dt_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[124]:


dt_feature_imp = pd.Series(dt.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
dt_feature_imp


# In[125]:


# Creating a bar plot
sns.barplot(x=dt_feature_imp, y=dt_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.show()


# In[126]:


print(classification_report(y_test, dt_pred))


# #### RandomForestClassifier

# In[127]:


rf=RandomForestClassifier(n_estimators=2000, max_depth=2)


# In[128]:


rf.fit(X_train_std,y_train)
rf_pred=rf.predict(X_test_std)


# In[129]:


print("Predicted Levels: ",list(rf_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[130]:


rf_feature_imp = pd.Series(rf.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
rf_feature_imp


# In[131]:


#Feature importance for Random Forest Model
# Creating a bar plot
sns.barplot(x=rf_feature_imp, y=rf_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.show()


# In[132]:


print(classification_report(y_test, rf_pred))


# #### Gradient Boosting Classifier

# In[133]:


#Fitting model
gb= GradientBoostingClassifier()# for parametes (n_estimators=2000, max_depth=2)
gb.fit(X_train_std,y_train)


# In[134]:


gb_pred=gb.predict(X_test_std)


# In[135]:


a=list(gb_pred[25:40])
b=list(y_test[25:40])


# In[136]:


print("Predicted: ",list(gb_pred[25:35]))
print("Actual: ",list(y_test[25:35]))


# In[137]:


gb_feature_imp = pd.Series(gb.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
gb_feature_imp


# In[138]:


#Feature importance for Gradient Boosting classfier Model
# Creating a bar plot
sns.barplot(x=gb_feature_imp, y=gb_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features for predicting reading proficiency labels")
#plt.legend()
plt.show()


# In[139]:


print(classification_report(y_test, gb_pred))


# #### ExtraTreesClassifier

# In[140]:


#Fitting model
et= ExtraTreesClassifier(n_estimators=2000, max_depth=2)# for parametes (n_estimators=2000, max_depth=2)
et.fit(X_train_std,y_train)


# In[141]:


#making predictions
et_pred=et.predict(X_test_std)


# In[142]:


# from sklearn.metrics import confusion_matrix
# matrix=confusion_matrix(y_test,et_pred)
# cm=pd.DataFrame(matrix,index=['1','2','3','4'],columns=['1','2','3','4'])
# print(cm)


# In[143]:


print("Predicted Levels: ",list(et_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[144]:


#Extra Trees
et_feature_imp = pd.Series(et.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
et_feature_imp


# In[145]:


sns.barplot(x=et_feature_imp, y=et_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[146]:


print(classification_report(y_test, et_pred))


# #### Key metric summary from the various models and cross validation accuracy

# In[147]:


#Computing accuracy
print("Accuracy of Logistic Regression model: %.2f" % accuracy_score(y_test,lr_pred))
print("Accuracy of SVC model: %.2f" % accuracy_score(y_test,svc_pred))
print("Accuracy of KNN model: %.2f" % accuracy_score(y_test,knn_pred))
print("Accuracy of Decision Trees: %.2f" % accuracy_score(y_test,dt_pred))
print("Accuracy of Random forest classfier model: %.2f" % accuracy_score(y_test,rf_pred))
print("Accuracy of Gradient Boosting classfier model: %.2f" % accuracy_score(y_test,gb_pred))
print("Accuracy of Extra trees classfier model: %.2f" % accuracy_score(y_test,et_pred))


# In[148]:


#Computing precision
print("Logistic Regression model: %.2f" % precision_score(y_test,lr_pred,average='weighted'))
print("SVC model: %.2f" % precision_score(y_test,svc_pred,average='weighted'))
print("KNN model: %.2f" % precision_score(y_test,knn_pred, average='weighted'))
print("Decision Trees: %.2f" % precision_score(y_test,dt_pred, average='weighted'))
print("Random forest classfier model: %.2f" % precision_score(y_test,rf_pred, average='weighted'))
print("Gradient Boosting classfier model: %.2f" % precision_score(y_test,gb_pred, average='weighted'))
print("Extra trees classfier model: %.2f" % precision_score(y_test,et_pred, average='weighted'))


# In[149]:


#Computing recall
print("Logistic Regression model: %.2f" % recall_score(y_test,lr_pred,average='weighted'))
print("SVC model: %.2f" % recall_score(y_test,svc_pred,average='weighted'))
print("KNN model: %.2f" % recall_score(y_test,knn_pred,average='weighted'))
print("Decision Trees: %.2f" % recall_score(y_test,dt_pred,average='weighted'))
print("Random forest classfier model: %.2f" % recall_score(y_test,rf_pred,average='weighted'))
print("Gradient Boosting classfier model: %.2f" % recall_score(y_test,gb_pred,average='weighted'))
print("Extra trees classfier model: %.2f" % recall_score(y_test,et_pred,average='weighted'))


# In[150]:


#Computing f1 score
print("Logistic Regression model: %.2f" % f1_score(y_test,lr_pred,average='weighted'))
print("SVC model: %.2f" % f1_score(y_test,svc_pred,average='weighted'))
print("KNN model: %.2f" % f1_score(y_test,knn_pred,average='weighted'))
print("Decision Trees: %.2f" % f1_score(y_test,dt_pred,average='weighted'))
print("Random forest classfier model: %.2f" % f1_score(y_test,rf_pred,average='weighted'))
print("Gradient Boosting classfier model: %.2f" % f1_score(y_test,gb_pred,average='weighted'))
print("Extra trees classfier model: %.2f" % f1_score(y_test,et_pred,average='weighted'))


# In[156]:


Results = {'Model Name':  ['Logistic Regression', 'SVC','KNN','Decision Trees','Random Forest','Gradient boosting','Extra Trees'],
        'Accuracy': ['0.43','0.41','0.42','0.39','0.41','0.48','0.39'],
        'Precision':  ['0.42','0.40','0.42','0.39','0.39','0.48','0.42'],
        'Recall': ['0.43','0.41','0.42','0.39','0.41','0.48','0.39'],
        'F1score': ['0.41','0.40','0.42','0.39','0.36','0.48','0.32']
        }

Summary = pd.DataFrame (Results, columns = ['Model Name','Accuracy','Precision','Recall','F1score'])
Summary.dtypes

Summary["Accuracy"] = Summary.Accuracy.astype(float)
Summary["Precision"] = Summary.Precision.astype(float)
Summary["Recall"] = Summary.Recall.astype(float)
Summary["F1score"] = Summary.F1score.astype(float)


# In[157]:


ax = Summary.plot.barh(x='Model Name', y='Accuracy', rot=0)
plt.xlabel('Accuracy')
plt.title("Accuracy of various models on predicting reading proficiency labels")
plt.legend()  
plt.show()


# In[153]:


Summary.head(10)


# In[154]:


models=[]
models.append(('Logistic Regression', lr))
models.append(('Knn', knn))
models.append(('SVC', svc))
models.append(('Decision Trees', dt))
models.append(('Random Forest', rf))
models.append(('Gradient Boosting', gb))
models.append(('Extra Trees',et))
models


# In[155]:


from sklearn import model_selection
results = []
names = []
scoring = 'accuracy'
print('A list of each algorithm, the mean accuracy and the standard deviation accuracy.')
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=7)
    cv_results = model_selection.cross_val_score(model, X_train_std,y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

