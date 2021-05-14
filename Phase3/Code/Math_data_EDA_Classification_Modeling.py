#!/usr/bin/env python
# coding: utf-8

# ### Phase 3: Conduct EDA and classification model construction using the master_math.csv file that contains all relevant features and target variable

# #### Importing all necessary libraries

# In[151]:


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


# In[152]:


cd /Users/dansa/Documents/GitHub/Phase1/Data/MASTER


# #### Loading the data and reformatting the school id column

# In[153]:


master_math_new = pandas.read_csv("master_math.csv")
master_math_new['NCESSCH'] = master_math_new['NCESSCH'].apply(lambda x: '{0:0>12}'.format(x))
master_math_new.head()


# In[154]:


master_math_new.shape


# #### Inspecting the data file

# In[155]:


master_math_new.columns


# Create a data frame with only the needed columns for further analysis

# In[156]:


math=pd.DataFrame(master_math_new, columns=[ 'NCESSCH', 'NAME', 'SCH_TYPE_x', 
       'TITLEI_STATUS','TEACHERS', 'FARMS_COUNT', 'Total_enroll_students',
       'SCH_FTETEACH_TOT', 'SCH_FTETEACH_CERT', 'SCH_FTETEACH_NOTCERT',
       'FTE_teachers_count', 'SalaryforTeachers', 'Total_SAT_ACT_students',
       'SCH_IBENR_IND_new', 'Total_IB_students', 'SCH_APENR_IND_new',
       'SCH_APCOURSES', 'SCH_APMATHENR_IND_new','Total_AP_math_students',
       'Total_students_tookAP', 'SCH_MATHCLASSES_ALG', 'SCH_MATHCERT_ALG',
       'Total_Alg1_enroll_students', 'Total_Alg1_pass_students',
       'Income_Poverty_ratio','ALL_MTH00PCTPROF_1718_new'])


# In[157]:


math.head()


# ##### Rename columns

# In[158]:


math.rename(columns={'NCESSCH':'School_ID', 'SCH_TYPE_x':'School_type','FARMS_COUNT':'No.FARMS_students',
                       'SCH_FTETEACH_TOT':'FTE_teachcount','SCH_FTETEACH_CERT':'Certified_FTE_teachers','SCH_FTETEACH_NOTCERT':
                       'Noncertified_FTE_teachers','Total_SAT_ACT_students':'Students_participate_SAT_ACT','SCH_IBENR_IND_new':'IB_Indicator',
                       'SCH_APENR_IND_new':'AP_Indicator','SCH_APCOURSES':'No.ofAP_courses_offer','SCH_APMATHENR_IND_new':'Students_enroll_inAPMath?',
                       'SCH_MATHCLASSES_ALG':'No.ofAlg1classes','SCH_MATHCERT_ALG':'Alg1_taught_by_certmathteahcers',
                       'ALL_MTH00PCTPROF_1718_new':'Percent_Math_Proficient'}, inplace=True)


# In[159]:


math.describe().T


# ##### IB has some missing values, lets clean than up

# In[160]:


counts = math['IB_Indicator'].value_counts().to_dict()
print (counts)


# In[161]:


math=math[math['IB_Indicator']!=-6]


# In[162]:


counts = math['IB_Indicator'].value_counts().to_dict()
print (counts)


# ##### Let take a closer look at the dataframe and datatypes

# In[163]:


print(math.info())


# We have 13,799 entries and no null values in any column. There are 26 columns, but we can drop the school_id and name and we'll want to split off the Percent_Math_Proficient.
# The object type features should be strings.
# 
# Let's take a quick look at some of the data.

# In[164]:


math.hist(bins=50, figsize=(20,15))
plt.show()


# We can see that some features have most of their instances at or near zero and relatively few instances at higher values, in some cases much higher. Other features cluster close to zero and have long tails. We also see the percent_math_proficient is almost normally distributed.

# In[165]:


math_EDA=math[['NAME','Income_Poverty_ratio','Percent_Math_Proficient']]


# In[166]:


highest_proficiency=math_EDA.sort_values(by=['Percent_Math_Proficient'], inplace=True, ascending=False)


# In[167]:


math_EDA.head()


# In[168]:


math_high = math_EDA.head(5)
math_high.shape


# In[169]:


plt.style.use('ggplot')

plt.barh(math_high.NAME, math_high.Percent_Math_Proficient, color='green')
plt.ylabel("School Names")
plt.xlabel("Percent Math Proficient")
plt.title("Top 5 high schools with highest percent math proficiency")
plt.xlim(0.0, 100.0)

plt.show()


# In[170]:


lowest_proficiency=math_EDA.sort_values(by=['Percent_Math_Proficient'], inplace=True, ascending=True)


# In[171]:


math_EDA.head()


# In[172]:


math_low = math_EDA.head(5)
math_low.shape


# In[173]:


plt.style.use('ggplot')

plt.barh(math_low.NAME, math_low.Percent_Math_Proficient, color='green')
plt.ylabel("School Names")
plt.xlabel("Percent Math Proficient")
plt.title("Top 5 high schools with highest percent math proficiency")
plt.xlim(0.0, 100.0)

plt.show()


# In[174]:


sns.set_style('darkgrid')
_plt = sns.countplot(x='TITLEI_STATUS', data=math)
_plt.set_title('School Title I status')
_plt.set_xticklabels(['Title I  schoolwide school','Not a Title I school','Title I schoolwide eligible- Title I targeted assistance program','Title I schoolwide eligible school-No program','Title I targeted assistance eligible school– No program','Title I targeted assistance school'])
_plt.set_xticklabels(_plt.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('/Users/dansa/Documents/Title1_M_dist.png', dpi=300, bbox_inches='tight')
plt.show()


# Most of the high schools appear to be Title 1 schoolwide or not Title 1 funded schools

# ##### Let's look at the distribution of the proficiency percentages

# In[175]:


# ax = sns.distplot(math['Percent_Math_Proficient'], bins=20, kde=False, fit=stats.norm);

# # Get the fitted parameters used by sns
# (mu, sigma) = stats.norm.fit(math['Percent_Math_Proficient'])
# #print 'mu={0}, sigma={1}'.format(mu, sigma)

# # Legend and labels 
# plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)])
# plt.ylabel('Frequency')
# # Cross-check this is indeed the case - should be overlaid over black curve
# x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
# ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))
# plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma),
#            "cross-check"])


# In[176]:


from scipy.stats import norm
# Plot Histogram
sns.distplot(math['Percent_Math_Proficient'] , bins=20, kde=False, fit=stats.norm);

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


# #### Lets find the percent of certified and non-certified teachers

# In[177]:


math['Pct_certified_teachers']=(math['Certified_FTE_teachers']/math['FTE_teachcount']*100) 


# In[178]:


math['Pct_noncertified_teachers']=(math['Noncertified_FTE_teachers']/math['FTE_teachcount']*100) 


# #### Lets find the salary per FTE in each school

# In[179]:


math['Salary_perFTE_teacher'] = math['SalaryforTeachers']/math['FTE_teachers_count'] 


# In[180]:


math['IPR_estimate'] = math['Income_Poverty_ratio'] #Income poverty ratio is reported as a percent 


# ##### Lets drop the unwanted columns

# In[181]:


math_clean=math.drop(['School_ID','NAME','Certified_FTE_teachers', 'Noncertified_FTE_teachers','FTE_teachcount','FTE_teachers_count','SalaryforTeachers','Income_Poverty_ratio' ], axis=1)


# In[182]:


math_clean.info()


# ##### Change school type from int to float

# In[183]:


math_clean['School_type'] = math_clean['School_type'].astype(float)


# In[184]:


math_clean.describe()


# ##### Check for missing or null values

# In[185]:


sns.heatmap(math_clean.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[186]:


math_clean.shape


# #### Let's create Labels for Math Proficiency based on the percent distribution of the schools

# In[187]:


math_clean[['Percent_Math_Proficient']].describe()


# In[188]:


math_clean['Percent_Math_Proficient'].plot(kind='hist')


# In[189]:


boxplot = math_clean.boxplot(column=['Percent_Math_Proficient'])
boxplot.plot()

plt.show()


# In[190]:


mu = 200
sigma = 25
n_bins = 5


fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(math_clean.Percent_Math_Proficient, n_bins, density=True, histtype='step',
                           cumulative=True, label='Empirical')

# Add a line showing the expected distribution.
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]

ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

# Overlay a reversed cumulative histogram.
ax.hist(math_clean.Percent_Math_Proficient, bins=bins, density=True, histtype='step', cumulative=-1,
        label='Reversed emp.')

# tidy up the figure
ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative step histograms')
ax.set_xlabel('Percent Math Proficiency')
ax.set_ylabel('Likelihood of occurrence')

plt.show()


# In[191]:


# getting data of the histogram
count, bins_count = np.histogram(math_clean.Percent_Math_Proficient, bins=10)
  
# finding the Probability Distribution Function of the histogram using count values
pdf = count / sum(count)
  
# using numpy np.cumsum to calculate the Cumulative Distribution Function
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
  
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()


# In[192]:


fig, ax = plt.subplots()
math_clean['Percent_Math_Proficient'].hist(bins=30, color='#A9C5D3', 
                             edgecolor='black', grid=False)
ax.set_title('Percent of Schools with at or above Math Proficiency Histogram', fontsize=12)
ax.set_xlabel('Percent of Math Proficiency', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
#Reference: https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b


# In[193]:


quantile_list = [0, .25, .5, .75, 1.]
quantiles = math_clean['Percent_Math_Proficient'].quantile(quantile_list)
quantiles


# In[194]:


fig, ax = plt.subplots()
math_clean['Percent_Math_Proficient'].hist(bins=30, color='#A9C5D3', 
                             edgecolor='black', grid=False)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)
ax.set_title('Percentages of Math Proficiency across all High Schools', fontsize=12)
ax.set_xlabel('Percent of Math Proficiency', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)


# In[195]:


pd.qcut(math_clean['Percent_Math_Proficient'], q=4, precision = 0).value_counts(ascending=True)


# In[196]:


quantile_labels = ['Low', 'Moderate','High','Very High']
quantile_numeric = [1,2,3,4]
math_clean['Percent_Math_Proficient_quantile_range'] = pd.qcut(
                                            math_clean['Percent_Math_Proficient'], 
                                            q=quantile_list)
math_clean['Percent_Math_Proficient_quantile_label'] = pd.qcut(
                                            math_clean['Percent_Math_Proficient'], 
                                            q=quantile_list,       
                                            labels=quantile_labels)
math_clean['Percent_Math_Proficient_quantile_encoded'] = pd.qcut(
                                            math_clean['Percent_Math_Proficient'], 
                                            q=quantile_list,
                                            labels=quantile_numeric,
                                            precision=0)

math_clean.head()


# In[197]:


math_clean['Percent_Math_Proficient_quantile_label'].value_counts(ascending=True)


# In[198]:


math_clean['Percent_Math_Proficient_quantile_encoded'] = math_clean['Percent_Math_Proficient_quantile_encoded'].astype(float)


# ### Looking for Correlations and Visualizing
# We should calculate data correlations and plot a scatter matrix.
# 
# For training the ML models, we'll want to separate the Percent_Math_Proficient from the rest of the data. But for investigating correlations, we'll want to include the target.

# In[199]:


math_clean1=math_clean[['School_type', 'TITLEI_STATUS', 'TEACHERS', 'No.FARMS_students',
       'Total_enroll_students', 'Students_participate_SAT_ACT', 'IB_Indicator',
       'Total_IB_students', 'AP_Indicator', 'No.ofAP_courses_offer',
       'Students_enroll_inAPMath?', 'Total_AP_math_students','Total_students_tookAP', 
       'No.ofAlg1classes','Alg1_taught_by_certmathteahcers', 'Total_Alg1_enroll_students','Total_Alg1_pass_students',
       'Pct_certified_teachers', 'Pct_noncertified_teachers',
       'Salary_perFTE_teacher', 'IPR_estimate','Percent_Math_Proficient_quantile_encoded']]


# In[200]:


math_clean1.dtypes


# In[201]:


correlation_matrix = math_clean1.corr()


# In[202]:


correlation_matrix['Percent_Math_Proficient_quantile_encoded'].sort_values(ascending=False)


# It seems like a few features (IPR_estimate, Total_students_tookAP, Total_AP_math_students have a weak to moderate positive correlation to the target (Percent_Math_Proficient), and a couple are somewhat negatively correlated (School_type).
# 
# * IPR_estimate is the Neighborhood Income Poverty Ratio.
# * Total_students_tookAP is the count of students who the AP exam.
# * Total_AP_math_students is the number of students who took an AP math course.
# * School_type is refers to whether the school is a "1-Regular School, 2-Special Education School, 3-Career and Technical School and 4-Alternative Education School"
# 
# We can look at a heatmap of the correlations of all numeric features to visualize which features are correlated.

# In[203]:


# correlation matrix heatmap
plt.figure(figsize=(28,15))
corr_heatmap = sns.heatmap(correlation_matrix, annot=True, linewidths=0.2, center=0, cmap="RdYlGn")
corr_heatmap.set_title('Correlation Heatmap')
plt.savefig('/Users/dansa/Documents/corr_heatmap.png', dpi=300, bbox_inches='tight')


# In[204]:


#test
corr_pairs = {}
feats = correlation_matrix.columns
for x in feats:
    for y in feats:
        if x != y and np.abs(correlation_matrix[x][y]) >= 0.7:  # which pairs are strongely correlated?
            if (y, x) not in corr_pairs.keys():
                corr_pairs[(x, y)] = correlation_matrix[x][y]


# In[205]:


corr_pairs


# In[206]:


attrs = ['IPR_estimate','Total_AP_math_students','Total_students_tookAP','Percent_Math_Proficient_quantile_encoded']


# In[207]:


sns.set(style='ticks', color_codes=True)
_ = sns.pairplot(data=correlation_matrix[attrs], height=3, aspect=1, kind='scatter', plot_kws={'alpha':0.9})


# In[208]:


sns.jointplot(x="IPR_estimate", y="Percent_Math_Proficient_quantile_encoded", data=math_clean1)


# In[209]:


sns.pairplot(math_clean1, hue = 'Percent_Math_Proficient_quantile_encoded',vars = ['IPR_estimate','Total_students_tookAP','Total_AP_math_students','No.ofAP_courses_offer'] )


# ### ML prep
# #### Separate labels
# Let's separate out the target from the predicting features.

# In[210]:


X = math_clean1.drop('Percent_Math_Proficient_quantile_encoded', axis=1)
y = math_clean1.Percent_Math_Proficient_quantile_encoded


# In[211]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[212]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Transform Categorical Features
# Since these categorical features don't appear to have an inherent ordering, let's try encoding them as one-hot vectors for better ML performance.

# In[213]:


train_data_onehot = pd.get_dummies(X_train, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])
train_data_onehot.head()

test_data_onehot = pd.get_dummies(X_test, columns=['TITLEI_STATUS'], prefix=['TITLEI_STATUS'])


# #### Scale Features
# We can check out the statistics for each feature, do they need to be normalized?

# In[214]:


train_data_onehot.describe()


# In[215]:


sc= StandardScaler()

X_train_scaled = sc.fit_transform(train_data_onehot)

X_test_scaled = sc.transform(test_data_onehot)


# In[216]:


print(X_train_scaled)
print(X_train_scaled.mean(axis=0))


# In[217]:


print(X_train_scaled.std(axis=0))


# In[218]:


X_train_std = pd.DataFrame(X_train_scaled, columns=train_data_onehot.columns)
X_test_std = pd.DataFrame(X_test_scaled, columns=test_data_onehot.columns)


# That should work better, the standard deviation for each feature is 1 and the mean is ~0.

# ### Classification models

# #### Logistic Regression

# In[219]:


lr=LogisticRegression()


# In[220]:


lr.fit(X_train_std,y_train)
lr_pred=lr.predict(X_test_std)


# In[221]:


print("Predicted Levels: ",list(lr_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[222]:


intercept = lr.intercept_


# In[223]:


coefficients = lr.coef_


# In[224]:


coef_list = list(coefficients[0,:])


# In[225]:


coef_df = pd.DataFrame({'Feature': list(X_train_std.columns),

'Coefficient': coef_list})

print(coef_df)


# In[226]:


predicted_prob = lr.predict_proba(X_test_std)[:,1]


# In[227]:


cm = pd.DataFrame(confusion_matrix(y_test, lr_pred))

cm['Total'] = np.sum(cm, axis=1)

cm = cm.append(np.sum(cm, axis=0), ignore_index=True)

cm.columns = ['1','2','3','4', 'Total']

cm = cm.set_index([['1','2','3','4', 'Total']])

print(cm)


# In[228]:


print(classification_report(y_test, lr_pred))


# #### SVC

# In[229]:


from sklearn import svm
svc = svm.SVC(kernel='linear')


# In[230]:


svc.fit(X_train_std,y_train)
svc_pred=svc.predict(X_test_std)


# In[231]:


print("Predicted Levels: ",list(svc_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[232]:


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


# In[233]:


print(classification_report(y_test, svc_pred))


# #### Knn Classifier

# In[234]:


knn=KNeighborsClassifier()


# In[235]:


knn.fit(X_train_std,y_train)
knn_pred=knn.predict(X_test_std)


# In[236]:


print("Predicted Levels: ",list(knn_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[237]:


print(classification_report(y_test, knn_pred))


# #### Decision Trees

# In[238]:


dt=DecisionTreeClassifier()


# In[239]:


dt.fit(X_train_std,y_train)
dt_pred=dt.predict(X_test_std)


# In[240]:


print("Predicted Levels: ",list(dt_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[241]:


dt_feature_imp = pd.Series(dt.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
dt_feature_imp


# In[242]:


sns.barplot(x=dt_feature_imp, y=dt_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.show()


# In[243]:


print(classification_report(y_test, dt_pred))


# #### Gaussian RandomForestClassifier

# In[244]:


rf=RandomForestClassifier(n_estimators=2000, max_depth=2)


# In[245]:


rf.fit(X_train_std,y_train)
rf_pred=rf.predict(X_test_std)


# In[246]:


print("Predicted Levels: ",list(rf_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[247]:


rf_feature_imp = pd.Series(rf.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
rf_feature_imp


# In[248]:


#Feature importance for Random Forest Model
# Creating a bar plot
sns.barplot(x=rf_feature_imp, y=rf_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.show()


# In[249]:


print(classification_report(y_test, rf_pred, labels=[1, 2, 3, 4]))


# #### GradientBoostingClassifier

# In[250]:


#Fitting model
gb= GradientBoostingClassifier()# for parametes (n_estimators=2000, max_depth=2)
gb.fit(X_train_std,y_train)


# In[251]:


gb_pred=gb.predict(X_test_std)


# In[252]:


print("Predicted: ",list(gb_pred[:10]))
print("Actual: ",list(y_test[:10]))


# In[253]:


gb_feature_imp = pd.Series(gb.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
gb_feature_imp


# In[254]:


#Feature importance for Gradient Boosting classfier Model
# Creating a bar plot
sns.barplot(x=gb_feature_imp, y=gb_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features for predicting math proficiency labels")
#plt.legend()
plt.show()


# In[255]:


print(classification_report(y_test, gb_pred, labels=[1, 2, 3, 4]))


# #### ExtraTreesClassifier

# In[256]:


#Fitting model
et= ExtraTreesClassifier(n_estimators=2000, max_depth=2)# for parametes (n_estimators=2000, max_depth=2)
et.fit(X_train_std,y_train)


# In[257]:


#making predictions
et_pred=et.predict(X_test_std)


# In[258]:


print("Predicted Levels: ",list(et_pred[:10]))
print("Actual Levels: ",list(y_test[:10]))


# In[259]:


et_feature_imp = pd.Series(et.feature_importances_,index=X_train_std.columns).sort_values(ascending=False)
et_feature_imp


# In[260]:


sns.barplot(x=et_feature_imp, y=et_feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.show()


# In[261]:


print(classification_report(y_test, et_pred, labels=[1, 2, 3, 4]))


# #### Key metric summary from the various models and cross validation accuracy

# In[262]:


#Computing accuracy
print("Accuracy of Logistic Regression model: %.2f" % accuracy_score(y_test,lr_pred))
print("Accuracy of SVC model: %.2f" % accuracy_score(y_test,svc_pred))
print("Accuracy of KNN model: %.2f" % accuracy_score(y_test,knn_pred))
print("Accuracy of Decision Trees model: %.2f" % accuracy_score(y_test,dt_pred))
print("Accuracy of Random forest classfier model: %.2f" % accuracy_score(y_test,rf_pred))
print("Accuracy of Gradient Boosting classfier model: %.2f" % accuracy_score(y_test,gb_pred))
print("Accuracy of Extra trees classfier model: %.2f" % accuracy_score(y_test,et_pred))


# In[263]:


#Computing precision
print("Logistic Regression model: %.2f" % precision_score(y_test,lr_pred,average='weighted'))
print("SVC model: %.2f" % precision_score(y_test,svc_pred,average='weighted'))
print("KNN model: %.2f" % precision_score(y_test,knn_pred, average='weighted'))
print("Decision Trees: %.2f" % precision_score(y_test,dt_pred, average='weighted'))
print("Random forest classfier model: %.2f" % precision_score(y_test,rf_pred, average='weighted'))
print("Gradient Boosting classfier model: %.2f" % precision_score(y_test,gb_pred, average='weighted'))
print("Extra trees classfier model: %.2f" % precision_score(y_test,et_pred, average='weighted'))


# In[264]:


#Computing recall
print("Logistic Regression model: %.2f" % recall_score(y_test,lr_pred,average='weighted'))
print("SVC model: %.2f" % recall_score(y_test,svc_pred,average='weighted'))
print("KNN model: %.2f" % recall_score(y_test,knn_pred,average='weighted'))
print("Decision Trees: %.2f" % recall_score(y_test,dt_pred,average='weighted'))
print("Random forest classfier model: %.2f" % recall_score(y_test,rf_pred,average='weighted'))
print("Gradient Boosting classfier model: %.2f" % recall_score(y_test,gb_pred,average='weighted'))
print("Extra trees classfier model: %.2f" % recall_score(y_test,et_pred,average='weighted'))


# In[265]:


#Computing f1 score
print("Logistic Regression model: %.2f" % f1_score(y_test,lr_pred,average='weighted'))
print("SVC model: %.2f" % f1_score(y_test,svc_pred,average='weighted'))
print("KNN model: %.2f" % f1_score(y_test,knn_pred,average='weighted'))
print("Decision Trees: %.2f" % f1_score(y_test,dt_pred,average='weighted'))
print("Random forest classfier model: %.2f" % f1_score(y_test,rf_pred,average='weighted'))
print("Gradient Boosting classfier model: %.2f" % f1_score(y_test,gb_pred,average='weighted'))
print("Extra trees classfier model: %.2f" % f1_score(y_test,et_pred,average='weighted'))


# In[271]:


Results = {'Model Name':  ['Logistic Regression', 'SVC','KNN','Decision Trees','Random Forest','Gradient boosting','Extra Trees'],
        'Accuracy': ['0.44','0.42','0.42','0.38','0.42','0.49','0.38'],
        'Precision':  ['0.42','0.43','0.42','0.39','0.45','0.48','0.30'],
        'Recall': ['0.44','0.42','0.42','0.38','0.42','0.49','0.38'],
        'F1score': ['0.42','0.38','0.42','0.38','0.38','0.47','0.33']
        }

Summary = pd.DataFrame (Results, columns = ['Model Name','Accuracy','Precision','Recall','F1score'])
Summary.dtypes

Summary["Accuracy"] = Summary.Accuracy.astype(float)
Summary["Precision"] = Summary.Precision.astype(float)
Summary["Recall"] = Summary.Recall.astype(float)
Summary["F1score"] = Summary.F1score.astype(float)


# In[272]:


ax = Summary.plot.barh(x='Model Name', y='Accuracy', rot=0)
plt.xlabel('Accuracy')
plt.ylabel('Model Name')
plt.title("Accuracy of various models on predicting math proficiency labels")
plt.legend()  
plt.show()


# In[273]:


Summary.head(10)


# In[269]:


models=[]
models.append(('Logistic Regression', lr))
models.append(('Knn', knn))
models.append(('SVC', svc))
models.append(('Decision Trees', dt))
models.append(('Random Forest', rf))
models.append(('Gradient Boosting', gb))
models.append(('Extra Trees',et))
models


# In[270]:


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

