
# coding: utf-8

# In[175]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[176]:


train = pd.read_csv('2018-05-17 - Recruit Sample Data Train.csv')
train.head()


# In[177]:


train.columns


# # To check if the First Payment Default is Skewed or Not

# In[178]:


train['First Payment Default'].value_counts().plot(kind='bar')


# # First Payment Exploration

# In[179]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(25,7))
df = pd.crosstab(train.State,train['First Payment Default']).apply(lambda r: r/r.sum(), axis=1)
df.plot.barh(stacked=True,width = 0.4,ax=axes[0])
axes[0].legend(loc='upper right')
axes[0].set_xlabel('Proportion')


df = pd.crosstab(train['Rent or Own'],train['First Payment Default']).apply(lambda r: r/r.sum(), axis=1)
df.plot.barh(stacked=True,width = 0.4,ax=axes[1])
axes[1].legend(loc='upper right')
axes[1].set_xlabel('Proportion')

df = pd.crosstab(train['Pay Cycle'],train['First Payment Default']).apply(lambda r: r/r.sum(), axis=1)
df.plot.barh(stacked=True,width = 0.4,ax=axes[2])
axes[2].legend(loc='upper right')
axes[2].set_xlabel('Proportion')


# For State we can see that in Texas there is less people with First Payment Default as False than in California.  
# For Rent or Own Graph there is almost same proportion of true and false, hence it donot matter weather person is living in rent or owning.  
# For Pay Cycle we can see that if person has pay cycle monthly BiWeekly then there is least chance that his First Payment Default is False.

# # Monthly Net Income vs First Payment Default

# ## Data Cleaning

# In[180]:


train['Monthly Net Income'] = train['Monthly Net Income'].str.replace(',', '')
train['Monthly Net Income'] = train['Monthly Net Income'].str.replace('$', '')
train['Monthly Net Income'] = train['Monthly Net Income'].astype(str).apply(lambda x: x.strip()).astype(float)

train['Paycheck Net Income'] = train['Paycheck Net Income'].str.replace(',', '')
train['Paycheck Net Income'] = train['Paycheck Net Income'].str.replace('$', '')
train['Paycheck Net Income'] = train['Paycheck Net Income'].astype(str).apply(lambda x: x.strip()).astype(float)

train['Loan Amount'] = train['Loan Amount'].str.replace(',', '')
train['Loan Amount'] = train['Loan Amount'].str.replace('$', '')
train['Loan Amount'] = train['Loan Amount'].astype(str).apply(lambda x: x.strip()).astype(float)

train.head()


# In[181]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(25,7))
train['Monthly Net Income'].dropna().hist(bins = 100,ax=axes[0])
axes[0].set_title('Monthly Net Income')
train['Paycheck Net Income'].dropna().hist(bins = 100,ax=axes[1])
axes[1].set_title('Paycheck Net Income')
train['Loan Amount'].dropna().hist(bins = 100,ax=axes[2])
axes[2].set_title('Loan Amount')


# We can see that there are lot of outliers in each of the graphs. So, we need to remove the outliers since they represent Outlier data which will cause problem when doing prediction.

# In[182]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[183]:


# Remove Outliers
train = remove_outlier(train,'Monthly Net Income')
train = remove_outlier(train,'Paycheck Net Income')
train = remove_outlier(train,'Loan Amount')


# In[184]:


# Now again seeing plots without outliers
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(25,7))
train['Monthly Net Income'].dropna().hist(bins = 100,ax=axes[0])
axes[0].set_title('Monthly Net Income')
train['Paycheck Net Income'].dropna().hist(bins = 100,ax=axes[1])
axes[1].set_title('Paycheck Net Income')
train['Loan Amount'].dropna().hist(bins = 100,ax=axes[2])
axes[2].set_title('Loan Amount')


# Now this data is very clean, hence we can proceed to work with our predictions.

# In[185]:


train.head()


# We can see there are columns like setID which are of no use to us because they donot help in prediction.  
# We can also see that there is date of funding and loan due date, which are important, but it will take a considerable amount of time to get information from the dates, hence to apply our first model, we ignore those column.

# In[186]:


train = train.drop(['SetID','Time of Application','Loan Funded Date','Loan Due Date'],axis = 1)
train.head()


# Converting State, Pay Cycle and other categorical columns into format so that Sklearn can read it. That is we will be creating columns for each value of State and Pay Cycle and its value will be based on if a row contains that value or not.

# In[187]:


train = pd.concat([train.drop('State', axis=1), pd.get_dummies(train.State)], axis=1)
train = pd.concat([train.drop('Rent or Own', axis=1), pd.get_dummies(train['Rent or Own'])], axis=1)
train = pd.concat([train.drop('Pay Cycle', axis=1), pd.get_dummies(train['Pay Cycle'])], axis=1)
train.head()


# One last thing we need to do before we proceed to prediction. It is that we need to fill null values in dataset if any. SO, let us check null values in dataset.

# In[188]:


for col in train.columns:
    if np.sum(train[col].isna()):
        print(col,' is null')
    else:
        print(col,' is not null')


# Since none of column is null hence we donot need to worry about that issue.

# # Testing Dataset

# In[189]:


test = pd.read_csv('2018-05-17 - Recruit Sample Data Test.csv')
test.head()


# Doing same things with testing datset as we did with training dataset.

# In[190]:


test = test.drop(['SetID','Time of Application','Loan Funded Date','Loan Due Date'],axis = 1)
test.head()


# Checking null values.

# In[191]:


for col in test.columns:
    if np.sum(test[col].isna()):
        print(col,' is null')
    else:
        print(col,' is not null')


# In[192]:


# Droping Last Column Since we need to predict it.
test = test.drop(['First Payment Default'],axis = 1)
test.head()


# In[193]:


test = pd.concat([test.drop('State', axis=1), pd.get_dummies(test.State)], axis=1)
test = pd.concat([test.drop('Rent or Own', axis=1), pd.get_dummies(test['Rent or Own'])], axis=1)
test = pd.concat([test.drop('Pay Cycle', axis=1), pd.get_dummies(test['Pay Cycle'])], axis=1)
test.head()


# # Machine Learning

# ## SVM

# In[203]:


from sklearn.svm import SVC


# In[204]:


train['First Payment Default'] = train['First Payment Default'].replace({True: 1, False: 0})


# In[205]:


cf = SVC()
X_train = train.drop(['First Payment Default'],axis = 1)
Y_train = train['First Payment Default']
X_test = test
cf.fit(X_train,Y_train)


# In[207]:


output = cf.predict(X_test)
output


# In[210]:


final_output = []
for val in output:
    if val == 0:
        final_output.append(False)
    else:
        final_output.append(True)


# In[211]:


test = pd.read_csv('2018-05-17 - Recruit Sample Data Test.csv')
test.head()


# In[212]:


test['First Payment Default'] = final_output
test.head()


# In[214]:


test.to_csv('output_file.csv')

