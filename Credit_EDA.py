#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# # Applications Dataset - Data Cleanup

# In[4]:


#Reading the csv file and creating a dataframe
application_data = pd.read_csv('application_data.csv')


# In[196]:


#No. of rows & columns in the dataframe - 
application_data.shape


# In[6]:


#Datatypes of the columns - 
application_data.info()


# In[7]:


#Understanding the target variable:
application_data.TARGET.value_counts()


# - 1 - Clients with payment difficulties 
# - 0 - All other cases

# In[8]:


application_data.iloc[:,41:].isnull().sum()*100/len(application_data)


# In[9]:


# Column OWN_CAR_AGE have more than 50% null values
# Will be dropping the column from the analysis


# In[10]:


application_data.iloc[:,41:81].isnull().sum()*100/len(application_data)


# ### Planning to drop all columns that have missing data more than 50%

# In[12]:


application_data.iloc[:,81:].isnull().sum()*100/len(application_data)


# In[14]:


col1 = application_data.iloc[:,81:88].columns
col1


# In[15]:


col2 = application_data.iloc[:,44:81].columns
col2


# In[16]:


col3 = ['OWN_CAR_AGE','EXT_SOURCE_1']


# In[17]:


cols = col1.append(col2)


# In[198]:


# Creating a new dataframe in order to preserve earlier data
# The new dataframe will be used for our further analysis

application_data_2 = application_data.drop(cols, axis = 1)
application_data_2.head()


# In[199]:


application_data_2.drop(col3, axis = 1, inplace = True)


# In[200]:


# Checking for missing values
application_data_2.iloc[:,:35].isnull().sum()*100/len(application_data_2)


# In[21]:


# Checking for missing values
application_data_2.iloc[:,35:].isnull().sum()*100/len(application_data_2)


# In[201]:


# Dropping WALLSMATERIAL_MODE that was missed during earlier operation
application_data_2.drop('WALLSMATERIAL_MODE', axis = 1, inplace = True)


# In[202]:


# From 122, we are now down to 75 columns.
application_data_2.shape


# ### After dropping columns, let's now understand if we can impute values in the ones that are missing

# In[203]:


application_data_2.iloc[:,:35].isnull().sum()


# In[26]:


# Treating null values of AMT_ANNUITY:
application_data_2[application_data_2.AMT_ANNUITY.isna() == True]


# In[204]:


# AMT_ANNUITY has 12 null values.
# The term for the loan is not mentioned, so the annuity calculation is a bit difficult.
# Can take a random number between 12 and 36 to divide the AMT_CREDIT and calculate AMT_ANNUITY

import random
rd = random.randint(12,37)
rd

# Using a lambda function
application_data_2.AMT_ANNUITY = application_data_2.apply(lambda x: round(application_data_2['AMT_CREDIT']/rd,1))


# In[205]:


application_data_2.AMT_ANNUITY.isnull().sum()


# In[206]:


# Treating missing values for AMT_GOODS_PRICE:
application_data_2[application_data_2.AMT_GOODS_PRICE.isna() == True]


# In[207]:


# The goods amount is the price of the goods for which the loan is given. Taking the mean might not be prudent
# as there might be outliers.
# Taking the median and imputing the same in the null values - 

application_data_2.AMT_GOODS_PRICE.median()


# In[208]:


application_data_2.AMT_GOODS_PRICE.fillna(application_data_2.AMT_GOODS_PRICE.median(), inplace = True)


# In[209]:


application_data_2.AMT_GOODS_PRICE.isnull().sum()


# In[210]:


# NAME_TYPE_SUITE is a categorical variable. The missing values can be imputed with the most occurring value
# which is mode
application_data_2.NAME_TYPE_SUITE.mode()[0]


# In[211]:


application_data_2.NAME_TYPE_SUITE.fillna(application_data_2.NAME_TYPE_SUITE.mode()[0],inplace = True)


# In[212]:


application_data_2.NAME_TYPE_SUITE.isnull().sum()


# In[213]:


# CNT_FAM_MEMBERS - have 2 missing values.
# Now, this is not a categorical variable, but it would still make sense to
# impute the values using mode()

application_data_2.CNT_FAM_MEMBERS.mode()[0]


# In[214]:


application_data_2.CNT_FAM_MEMBERS.fillna(application_data_2.CNT_FAM_MEMBERS.mode()[0], inplace = True)


# In[215]:


application_data_2.CNT_FAM_MEMBERS.isnull().sum()


# In[216]:


application_data_2.isnull().sum()


# In[217]:


# Understanding the correlation between Organization Type, Income, Credit Amount and Occupation Type
application_data_2[['ORGANIZATION_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT','OCCUPATION_TYPE']]


# In[218]:


application_data_2.ORGANIZATION_TYPE.replace('XNA',np.NaN, inplace = True)


# In[219]:


# Ideal way to impute these null values would be to randomly distribute the first 3 values (most occuring)
# across the null values, or to select values based on OCCUPATION_TYPE
# However, imputing mode() for convenience 

application_data_2.ORGANIZATION_TYPE.fillna(application_data_2.ORGANIZATION_TYPE.mode()[0], inplace = True)


# In[220]:


application_data_2.ORGANIZATION_TYPE.isnull().sum()


# In[221]:


# Ideal way to impute these null values would be to randomly distribute the first 3 values (most occuring)
# across the null values, or to select values based on ORGANIZATION_TYPE
# However, imputing mode() for convenience 

application_data_2.OCCUPATION_TYPE.fillna(application_data_2.OCCUPATION_TYPE.mode()[0], inplace = True)


# In[222]:


application_data_2.OCCUPATION_TYPE.isnull().sum()


# In[223]:


# EXT_SOURCE_2 has normalized values
# Imputing median over the null values

application_data_2.EXT_SOURCE_2.fillna(application_data_2.EXT_SOURCE_2.median(), inplace = True)


# In[224]:


application_data_2.EXT_SOURCE_2.isnull().sum()


# In[225]:


# EXT_SOURCE_3 has normalized values
# Imputing median over the null values

application_data_2.EXT_SOURCE_3.fillna(application_data_2.EXT_SOURCE_3.median(), inplace = True)


# In[226]:


application_data_2.EXT_SOURCE_2.isnull().sum()


# In[227]:


# TOTALAREA_MODE has normalized values
# Imputing median over null values

application_data_2.TOTALAREA_MODE.fillna(application_data_2.TOTALAREA_MODE.median(), inplace = True)


# In[228]:


application_data_2.TOTALAREA_MODE.isnull().sum()


# In[229]:


# EMERGENCYSTATE_MODE has categorical values - Yes and No

application_data_2.EMERGENCYSTATE_MODE.value_counts()


# In[230]:


# Imputing mode over null values

application_data_2.EMERGENCYSTATE_MODE.fillna(application_data_2.EMERGENCYSTATE_MODE.mode()[0], inplace = True)


# In[231]:


application_data_2.EMERGENCYSTATE_MODE.isnull().sum()


# In[232]:


# Imputing the null values with median to get an estimate of observation of client's social surroundings 
# with observable 30 DPD (days past due) default

application_data_2.OBS_30_CNT_SOCIAL_CIRCLE.fillna(application_data_2.OBS_30_CNT_SOCIAL_CIRCLE.median(), inplace = True)


# In[233]:


application_data_2.OBS_30_CNT_SOCIAL_CIRCLE.isnull().sum()


# In[234]:


# Imputing the null values with median to get an estimate of observation of client's social surroundings 
# defaulted on 30 DPD (days past due)

application_data_2.DEF_30_CNT_SOCIAL_CIRCLE.fillna(application_data_2.DEF_30_CNT_SOCIAL_CIRCLE.median(), inplace = True)


# In[235]:


application_data_2.DEF_30_CNT_SOCIAL_CIRCLE.isnull().sum()


# In[236]:


# Imputing the null values with median to get an estimate of observation of client's social surroundings
# with observable 60 DPD (days past due) default

application_data_2.OBS_60_CNT_SOCIAL_CIRCLE.fillna(application_data_2.OBS_60_CNT_SOCIAL_CIRCLE.median(), inplace = True)


# In[237]:


application_data_2.OBS_60_CNT_SOCIAL_CIRCLE.isnull().sum()


# In[238]:


# Imputing the null values with median to get an estimate of observation of client's social surroundings
# defaulted on 60 (days past due) DPD

application_data_2.DEF_60_CNT_SOCIAL_CIRCLE.fillna(application_data_2.DEF_60_CNT_SOCIAL_CIRCLE.median(), inplace = True)


# In[239]:


application_data_2.DEF_60_CNT_SOCIAL_CIRCLE.isnull().sum()


# In[240]:


application_data_2.DAYS_LAST_PHONE_CHANGE.fillna(0.0, inplace = True)


# In[241]:


# Imputing the missing values with median of AMT_REQ_CREDIT_BUREAU_HOUR

application_data_2.AMT_REQ_CREDIT_BUREAU_HOUR.fillna(application_data_2.AMT_REQ_CREDIT_BUREAU_HOUR.median(), inplace = True)


# In[242]:


application_data_2.AMT_REQ_CREDIT_BUREAU_HOUR.isnull().sum()


# In[243]:


# Imputing the missing values with median of AMT_REQ_CREDIT_BUREAU_DAY

application_data_2.AMT_REQ_CREDIT_BUREAU_DAY.fillna(application_data_2.AMT_REQ_CREDIT_BUREAU_DAY.median(), inplace = True)


# In[244]:


application_data_2.AMT_REQ_CREDIT_BUREAU_DAY.isnull().sum()


# In[245]:


# Imputing the missing values with median of AMT_REQ_CREDIT_BUREAU_WEEK

application_data_2.AMT_REQ_CREDIT_BUREAU_WEEK.fillna(application_data_2.AMT_REQ_CREDIT_BUREAU_WEEK.median(), inplace = True)


# In[246]:


application_data_2.AMT_REQ_CREDIT_BUREAU_WEEK.isnull().sum()


# In[247]:


# Imputing the missing values with median of AMT_REQ_CREDIT_BUREAU_MON

application_data_2.AMT_REQ_CREDIT_BUREAU_MON.fillna(application_data_2.AMT_REQ_CREDIT_BUREAU_MON.median(), inplace = True)


# In[248]:


application_data_2.AMT_REQ_CREDIT_BUREAU_MON.isnull().sum()


# In[249]:


# Imputing the missing values with median of AMT_REQ_CREDIT_BUREAU_QRT

application_data_2.AMT_REQ_CREDIT_BUREAU_QRT.fillna(application_data_2.AMT_REQ_CREDIT_BUREAU_QRT.median(), inplace = True)


# In[250]:


application_data_2.AMT_REQ_CREDIT_BUREAU_QRT.isnull().sum()


# In[251]:


# Imputing the missing values with median of AMT_REQ_CREDIT_BUREAU_YEAR

application_data_2.AMT_REQ_CREDIT_BUREAU_YEAR.fillna(application_data_2.AMT_REQ_CREDIT_BUREAU_YEAR.median(), inplace = True)


# In[252]:


application_data_2.AMT_REQ_CREDIT_BUREAU_YEAR.isnull().sum()


# In[253]:


# All null values have been handled

application_data_2.isnull().sum()


# In[254]:


application_data_2.shape


# ### Exporting the data to a CSV file

# In[255]:


application_data_2.to_csv('/Users/manalibhagwat-reddy/Desktop/Python py Files/application_data_2.csv')


# # Previous Applications Dataset - Data Cleanup

# In[79]:


# Reading the data in a dataframe
previous_applications = pd.read_csv('previous_application.csv')


# In[256]:


#No. of rows & columns in the dataframe - 
previous_applications.shape


# In[81]:


#Datatypes of the columns -
previous_applications.info()


# In[82]:


#Understanding the target variable before starting the analysis:
previous_applications.NAME_CONTRACT_STATUS.value_counts()


# #### When a client applies for a loan, there are four types of decisions that could be taken by the client/company:
# 
# `Approved`: The Company has approved loan Application
# 
# `Cancelled`: The client cancelled the application sometime during approval. Either the client changed her/his mind about the loan or in some cases due to a higher risk of the client he received worse pricing which he did not want.
# 
# `Refused`: The company had rejected the loan (because the client does not meet their requirements etc.).
# 
# `Unused offer`:  Loan has been cancelled by the client but on different stages of the process.
# 
# #### In this case study, we will use EDA to understand how consumer attributes and loan attributes influence the tendency of default.

# In[83]:


previous_applications.isnull().sum()*100/len(previous_applications)


# ### For this dataset, columns more than 40% of missing data will be dropped (except NFLAG_INSURED_ON_APPROVAL)

# In[257]:


columns = ['AMT_DOWN_PAYMENT','RATE_DOWN_PAYMENT','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED','NAME_TYPE_SUITE',
          'DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']


# In[258]:


previous_applications_2 = previous_applications.drop(columns, axis = 1)


# In[259]:


# From 37, we are down to 27 columns

previous_applications_2.shape


# In[260]:


previous_applications_2.isnull().sum()


# In[261]:


# AMT_CREDIT has 1 null value and that can be replaced directly by 0.0
previous_applications_2.AMT_CREDIT.fillna(0.0, inplace = True)


# In[262]:


# AMT_CREDIT is handled
previous_applications_2.AMT_CREDIT.isnull().sum()


# In[263]:


# AMT_GOODS_PRICE has 23% missing values - 

previous_applications_2[previous_applications_2.AMT_GOODS_PRICE.isna() == False]

# Here we can see that the AMT_GOODS_PRICE takes the same values as AMT_APPLICATION
# Therefore it is safe to impute the same values in AMT_GOODS_PRICE wherever they are null


# In[264]:


previous_applications_2.AMT_GOODS_PRICE.fillna(previous_applications_2.AMT_APPLICATION, inplace = True)


# In[265]:


# AMT_GOODS_PRICE is handled - 

previous_applications_2.AMT_GOODS_PRICE.isnull().sum()


# In[266]:


previous_applications_2.NAME_CONTRACT_TYPE.value_counts()


# ### Here, we can see that XNA is only null values but has a string value of XNA and hence does not show up in our null values analysis.
# 
# > 1. Usually for categorical variable, we can impute using mode()
# > 2. However, in this case, Cash loans and Consumer loans are quite close and we cannot simply assume that the null values would take 'Cash loans' value
# 
# 
# ### Therefore, writing a code to randomly assign the two variables to XNA

# In[267]:


# Sorting the values in descending values so that all our XNA values are grouped together

previous_applications_2 = previous_applications_2.sort_values(['NAME_CONTRACT_TYPE'], ascending = False)
previous_applications_2 = previous_applications_2.reset_index(drop = True)


# In[268]:


len_NAME_CONTRACT_TYPE = len(previous_applications_2[previous_applications_2.NAME_CONTRACT_TYPE == 'XNA'])


# In[269]:


import random
NAME_CONTRACT_TYPE_RANDOM = ['Cash loans', 'Consumer loans']
for i in range(len_NAME_CONTRACT_TYPE):
    if previous_applications_2.NAME_CONTRACT_TYPE[i] == 'XNA':
        var = random.choice(NAME_CONTRACT_TYPE_RANDOM)
        previous_applications_2.NAME_CONTRACT_TYPE[i] = previous_applications_2.NAME_CONTRACT_TYPE[i].replace(previous_applications_2.NAME_CONTRACT_TYPE[i],var)


# In[270]:


# As you can see, the XNA values are randomly distributed between the first two values

previous_applications_2.NAME_CONTRACT_TYPE.value_counts()


# In[271]:


previous_applications_2.PRODUCT_COMBINATION.value_counts()


# In[272]:


previous_applications_2.PRODUCT_COMBINATION.isnull().sum()


# In[273]:


# Here we can impute to mode() as there is a considerable difference between first two values

previous_applications_2.PRODUCT_COMBINATION.fillna(previous_applications_2.PRODUCT_COMBINATION.mode()[0], inplace = True)


# In[274]:


previous_applications_2.PRODUCT_COMBINATION.isnull().sum()


# In[275]:


previous_applications_2.CNT_PAYMENT.isnull().sum()


# In[276]:


previous_applications_2.CNT_PAYMENT.value_counts().head()


# In[277]:


# Based on the data above, most CNT_PAYMENT values are one one of the following - 
# 12.0,6.0,0.0,10.0,24.0
# Writing a code to randomly assign a value between the selected 5 float values

import random
import math
CNT_PAYMENT_LIST = [12.0,6.0,0.0,10.0,24.0]

for i in range(len(previous_applications_2.CNT_PAYMENT)):
    if math.isnan(previous_applications_2.CNT_PAYMENT[i]):
        var = random.choice(CNT_PAYMENT_LIST)
        previous_applications_2.CNT_PAYMENT[i] = var


# In[278]:


previous_applications_2.CNT_PAYMENT.isnull().sum()


# In[279]:


# Let us now handle Missing Annuity values

previous_applications_2.AMT_ANNUITY.fillna(round(previous_applications_2.AMT_CREDIT/previous_applications_2.CNT_PAYMENT,1),inplace = True)


# In[280]:


previous_applications_2.AMT_ANNUITY.isnull().sum()


# In[281]:


# Even after using fillna() the values were not imputed because amount credit and cnt payment values were both 0
# In that case, we need to use a different approach - 

import math

for i in range(len(previous_applications_2.AMT_ANNUITY)):
    if math.isnan(previous_applications_2.AMT_ANNUITY[i]):
        if (previous_applications_2.AMT_CREDIT[i] == 0.0) and (previous_applications_2.CNT_PAYMENT[i] == 0.0):
            previous_applications_2.AMT_ANNUITY[i] = 0.0


# In[282]:


# The values have now been correctly handled

previous_applications_2.AMT_ANNUITY.isnull().sum()


# > - There are still a few columns with hidden missing values. They are set to XNA instead of NaN and hence cannot be found using the isnull() function
# > - Replacing all XNA to NaN and then imputing them with mode() values since they are all categorical

# In[283]:


previous_applications_2.NAME_GOODS_CATEGORY.value_counts()


# In[284]:


previous_applications_2.NAME_GOODS_CATEGORY.replace('XNA', np.NaN, inplace = True)


# In[285]:


previous_applications_2.NAME_GOODS_CATEGORY.fillna(previous_applications_2.NAME_GOODS_CATEGORY.mode()[0], inplace = True)


# In[286]:


previous_applications_2.NAME_GOODS_CATEGORY.value_counts()


# In[287]:


previous_applications_2.NAME_PRODUCT_TYPE.replace('XNA', np.NaN, inplace = True)


# In[288]:


previous_applications_2.NAME_PRODUCT_TYPE.fillna(previous_applications_2.NAME_PRODUCT_TYPE.mode()[0], inplace = True)


# In[289]:


previous_applications_2.NAME_PRODUCT_TYPE.value_counts()


# In[290]:


previous_applications_2.NAME_YIELD_GROUP.replace('XNA', np.NaN, inplace = True)


# In[291]:


previous_applications_2.NAME_YIELD_GROUP.fillna(previous_applications_2.NAME_YIELD_GROUP.mode()[0], inplace = True)


# In[292]:


previous_applications_2.NAME_YIELD_GROUP.value_counts()


# In[293]:


previous_applications_2.NFLAG_INSURED_ON_APPROVAL.value_counts()


# - Now even though NFLAG_INSURED_ON_APPROVAL is a categorical variable, it would seem prudent to assume that if some information was not filled, was not available, it can be treated as the client/consumer did not opt for the insurance

# In[294]:


previous_applications_2.NFLAG_INSURED_ON_APPROVAL.fillna(0.0,inplace = True)


# In[295]:


previous_applications_2.NFLAG_INSURED_ON_APPROVAL.isnull().sum()


# In[296]:


previous_applications_2.NAME_SELLER_INDUSTRY.value_counts()


# In[297]:


previous_applications_2.NAME_SELLER_INDUSTRY.replace('XNA', np.NaN, inplace = True)


# In[298]:


previous_applications_2.NAME_SELLER_INDUSTRY.fillna(previous_applications_2.NAME_SELLER_INDUSTRY.mode()[0], inplace = True)


# In[299]:


previous_applications_2.NAME_SELLER_INDUSTRY.value_counts()


# In[300]:


previous_applications_2.NAME_CASH_LOAN_PURPOSE.value_counts(normalize = True)


# In[301]:


# XAP and XNA are null values in the form of strings
# Converting all XAP and XNA to NaN - 

previous_applications_2.NAME_CASH_LOAN_PURPOSE.replace('XAP', np.NaN, inplace = True)
previous_applications_2.NAME_CASH_LOAN_PURPOSE.replace('XNA', np.NaN, inplace = True)


# In[302]:


previous_applications_2.NAME_CASH_LOAN_PURPOSE.isnull().sum()*100/len(previous_applications_2)


# - Make a note of the 95% missing values here, which force us to either drop the column or plot this usng only the 5% data available against the Target variable for previous applications dataset
# - For now, I am not treating these null values as I would like to analyse the given 5% data and see if gives us any insights

# In[303]:


previous_applications_2.isnull().sum()


# ### Exporting the data to a CSV

# In[304]:


previous_applications_2.to_csv('/Users/manalibhagwat-reddy/Desktop/Python py Files/previous_applications_2.csv')


# # Data Analysis - Application Dataset

# ### Univariate Analysis - 1. Total Income

# In[305]:


plt.figure(figsize = [10,6])
sns.set_theme()
sns.boxplot(application_data_2['AMT_INCOME_TOTAL'])
plt.xscale('log')
plt.title('Univariate Analysis - Total Income')
plt.show()


# ### `Inferences:`
# 
# - The Income amount has 1 extreme outlier and many close range outliers based on the boxplot
# - The Interquartile range is not too broad however they seem to have a similar breadth

# ### Univariate Analysis - 2. Total Credit Amount

# In[306]:


plt.figure(figsize = [10,6])
sns.set_theme()
sns.boxplot(application_data_2['AMT_CREDIT'])
plt.title('Univariate Analysis - Total Credit Amount')
plt.show()


# ### `Inferences:`
# 
# - The Issued Loan amount has many outliers based on the boxplot
# - The Interquartile range is broader than the Income boxplot

# ### Univariate Analysis - 3. Total Annuity Amount

# In[307]:


plt.figure(figsize = [10,8])
sns.set_theme()
sns.distplot(application_data_2['AMT_ANNUITY'], bins = 10, kde = False)
plt.title('Univariate Analysis - Annuity Amount')
plt.show()


# ### `Inferences:`
# 
# - The distribution plot tells us that the loan annuity distribution is right skewed or positively skewed
# - This means that the mean is to the right of the median/mode

# ### Univariate Analysis - 4. Gender & The Data Imbalance Ratio

# In[308]:


# Let us see the Gender distribution across the datasets for loan applicants

x = ['Females', 'Males']
y = [application_data_2.CODE_GENDER.value_counts()[0],application_data_2.CODE_GENDER.value_counts()[1]]

plt.figure(figsize = [8,8])
ax = sns.barplot(x = x, y = y)
plt.title('Univariate Analysis - Gender Distribution')


# ### `Inferences:`
# 
# - Female applicants are much higher in number than the male applicants
# - This could be in case the rate of interests are different (discounted) for women applicants
# - This also shows the data imbalance that we have for gendered applicants

# ### Univariate Analysis - 5. Region Population Relative

# In[309]:


plt.figure(figsize = [10,8])
sns.set_theme()
sns.boxplot(application_data_2.REGION_POPULATION_RELATIVE)
plt.title('Univariate Analysis - Region Population Relative')
plt.show()


# ### `Inferences:`
# 
# - This data shows normalized population of region where client lives (higher number means the client lives in more populated region)
# - There is only one outlier
# - The interquartile range is very broad meaning most values are inclusive in the IQR

# ### Univariate Analysis - 6. Housing Type

# In[310]:


labels = ['House / Apartment','With parents','Municipal apartment','Rented apartment','Office Apartment','Co-op Apartment',]
values = [x for x in application_data_2.NAME_HOUSING_TYPE.value_counts()]

fig = plt.figure(figsize =(10, 10))
plt.pie(values, labels = labels)
plt.title('Univariate Analysis - Housing Type')  
plt.show()


# ### `Inferences:`
# 
# - Majority of loan applicants have accommodation type - House/Apartment

# ### Univariate Analysis - 7. Marital Status

# In[311]:


labels = ['Married', 'Single/Not Married','Civil Marriage','Separated','Widowed','Unknown']
values = [x for x in application_data_2.NAME_FAMILY_STATUS.value_counts()]

fig = plt.figure(figsize =(10, 10))
plt.pie(values, labels = labels)
plt.title('Univariate Analysis - Marital Status')  
  

plt.show()


# ### `Inferences:`
# 
# - Majority of loan applicants are married

# ### Univariate Analysis - 8. Age of the Client in Calendar Years

# In[312]:


plt.figure(figsize = [8,6])
sns.set_theme()
sns.boxplot((application_data_2.DAYS_BIRTH)/-365)
plt.xlabel('Age in Years')
plt.title("Univariate Analysis - Client's Age in Years")
plt.show()


# ### `Inferences:`
# 
# - Majority of the applicants fall under 35 years to 55 years (approx)
# - The upper and lower extremes tell us that the applicants range only between 20 and 68 years of age
# - There are no outliers, meaning the non-earning crowd remains an untapped marked

# ### Univariate Analysis - 9. Employment Status of Client

# In[313]:


application_data_2.DAYS_EMPLOYED.describe()


# In[314]:


plt.figure(figsize = [8,6])
sns.set_theme()
sns.boxplot((application_data_2.DAYS_EMPLOYED))
plt.title('Univariate Analysis - Employment Status')

# plt.xlabel('Age in Years')
plt.show()


# - Since there is just the one outlier with a value completely different from the other, let us plot the boxplot by ignoring this outlier

# In[315]:


plt.figure(figsize = [8,6])
sns.set_theme()
sns.boxplot(x = 'DAYS_EMPLOYED', data = application_data_2, showfliers = False)
plt.title('Univariate Analysis - Employment Status')
plt.show()


# ### `Inferences:`
# 
# - The column gives us the data for how many number of days before the application the applicant started current employment
# - This data tells us that 75% of the applicants had a minimum of 9 months of experience before applying for a loan
# - The newly employed or freshers should therefore be targeted with attractive offers

# ### Univariate Analysis - 10. Rating of Client's City

# In[316]:


# Distribution of City Ratings across loan applicants

x = ['Rating - 2', 'Rating - 3', 'Rating - 1']
y = [application_data_2.REGION_RATING_CLIENT_W_CITY.value_counts()[2],
     application_data_2.REGION_RATING_CLIENT_W_CITY.value_counts()[3],
     application_data_2.REGION_RATING_CLIENT_W_CITY.value_counts()[1]]
plt.figure(figsize = [8,8])
ax = sns.barplot(x = x, y = y)


# ### `Inferences: `
# 
# - Majority of the applicants fall under Rating - 2 cities
# - This means that we should focus on why there are less applicants from Rating - 1 and Rating - 2 cities and expand there as well

# ### Univariate Analysis - Understanding the Target Variable

# In[317]:


# Let us see the Gender distribution across the datasets for loan applicants

x = ['Target - 0', 'Target - 1']
y = [application_data_2.TARGET.value_counts()[0],application_data_2.TARGET.value_counts()[1]]

plt.figure(figsize = [8,8])
ax = sns.barplot(x = x, y = y)
plt.title('Univariate Analysis - Target Data Imbalance')


# In[318]:


application_data_2.TARGET.value_counts(normalize = True)


# ### Bivariate Analysis - 1. Target v/s Total Income

# - Now, the income amount is a continuous variable, and is difficult to understand once plotted because of the wide range (as apparent from the first boxplot)
# - It is therefore prudent to convert this data into a categorical variable - Income Range

# In[319]:


application_data_2['INCOME_RANGE'] = pd.qcut(application_data_2['AMT_INCOME_TOTAL'], q = 3)


# #### Here, defining the three bins into three income ranges - low, mid, and high
# 
# - (25649.999, 117000.0]------ 1 - low income range
# - (117000.0, 180000.0] -------  2 - mid income range
# - (180000.0, 117000000.0]----3 - high income range

# In[320]:


# The datatype of INCOME_RANGE is CategoricalDtype.
# Type casting to String for ease in plotting

application_data_2.INCOME_RANGE = application_data_2.INCOME_RANGE.astype('string')


# In[321]:


# Replacing the values wit this range to represent - Low Income Range

application_data_2.INCOME_RANGE.replace('(25649.999, 117000.0]','Low Income Range', inplace = True)


# In[322]:


# Replacing the values wit this range to represent - Mid Income Range

application_data_2.INCOME_RANGE.replace('(117000.0, 180000.0]','Mid Income Range', inplace = True)


# In[323]:


# Replacing the values wit this range to represent - High Income Range

application_data_2.INCOME_RANGE.replace('(180000.0, 117000000.0]','High Income Range', inplace = True)


# In[376]:


application_data_2.CODE_GENDER.replace('XNA',application_data_2.CODE_GENDER.mode()[0], inplace = True)


# - Now, we can finally perform the bivariate analysis between Target and Total Income based on Gender distribution

# In[377]:


plt.figure(figsize = [8,8])
ax = sns.barplot(x = 'INCOME_RANGE', y = 'TARGET', hue = 'CODE_GENDER', data = application_data_2)
plt.title('Multivariate Analysis - Target v/s Total Income')
plt.show()


# - The barplot above shows us that Male applicants from Low Income Range (Income range between 25649.999 and 117000.0) are more likely to default followed closely by Male applicants from Mid Income Range (Income range between 117000.0 and 180000.0)

# ### Bivariate Analysis - 2. Target v/s Occupation Type

# In[325]:


plt.figure(figsize = [12,10])
ax = sns.barplot(x = application_data_2['OCCUPATION_TYPE'], y = application_data_2['TARGET'], data = application_data_2)
plt.title('Bivariate Analysis - Target v/s Occupation Type')
plt.xticks(rotation = 90)
plt.show()


# ### `Inferences:`
# 
# - Low skill Laborers are highly likely to default on loans (as compared to regular laborers as well)
# - Accountants and High Skill Tech Staff are least likely to default on loans - probably because of steady incomes

# ### Segmented Multivariate Analysis - 3. Total Credit (Loan Amount)  v/s Income Type with Target Differentiation

# In[326]:


plt.figure(figsize = [12,10])
ax = sns.barplot(hue = 'NAME_INCOME_TYPE', y = 'AMT_CREDIT', x = 'TARGET', data = application_data_2)
plt.title('Multivariate Analysis - Total Credit v/s Income Type')
plt.show()


# ### `Inferences:`
# 
# - Students and Businessmen are highly unlikely to Default on loans
# - Businessmen have a higher credit availability as well with no defaults (as per this data)
# - Women on Maternity Leave are much more likely to Default. This might be due to unpaid leaves by the organizations. In some cases, the women might not be earning, but the additional expenses cause the loans to be defaulted

# ### Bivariate Analysis - 4. Target  v/s Education Type with Family Status Differentiation

# In[327]:


plt.figure(figsize = [12,10])
ax = sns.barplot(x = 'NAME_EDUCATION_TYPE', y = 'TARGET', hue = 'NAME_FAMILY_STATUS', data = application_data_2)
plt.title('Multivariate Analysis - Target v/s Education Type Across Family Status')
plt.show()


# ### `Inferences:`
# 
# - Applicants who have been separated from their spouses and have lower secondary education are more likely to default
# - Married applicants with an academic degree are very less likely to default

# ### Bivariate Analysis - 5. Target  v/s Education Type with Income Range Differentiation

# In[328]:


plt.figure(figsize = [12,10])
ax = sns.barplot(x = 'NAME_EDUCATION_TYPE', y = 'TARGET', hue = 'INCOME_RANGE', data = application_data_2)
plt.title('Multivariate Analysis - Target v/s Education Type Across Income Range')
plt.show()


# ### `Inferences:`
# 
# - Applicants with Academic Degree and High Income are least likely to default
# - Applicants with Lower Secondary Education and Mid Income Range are very likely to default followed by Lower Secondary - Low Income Range

# ## Multivariate -  Let us now try to find correlation between different columns

# - Ensuring FLAG_OWN_CAR and FLAG_OWN_REALTY are in binary instead of Y/N

# In[329]:


application_data_2.FLAG_OWN_CAR.replace('Y',1, inplace = True)


# In[330]:


application_data_2.FLAG_OWN_CAR.replace('N',0, inplace = True)


# In[331]:


application_data_2.FLAG_OWN_CAR = application_data_2.FLAG_OWN_CAR.astype('int')


# In[332]:


application_data_2.FLAG_OWN_CAR.value_counts()


# In[333]:


application_data_2.FLAG_OWN_REALTY.replace('Y',1, inplace = True)


# In[334]:


application_data_2.FLAG_OWN_REALTY.replace('N',0, inplace = True)


# In[335]:


application_data_2.FLAG_OWN_REALTY.value_counts()


# ### Extracting columns for which correlation needs to be established

# In[336]:


res1 = application_data_2[['FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT','CNT_FAM_MEMBERS']]
corrMatrix = res1.corr()

plt.figure(figsize = [10,8])
sns.heatmap(data = corrMatrix, annot = True, cmap = 'Blues')
plt.title('Multivariate Analysis - Correlation Between Different Columns')
plt.show()


# ### `Inferences:`
# 
# - Based on the correlation between the columns represented in the heatmap, there is a very strong relation between count of children and count of family. This is obvious as the two are not mutually exhaustive
# - There is also a positive correlation between credit amount and income
# - There is very less correlation between  count of children and the credit amount
# - There is a slightly higher relation between count of family members and credit amount

# ## Segmented Univariate - Credit Data Based on Target Variable

# In[337]:


Target_1 = application_data_2[application_data_2.TARGET == 1]
Target_0 = application_data_2[application_data_2.TARGET == 0]


# In[338]:


Target_0.AMT_CREDIT.describe()


# In[339]:


Target_1.AMT_CREDIT.describe()


# In[340]:


plt.figure(figsize = [8,10])
ax = sns.boxplot(y = 'AMT_CREDIT' ,x = 'TARGET', data = application_data_2)
plt.title('Segmented Univariate Analysis for Target 1 and Target 0')
plt.show()


# ### `Inferences:`
# 
# - The mean of both - Target 1 and Target 0 are almost same
# - The interquartile range for Target 0 is broader than Target 1

# ## Segmented Multivariate - Credit Data v/s Target Variable with respect to Education Type

# In[341]:


plt.figure(figsize = [8,10])
ax = sns.boxplot(y = 'AMT_CREDIT' ,x = 'TARGET', hue = 'NAME_EDUCATION_TYPE',data = application_data_2)
plt.title('Segmented Multivariate Analysis for Target 1 and Target 0')
plt.show()


# ### `Inferences:`
# 
# - For Target 1, upper threshold is highest for Higher Education - which means they apply for higher credit but are also likely to default
# - For Target 0, the Academic degree holders seems to have a good spread in the Interquartile Range and fewer outliers

# ## Segmented Multivariate - Credit Data v/s Target Variable with respect to Income Range

# In[342]:


plt.figure(figsize = [8,10])
ax = sns.boxplot(y = 'AMT_CREDIT' ,x = 'TARGET', hue = 'INCOME_RANGE',data = application_data_2)
plt.title('Segmented Multivariate Analysis for Target 1 and Target 0')
plt.show()


# ### `Inferences:`
# 
# - For Target 1, the mean is not similar for all three income ranges. This means, there is no threshold limit that can be put on as a red flag while applicants from any of the Income Ranges are applying for a loan
# - For Target 0, the High Income Range applicants seem to have not only applied for a higher loan but have also paid it off without difficulties

# # Data Analysis - Previous Application Dataset

# ### Univariate Analysis - 1. Credit Amount

# In[343]:


plt.figure(figsize = [10,6])
sns.set_theme()
sns.boxplot(previous_applications_2['AMT_CREDIT'])
plt.title('Univariate Analysis - Total Credit Amount')
plt.show()


# - Since the outliers are not giving us a complete picture, let us ignore them for our analysis

# In[344]:


plt.figure(figsize = [10,6])
sns.set_theme()
sns.boxplot(x = 'AMT_CREDIT',data = previous_applications_2, showfliers = False)
plt.title('Univariate Analysis - Total Credit Amount')
plt.show()


# ### `INFERENCES:`
# 
# - The data is for total credit amount for previous loan applications
# - After excluding the outliers, we can clearly see that the third quartile is much broader than the first quartile

# ### Univariate Analysis - 2. Annuity Amount

# In[345]:


plt.figure(figsize = [10,6])
sns.boxplot(application_data_2['AMT_ANNUITY'])
plt.title('Univariate Analysis - Total Annuity Amount')
plt.show()


# ### `Inferences:`
# 
# - The Interquartile Range shows that the first quartile is smaller than the third quartile
# - The outlier are clustered and a few are very extreme
# - This is in relation to the extreme outliers of the credit amount

# ### Univariate Analysis - 3. Loan Status for Previous Applications

# In[346]:


labels = ['Approved', 'Canceled','Refused','Unused Offer']
values = [x for x in previous_applications_2.NAME_CONTRACT_STATUS.value_counts()]

fig = plt.figure(figsize =(10, 10))
plt.pie(values, labels = labels)
plt.title('Univariate Analysis - Loan Status for Previous Applications')

plt.show()


# ### `Inferences:`
# 
# - Majority of the loans get approved
# - The second highest percentage is that of cancelled

# ### Univariate Analysis - 4. Loan Types for Previous Applications

# In[347]:


# Distribution of City Ratings across loan applicants

x = ['Cash Loans', 'Consumer Loans', 'Revolving Loans']
y = [previous_applications_2.NAME_CONTRACT_TYPE.value_counts()[0],
     previous_applications_2.NAME_CONTRACT_TYPE.value_counts()[1],
     previous_applications_2.NAME_CONTRACT_TYPE.value_counts()[2]]

plt.figure(figsize = [8,8])
plt.title('Univariate Analysis - Loan Types for Previous Applications')
ax = sns.barplot(x = x, y = y)


# ### `Inferences:`
# 
# - Applicants mostly take cash or consumer loans
# - Revolving loans should be made more attractive and the reason for people not opting for them should be understood

# ### Univariate Analysis - 5. Cash Loan Purposes of the Clients

# - Though this data is not sufficient to perform any analysis, plotting the chart nevertheless, to see how the available data is distributed

# In[348]:


plt.figure(figsize = [12,12])
previous_applications_2.NAME_CASH_LOAN_PURPOSE.value_counts().plot.barh()
plt.title('Univariate Analysis - Cash Loan Purpose of Client')
plt.show()


# ### `Inferences:`
# 
# - Most cash loans are taken for repairs followed by 'other' category and urgent cash needs

# ### Univariate Analysis - 6. Types of Clients from the Previous Data

# In[349]:


# Ensuring XNA values are corrected treated - 

previous_applications_2.NAME_CLIENT_TYPE.replace('XNA', previous_applications_2.NAME_CLIENT_TYPE.mode()[0], inplace = True)


# In[350]:


# Distribution of types of loan applicants

x = ['Repeater', 'New', 'Refreshed']
y = [previous_applications_2.NAME_CLIENT_TYPE.value_counts()[0],
     previous_applications_2.NAME_CLIENT_TYPE.value_counts()[1],
     previous_applications_2.NAME_CLIENT_TYPE.value_counts()[2]]

plt.figure(figsize = [8,8])
plt.title('Univariate Analysis - Types of Clients from Previous Applications')
ax = sns.barplot(x = x, y = y)


# ### `Inferences:`
# 
# - Reapeated customers are in majority followed by New applicants

# ### Univariate Analysis - 7. Status of Insurance opted in the Previous Data (Data Imbalance)

# In[1084]:


# Distribution of types of loan applicants

x = ['Insured on Approval', 'Not Insured on Approval']
y = [previous_applications_2.NFLAG_INSURED_ON_APPROVAL.value_counts()[1],
     previous_applications_2.NFLAG_INSURED_ON_APPROVAL.value_counts()[0]]

plt.figure(figsize = [8,8])
plt.title('Univariate Analysis - Status of Insurance Opted by Clients')
ax = sns.barplot(x = x, y = y)


# ### `Inferences:`
# 
# - Applicants have mostly not opted for insurance on loan approval

# ### Multivariate Analysis - 1. Total Credit Amount v/s Loan Type distributed over Approval Status

# In[351]:


plt.figure(figsize = [12,10])
ax = sns.barplot(x = 'NAME_CONTRACT_TYPE', y = 'AMT_CREDIT', hue = 'NAME_CONTRACT_STATUS', data = previous_applications_2)
plt.title('Multivariate Analysis - Total Credit Amount v/s Loan Typed based on Approval Status')
plt.show()


# ### `Inferences:`
# 
# - Cash loans with a higher credit have been refused

# ### Multivariate Analysis - 2. Total Credit Amount v/s Loan Application Days distributed over Approval Status

# In[352]:


plt.figure(figsize = [12,10])
ax = sns.barplot(x = 'WEEKDAY_APPR_PROCESS_START', y = 'AMT_CREDIT', hue = 'NAME_CONTRACT_STATUS', data = previous_applications_2)
plt.title('Multivariate Analysis - Total Credit Amount v/s Day of the Week based on Approval Status')
plt.show()


# ### `Inferences:`
# 
# - Initial assumption was that the day of the week when the loan was applied for, might have some influence on them getting Approved or otherwise
# - However, we can see that the day of the week to apply for loan has no effect on it being Approved/Refused
# - The general trend however seems to be more on weekdays and less on weekends

# ### Multivariate Analysis - 3.  Cash Loan Purpose v/s Total Credit Amount

# In[353]:


plt.figure(figsize = [16,14])
ax = sns.barplot(y = 'NAME_CASH_LOAN_PURPOSE', x = 'AMT_CREDIT', hue = 'NAME_CONTRACT_STATUS' ,data = previous_applications_2)
plt.title('Multivariate Analysis - Loan Typed v/s Total Credit Amount based on Approval Status')
plt.show()


# ### `Inferences:`
# 
# - Hardly any unused loans observed here
# - Loans were most Refused for buying a new car

# ### Bivariate Analysis - 4.  Client Type and Approval Status

# In[354]:


previous_applications_2.NAME_CLIENT_TYPE.value_counts()


# In[378]:


plt.figure(figsize = [8,8])
plt.title('Bivariate Analysis - Client Type and Approval Status')
ax = sns.barplot(x = 'NAME_CLIENT_TYPE', y = 'AMT_CREDIT', hue = 'NAME_CONTRACT_STATUS', data = previous_applications_2)
plt.show()


# ### `Inferences:`
# 
# - Loan Refusal late is the least for New clients and most for Repeated clients
# - Approval rate however is most for Repeated clients and least for New clients
# - The Refreshed and Repeated clients have a similar demographic as per loan amount is concerned

# ## Merging the Two Dataframes

# In[358]:


# Necessary columns from preapplication_data_2

prev_appl_cols = ['SK_ID_CURR',
'NAME_CONTRACT_TYPE',
'AMT_CREDIT',
'NAME_CONTRACT_STATUS',
'NAME_CLIENT_TYPE',
'NFLAG_INSURED_ON_APPROVAL']


# In[359]:


# Necessary columns from application_data_2

appl_data_cols = ['SK_ID_CURR',
'TARGET',
'NAME_CONTRACT_TYPE',
'CODE_GENDER',
'AMT_INCOME_TOTAL',
'AMT_CREDIT',
'NAME_INCOME_TYPE',
'NAME_EDUCATION_TYPE',
'NAME_FAMILY_STATUS',
'NAME_HOUSING_TYPE',
'OCCUPATION_TYPE',
'ORGANIZATION_TYPE']


# In[360]:


application_data_3 = application_data_2[appl_data_cols]


# In[361]:


previous_applications_3 = previous_applications_2[prev_appl_cols]


# - Merging the two dfs and creating a new one

# In[362]:


merged_df = pd.merge(application_data_3,previous_applications_3, on = 'SK_ID_CURR', how = 'inner')


# In[363]:


# Performing sanity check based on first 20 rows in the merged dataframe

merged_df.head(20)


# In[364]:


merged_df.shape


# > `Sanity Check:`
# 
# - We have 17 columns and 1413701 rows in this dataframe

# In[365]:


merged_df.info()


# In[366]:


# Renaming the columns from auto nomenclature during merging to something that is more legible

merged_df.rename(columns = {'NAME_CONTRACT_TYPE_x':'NAME_CONTRACT_TYPE','NAME_CONTRACT_TYPE_y':'NAME_CONTRACT_TYPE_PREV',
                           'AMT_CREDIT_x':'AMT_CREDIT', 'AMT_CREDIT_y': 'AMT_CREDIT_PREV'}, inplace = True)


# In[367]:


# Dataframe with renamed columns - 

merged_df.head()


# ## Bivariate Analysis - 1. Target Column v/s Opting for Insurance

# - #### Here, I am trying to understand if there is any correlation between applicants opting for insurance and them defaulting on the loans

# In[368]:


res1 = merged_df[['TARGET','NFLAG_INSURED_ON_APPROVAL']]
corrMatrix = res1.corr()

plt.figure(figsize = [8,6])
sns.heatmap(data = corrMatrix, annot = True, cmap = 'Blues')
plt.title('Bivariate Analysis - Target v/s Opting for Insurance')
plt.show()


# ### `Inferences:` 
# - Nothing conclusive could be drawn.
# - Opting for insurance does not seem to affect the default rates very heavily

# ## Multivariate Analysis - 2. Target Column v/s Type of Accomodation

# In[369]:


plt.figure(figsize = [12,10])
ax = sns.barplot(x = 'NAME_HOUSING_TYPE', y = 'AMT_CREDIT', hue = 'TARGET', data = merged_df)
plt.title('Bivariate Analysis - Target v/s Type of Accommodation')
plt.show()


# ### `Inferences:` 
# - Applicants that have defaulted the least are the ones staying with their parents
# - Applicants that have defaulted the most are from co-operative apartments and/or Office Apartments

# ## Bivariate Analysis - 3. Total Amount Credit v/s Target Column based on Client Types

# In[370]:


merged_df.NAME_CLIENT_TYPE.replace('XNA',merged_df.NAME_CLIENT_TYPE.mode()[0], inplace = True)


# In[371]:


plt.figure(figsize = [8,10])
ax = sns.barplot(hue = 'NAME_CLIENT_TYPE', y = 'AMT_CREDIT', x = 'TARGET', data = merged_df)
plt.title('Bivariate Analysis - Total Credit v/s Target based on Client Types')
plt.show()


# ### `Inferences:` 
# - Rate of Default is almost equal for New, Repeated, and Refreshed clients
# - This means that, just because an applicant is an old client, there should be no assumption that won't default the loan that they're applying for

# ## Bivariate Analysis - 4. Occupation Type v/s Total Amount Credit based on Client Types

# In[372]:


plt.figure(figsize = [10,12])
ax = sns.barplot(y = 'OCCUPATION_TYPE', x = 'AMT_CREDIT_PREV', hue = 'NAME_CONTRACT_STATUS',data = merged_df)
plt.title('Bivariate Analysis - Occupation Type v/s Total Amount Credit based on Client Types')
plt.show()


# ### `Inferences:`
# 
# - Managers have been most refused of the loans. So have the applicants from Realty background or waiting staff (barmen). It indicates it might be driven by uneven and irregular tips and sales commissions
# - Approval rate is high in accountants and general population with a relatively stable income

# ## Segmented Multivariate - 5. Total Credit v/s Previous Credit with respect to Target Variable

# In[373]:


plt.figure(figsize = [10,12])
ax = sns.scatterplot(y = 'AMT_CREDIT' ,x = 'AMT_CREDIT_PREV', hue = 'TARGET',data = merged_df)
plt.title('Segmented Multivariate Analysis - Current & Outstanding Loan wrt Target')
plt.show()


# ### `Inferences:`
# 
# - Direction - Linear/Linear CLuster
# - Form - Generally Positive
# - Strength - Strong
# - Outliers - Present
# 
# > - The pattern indicates that, higher the outstanding loan, higher are the chances of the applicants defaulting on the current loan
# > - In some cases, the Target 1 is not affected by the correlation between the previous and current loan amounts and there might be some other factors influencing the tendency to loan default

# In[ ]:





# ### SUMMARY OF THE EDA:
# 
# > ### 1. The driver variables for Target 1/0 – 
# >> - Income Range
# >> - Occupation Type
# >> - Education Type and Education Level
# >> - Family status – (Marital status, no. of family members)
# >> - Type of Accommodation
# >> - Previous outstanding loans
# 
# > ### 2. The driver variables for getting the loan Approved/Rejected – 
# >> - Type of loan applied
# >> - Purpose of loan applied
# >> - Client type – (New/ Repeated)
# 
# > ### 3. Assumptions made in the EDA – 
# >> - Columns with more than 50% missing values have been dropped
# >> - Categorical missing values have been imputed by a value that is picked at random from the top 2 or 3
# 
# > ### 4. Validating hypotheses for correlation –
# >> - There might be some correlation between clients opting for insurance after loan getting approved and their tendency to default
# >> - The day of the week on which loan was applied, might have some effect on the loan getting Approved or otherwise

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[374]:


import jovian 
jovian.commit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




