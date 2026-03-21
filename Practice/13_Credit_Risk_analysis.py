# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# %%
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# %%
train_input = pd.read_csv("Credit_Risk_Train_Data.csv")
validate_input = pd.read_csv("Credit_Risk_Validate_Data.csv")

# %%
print(train_input.columns)
print(train_input.shape)
print(validate_input.columns)
print(validate_input.shape)

# %%
validate_input.rename(columns={"outcome":"Loan_Status"},inplace=True)

# %%
data_all = pd.concat([train_input,validate_input],ignore_index=True)
data_all.shape

# %%
data_all.head()

# %%
data_all.tail()

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %% [markdown]
# ## Missing Value Pre-Processing

# %%
plt.figure(figsize=(20,10))
sns.heatmap(data_all.isnull(), cbar=False)

# %%
data_all.isnull().sum() 

# %% [markdown]
# ## NaN fill with mode:

# %%
Counter(data_all['Gender'])

# %%
data_all.fillna({'Gender':'Male'},inplace=True)

# %%
Counter(data_all['Gender'])

# %%
print(Counter(data_all['Married'])) 

# %%
data_all.fillna({'Married':'Yes'},inplace=True)

# %%
print(Counter(data_all['Married']))

# %%
data_all.fillna({'Married':'Yes'},inplace=True)

# %%
data_all.isnull().sum()

# %% [markdown]
# ## NaN fill with Cross Tab

# %%
Counter(data_all['Dependents'])

# %%
pd.crosstab(data_all['Married'],data_all['Dependents'].isnull())

# %%
pd.crosstab(data_all['Dependents'],data_all['Married'])

# %%
bachelor_nulldependent = data_all[(data_all['Married']=="No") &
        (data_all['Dependents'].isnull())].index.tolist()

# %%
print(bachelor_nulldependent)

# %%
data_all['Dependents'].iloc[bachelor_nulldependent]='0'

# %%
Counter(data_all['Dependents'])

# %%
pd.crosstab(data_all['Gender'], data_all['Dependents'])

# %%
pd.crosstab((data_all['Gender']=='Male') &
            (data_all['Married']=='Yes'),data_all['Dependents'])

# %%
data_all['Dependents'].iloc[data_all[data_all['Dependents'].isnull()].index.tolist()]="1"

# %%
data_all.isnull().sum()

# %%
Counter(data_all['Self_Employed'])

# %%
data_all.fillna({'Self_Employed':'No'},inplace=True)

# %%
data_all.isnull().sum()

# %% [markdown]
# ## NanN fill with Mean

# %%
pd.crosstab(data_all['LoanAmount'].isnull(),data_all['Loan_Amount_Term'].isnull())

# %%
pd.crosstab(data_all['LoanAmount'].isnull(), data_all['Loan_Amount_Term'])

# %%
data_all.groupby(data_all['Loan_Amount_Term'])['LoanAmount'].mean()

# %%
# lets fill the missing values in LoanAmount
data_all['LoanAmount'][(data_all['LoanAmount'].isnull()) & (data_all['Loan_Amount_Term']==360)]=144

# lets fill the missing values in LoanAmount
data_all['LoanAmount'][(data_all['LoanAmount'].isnull()) & (data_all['Loan_Amount_Term']==480)]=137

# %%
data_all['LoanAmount'][(data_all['LoanAmount'].isnull())]=130

# %%
(data_all['Loan_Amount_Term']).value_counts()

# %%
data_all['Loan_Amount_Term'][data_all['Loan_Amount_Term'].isnull()]=360

# %%
data_all.isnull().sum()

# %%
data_all['Credit_History'].value_counts()

# %%
pd.crosstab(data_all['Self_Employed'],data_all['Credit_History'])

# %%
pd.crosstab(data_all['Education'],data_all['Credit_History'])

# %%
# Married makes no difference
pd.crosstab(data_all['Married'],data_all['Credit_History'])

# %%
data_all.fillna({'Credit_History':1},inplace=True)

# %%
data_all.isnull().sum()

# %% [markdown]
# ## Categorical Feature Engineering

# %%
data_all.head()

# %%
data_all['Dependents'].value_counts()

# %%
data_all['Dependents'][data_all['Dependents']=='3+']='3'
data_all['Dependents'].value_counts()

# %%
data_all['Dependents'].head()

# %%
from sklearn.preprocessing import LabelEncoder

# %%
encoder = LabelEncoder()

# %%
result = encoder.fit_transform(data_all['Dependents'])
print(result[1:10])

# %%
dependents = pd.Series(result)
dependents.value_counts()

# %%
data_all['Dependents']=dependents

# %%
data_all['Dependents'].head()

# %%
data_all_new = pd.get_dummies(data_all.drop(['Loan_ID'],axis=1), drop_first=True, dtype=int)

# %%
data_all_new.head()

# %%
X = data_all_new.drop(['Loan_Status_Y'],axis=1)
y = data_all_new['Loan_Status_Y']

# %%
X.head()

# %%
y.head()

# %% [markdown]
# ## Data Splitting:

# %%
from sklearn.model_selection import train_test_split

# %%
seed = 42

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=seed)

# %%
X_train.shape

# %%
X_test.shape

# %% [markdown]
# # Feature Scaling:

# %%
from sklearn.preprocessing import StandardScaler

# %%
scaler = StandardScaler()

# %%
# Fit only to the training data
scaler.fit(X)

# %%
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%

X_train[:5]

# %%
y_train.describe()

# %%
y_test.describe()

# %% [markdown]
# # Model Training and Comparision

# %% [markdown]
# ## 1) Support Vector Machine Classification model

# %%
from sklearn.svm import SVC

# %%
# Train the SVM regression model
svm = SVC(kernel='linear')
svm.fit(X_train,y_train)

# %%
# Make predictions on the testing set
y_pred = svm.predict(X_test)
print(y_pred)

# %%
accuracy_SVR = svm.score(X_test,y_test)

print(f'Accuracy score of Support Vector Machine Regression is {(accuracy_SVR*100):.2f}')

# %% [markdown]
# ## 2) Logistic Regression Model

# %%
%pip install statsmodels

# %%
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf

# %%


# %%
LR  = LogisticRegression()
LR.fit(X_train,y_train)

# %%
y_pred = LR.predict(X_test)


# %%
accuracy = LR.score(X_test,y_test)
print(f'Accuracy of Logistic Regression is {accuracy*100:.2f}')

# %% [markdown]
# ### 3) K-NN model predictions.

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# %%
# initializes and runs the classifier
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X_train, y_train)

# %%

y_pred = classifier.predict(X_test) 

# gives the confusion matrix
conf = metrics.confusion_matrix(y_test, y_pred)
print(conf)

# %%
sns.heatmap(conf, annot=True, fmt=".3f", square = True);
plt.ylabel('Actual');
plt.xlabel('Predicted');
plt.title('Confusion matrix for the 3NN \n algorithm performed on the Credit Risk Analysis', fontsize = 15);

# %%
accuracy_score_clf = classifier.score(X_test,y_test)
print(f'Accuracy of K-NN {(accuracy_score_clf*100):.2f}')

# %%



