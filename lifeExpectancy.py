# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score                # we are using this for model tunning

from warnings import filterwarnings
filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


Life_Expectancy_Data = pd.read_csv("Life Expectancy Data.csv")
data = Life_Expectancy_Data.copy()
data = data.dropna()            # If there is a missing or empty observation, delete it. Or 'data.fillna(data.mean(), inplace=True)' with this make NaN values take mean

lindata = data.copy()
multidata = data.copy()
polydata = data.copy()
RFdata = data.copy()
logdata = data.copy()


# # 
# Linear Regression

# In[270]:


lindata.info()


# In[271]:


lindata.head()


# In[272]:


lindata.corr()


# Looking at heatmap, there is a good relationship (correlation exists) between the best 'GDP' and 'percentage expenditure' in the Life Expectation data.
# 

# In[273]:


# plot the heatmap
corr = lindata.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)



# Here it is better to establish a linear model between 'GDP' and 'percentage expenditure'. Let's see what our spending percentages are compared to the "GDP" increase. Let's create and fit our linear model.

# In[274]:


linear_reg = LinearRegression()
x = lindata.GDP.values.reshape(-1,1)
y = lindata['percentage expenditure'].values.reshape(-1,1)          

linear_reg.fit(x,y)


# ## y = b0 + b1*x is our linear regression model.
# Let's see estimated percentage of expenditure in GDP 10 thousand:

# In[275]:


b0 = linear_reg.predict(([[10000]]))       
print("b0: ", b0)

b1 = linear_reg.coef_
print("b1: ", b1)


# In[276]:


x_array = np.arange(min(lindata.GDP),max(lindata.GDP)).reshape(-1,1)  # this for information about the line to be predicted

plt.scatter(x=x,y=y)
y_head = linear_reg.predict(x_array)                                 # this is predict percentage of expenditure
plt.plot(x_array,y_head,color="red")
plt.show()

from sklearn import metrics
print("Mean Absolute Error: ", metrics.mean_absolute_error(x_array,y_head))
print("Mean Squared Error: ", metrics.mean_squared_error(x_array,y_head))
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(x_array, y_head)))



# In[277]:


print(r2_score(y, linear_reg.predict(x)))


# #### The conclusion here is: the estimate made has 92% accuracy.

# # Multi Linear Regression
# 
# * Here, let's take a look at the variable that depends on Life Expectancy.
# * If there is missing observation or empty, delete it. Or 'data.fillna (data.mean (), inplace = True)' with this make NaN values averaged.
# * When we look at the data, Country and Status columns are composed of objects. Because we need to be int or float.
# * and let's take the last two columns (Income composition of resources, schooling) as independent variables.

# In[278]:


Life_Expectancy_Data = pd.read_csv("Life Expectancy Data.csv")
data = Life_Expectancy_Data.copy()
data = data.dropna()

multidata = data.copy()

multidata.drop(["Country", "Status"], axis=1, inplace=True)             # When we look at the data, Country and Status columns are composed of objects. Because we need to be int or float.

x = multidata.iloc[:, [-2,-1]].values                                   # I took the last two columns (Income composition of resources, schooling) as independent variables.
y = multidata["percentage expenditure"].values.reshape(-1,1)            # our independent variable


# In[279]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)
lm = LinearRegression()
model = lm.fit(x_train,y_train)


# In[280]:


print("b0: ", lm.intercept_)
print("b1,b2: ", lm.coef_)


# We look at what the data set we created here will affect how much it will affect our survival.

# In[281]:


new_data = [[0.4,8], [0.5,10]]   
new_data = pd.DataFrame(new_data).T       # .T is transfor the chart.
model.predict(new_data) 


# ### Now let's look at the correctness of the evaluation we made. If the difference between the train error and the test error is not much, modeling is good.

# In[282]:


rmse = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
rmse


# In[283]:


model.score(x_train, y_train) 


# ### CV $r^2$ value of the model:

# In[284]:


cross_val_score(model, x_train,  y_train, cv= 10, scoring="r2").mean()


# Predicts of Train set values:

# In[285]:


y_head = model.predict(x_test)
y_head[0:5]


# In[286]:


y_test_1 =np.array(range(0,len(y_test)))


# In[287]:


# r2 value: 
r2_degeri = r2_score(y_test, y_head)
print("Test r2 error = ",r2_degeri) 

plt.plot(y_test_1,y_test,color="r")
plt.plot(y_test_1,y_head,color="blue")
plt.show()


# # Polynomial Regression
# 
# We will use the same data set.

# In[288]:


from sklearn.preprocessing import PolynomialFeatures     # this gives properties of polynomial

Life_Expectancy_Data = pd.read_csv("Life Expectancy Data.csv")
data = Life_Expectancy_Data.copy()
data = data.dropna()        

polydata = data.copy()


# Let's see what our spending percentages are compared to the "GDP" increase. Let's create and fit our linear model.

# In[289]:


linear_reg = LinearRegression()
x = polydata.GDP.values.reshape(-1,1)
y = polydata['percentage expenditure'].values.reshape(-1,1)          

linear_reg.fit(x,y)


# In[290]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)


# Let's look at the 15th degree. If it's not, we should change it.

# In[291]:


polynomial_regression = PolynomialFeatures(degree = 15)    
x_polynomial = polynomial_regression.fit_transform(x)

linear_reg2 = LinearRegression()
linear_reg2.fit(x_polynomial,y)

y_head = linear_reg2.predict(x_polynomial)

plt.plot(x,y_head,color="green",label="poly")
plt.legend()
plt.show()


# With degree we determine the precision of our forecast. If it is too large, it will deteriorate, so it is necessary to determine according to the data.

# In[292]:


pol_reg = PolynomialFeatures(degree = 8)                    

level_poly = pol_reg.fit_transform(x_train)                 # According to the polynomial, x_train is defined

lm = LinearRegression()                                     
lm.fit(level_poly,y_train)


# In[293]:


y_head = lm.predict(pol_reg.fit_transform(x_train))
y_test =np.array(range(0,len(y_train)))


# Consistency and scatter drawing of $r^2$ model:

# In[294]:


r2 = r2_score(y_train, y_head)
print("r2 value: ", r2)                               # percentage of significance


plt.scatter(y_test, y_train, color="red")
plt.scatter(y_test, y_head, color = "g")
plt.xlabel("GDP")
plt.ylabel("percentage expenditure")
plt.show()


# In[295]:


plt.plot(y_test,y_train, color="red")
plt.plot(y_test, y_head, color = "blue")
plt.xlabel("GDP")
plt.ylabel("percentage expenditure")
plt.show()


# # Decision Tree Regression

# In[296]:


from sklearn.tree import DecisionTreeRegressor               # for our predict model

Life_Expectancy_Data = pd.read_csv("Life Expectancy Data.csv")
data = Life_Expectancy_Data.copy()
data = data.dropna()                                         # same is done 

DTdata = data.copy()


# In[297]:


x = polydata.GDP.values.reshape(-1,1)
y = polydata['percentage expenditure'].values.reshape(-1,1)


# In[298]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)


# Let's see Expenditure percentage estimation of the country with "GDP" value of 1000:

# In[299]:


DT_reg = DecisionTreeRegressor()           # created model
DT_reg.fit(x_train,y_train)                # fitted model according to train values

print(DT_reg.predict([[1000]]))            


# In[300]:


x_array = np.arange(min(x),max(x),0.01).reshape(-1,1)   # line information to be drawn as a predict
y_head = DT_reg.predict(x_array)                        # percentage of spend estimate

plt.scatter(x,y, color="red")
plt.plot(x_array,y_head,color="blue")
plt.xlabel("GDP")
plt.ylabel("percentage expenditure")
plt.show()


# ### Result: See how it is nice picture and very successful accuracy score.

# # Random Forest Regression
# * A logic of DecisionTree. For example, 3000 sample data is selected from 100 thousand data and the result is obtained.

# In[301]:


from sklearn.ensemble import RandomForestRegressor           # for our predict model

Life_Expectancy_Data = pd.read_csv("Life Expectancy Data.csv")
data = Life_Expectancy_Data.copy()
data = data.dropna()                                         # same is done 

RFdata = data.copy()


# In[302]:


x = polydata.GDP.values.reshape(-1,1)
y = polydata['percentage expenditure'].values.reshape(-1,1)


# Create regression with 5 DecisionTreeReg in the sklearn RandomForestRegressor model. We can do as many as we want. Giving random_state does not change the outcome. When we say 1, it should not change once.

# In[303]:


RF_reg = RandomForestRegressor(n_estimators=100, random_state=42)          
RF_reg.fit(x,y)                                                # the best fit line is drawn


# Expenditure percentage estimation of the country with "GDP" value of 1000:

# In[304]:


print(RF_reg.predict([[1000]]))            


# In[305]:


x_array = np.arange(min(x),max(x),0.01).reshape(-1,1)   # line information to be drawn as a predict
y_head = RF_reg.predict(x_array)                        # percentage of spend predict

plt.scatter(x,y, color="red")
plt.plot(x_array,y_head,color="blue")
plt.xlabel("GDP")
plt.ylabel("percentage expenditure")
plt.show()


# ### Result: This result good but not so good as BEFORE.

# # Logistic Regression Model
# 
# * The aim is to reveal the class that will occur when a set of x values that have not yet been observed, to predict a classifier.
# * For the classification problem, to establish a linear model that defines the relationship between dependent and independent variables.
# * Regarding whether the dependent variable is 1 or 0 or yes or no status
# 
# 
# ** In this data, we will examine the states of Developed countries (Developed) = 0 and Developing = 1. I want to find the level of development I want, so close to 1!

# When we look at the country column data, it consists of objects, let's drop it. Because we need int or float values.

# In[306]:


logdata.drop(["Country"], axis=1, inplace=True)  
logdata.head()


# Our variable class, which is 1 to 0, let's examine this.

# In[307]:


logdata["Status"].value_counts()


# Let's continue with the review.

# In[308]:


logdata["Status"].value_counts().plot.barh();


# We need to create binary, that is, from 0 to 1. Let's do the necessary transformations.

# In[309]:


logdata.Status = [1 if each == "Developing" else 0 for each in logdata.Status]   


# Let's look at their general statistical properties.

# In[310]:


logdata.describe().T


# Let's create our variables now.

# In[311]:


y = logdata["Status"]
X_data = logdata.drop(["Status"], axis=1)


# Let's do normalization in our data.

# In[312]:


#*** Normalize ***#

X = (X_data - np.min(X_data))/(np.max(X_data) - np.min(X_data))


# Let's build a model through statsmodels and make it fit. Here, the meaning of the model and how much of this variable affects us, comes from this table.

# In[313]:


loj = sm.Logit(y, X)
loj_model= loj.fit()
loj_model.summary()


# Then see model:

# In[314]:


from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
loj_model


# In[315]:


# constant value
loj_model.intercept_


# Coefficient values of all independent variables:

# In[316]:


loj_model.coef_


# # PREDICT and MODEL TUNNING

# In[317]:


y_pred = loj_model.predict(X)              # predict


# Confusion Matrix: Those that are 1 (PP) when it is 1 in reality, 0 (PN) when it is 1, 1 (NP) when it is 0 when it is 0 (NN) when it is 0.

# In[318]:


confusion_matrix(y, y_pred)


# See accuracy score:

# In[319]:


accuracy_score(y, y_pred)


# One of the outputs that will evaluate the results of a most detailed classification algorithm.

# In[320]:


print(classification_report(y, y_pred))


# See top 10 model predict values:

# In[321]:


loj_model.predict(X)[0:10]


# * Using the 'predict_proba' module if we want to give the noble values rather than the values given above 1 and 0.
# 
# 
# * Returns the values of 0 in the index or left side of 0, and values of 1 in the index 1 or of the right side of the matrix.

# In[322]:


loj_model.predict_proba(X)[0:10][:,0:2]                # Top 10


# Now let's try to model the ten prediction probability values above 'predict_proba'.

# In[323]:


y_probs = loj_model.predict_proba(X)
y_probs = y_probs[:,1]


# In[324]:


y_probs[0:10]               # top 10


# Put our guess values here in the loop and give it 1 to 0.5 and 0 to the little ones.

# In[325]:


y_pred = [1 if i > 0.5 else 0 for i in y_probs]


# When we look at the value above, we notice the change. Our purpose to do this is to verify our model.

# In[326]:


y_pred[0:10]


# In[327]:


confusion_matrix(y, y_pred)


# In[328]:


accuracy_score(y, y_pred)


# In[329]:


print(classification_report(y, y_pred))


# Let's do one more look at the top 5 elements we did above.

# In[330]:


loj_model.predict_proba(X)[:,1][0:5]


# In[331]:


logit_roc_auc = roc_auc_score(y, loj_model.predict(X))


# In[332]:


fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()


# Here, 
# 
# - blueline: The graphic of our success regarding the model we have established.
# - redline: If we don't do anything, our model will be this way. 

# In[333]:


# test train is subjected to separation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# # Let's create and fit our model.

# In[334]:


loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)
loj_model



# Let's see accuracy score:

# In[335]:


accuracy_score(y_test, loj_model.predict(X_test))


# Finally Tunned model score:

# In[336]:


cross_val_score(loj_model, X_test, y_test, cv = 10).mean()


# ### Result: From this data, we understand: 89% of the countries that are developing are developing countries, and the effects of the variables that will question life expectancies can be examined.

# # Conclusion
# We examined the **Life Expectancy (WHO)** data set with the basic models in Machine Learing and made some comments.
# 
# Note:
# 
#    - After this notebook, my aim is to prepare 'kernel' which is 'not clear' data set.
# 
#    - If you have any suggestions, please could you write for me? I wil be happy for comment and critics!
# 
#    - Thank you for your suggestion and votes ;)
# 

# In[ ]:





# In[ ]:




