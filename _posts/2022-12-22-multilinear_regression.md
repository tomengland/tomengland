---
title: Multilinear_Regression
layout: post
image: /assets/images/ds.jpg
source:
tags:
  - data-science
  - machine-learning
  - project
---

# Multilinear Regression

* It's rare that one input explains the output
* We often need more predictors to improve models. 
* Be aware of multicolinearity 

### Maybe Color and Shape can impact the price of a diamond. 


```python
# Libraries

import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns

%cd /Users/thomasengland/Dev/Python Analytics/Statistics and Descriptive Analytics/Multilinear Regression
```

    /Users/thomasengland/Dev/Python Analytics/Statistics and Descriptive Analytics/Multilinear Regression



```python
# Load the Data
df = pd.read_csv("salaries.csv")
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>discipline</th>
      <th>yrs.since.phd</th>
      <th>yrs.service</th>
      <th>sex</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Prof</td>
      <td>B</td>
      <td>19</td>
      <td>18</td>
      <td>Male</td>
      <td>139750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Prof</td>
      <td>B</td>
      <td>20</td>
      <td>16</td>
      <td>Male</td>
      <td>173200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AsstProf</td>
      <td>B</td>
      <td>4</td>
      <td>3</td>
      <td>Male</td>
      <td>79750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Prof</td>
      <td>B</td>
      <td>45</td>
      <td>39</td>
      <td>Male</td>
      <td>115000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Prof</td>
      <td>B</td>
      <td>40</td>
      <td>41</td>
      <td>Male</td>
      <td>141500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Analyze the data
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yrs.since.phd</th>
      <th>yrs.service</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>397.000000</td>
      <td>397.000000</td>
      <td>397.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.314861</td>
      <td>17.614610</td>
      <td>113706.458438</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.887003</td>
      <td>13.006024</td>
      <td>30289.038695</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>57800.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>91000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.000000</td>
      <td>16.000000</td>
      <td>107300.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>32.000000</td>
      <td>27.000000</td>
      <td>134185.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>56.000000</td>
      <td>60.000000</td>
      <td>231545.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df["yrs.service"].hist())
print(df["yrs.since.phd"].hist())
```

    AxesSubplot(0.125,0.11;0.775x0.77)
    AxesSubplot(0.125,0.11;0.775x0.77)



    
![png](/assets/images/multilinear_regression_files/multilinear_regression_5_1.png)
    



```python
# plotting continuous independent variables against dependent variable
sns.set(font_scale = 2)
sns.pairplot(data=df, y_vars=["salary"], x_vars=["yrs.service", "yrs.since.phd"], height = 7)
```




    <seaborn.axisgrid.PairGrid at 0x17d485160>




    
![png](/assets/images/multilinear_regression_files/multilinear_regression_6_1.png)
    



```python
sns.heatmap(df.corr(), annot=True, fmt=".2g", center=0, cmap="coolwarm", linewidths=1, linecolor="black")
```




    <AxesSubplot:>




    
![png](/assets/images/multilinear_regression_files/multilinear_regression_7_1.png)
    



```python
#  Tells me that I should probably just use yrs.since.phd since that's the closest correlated to salary. 

# Categorical Variables 

df["rank"].value_counts() 
# Let's collect all categorical variables
df.select_dtypes(include="object").value_counts()
```




    rank       discipline  sex   
    Prof       B           Male      125
               A           Male      123
    AsstProf   B           Male       38
    AssocProf  B           Male       32
               A           Male       22
    AsstProf   A           Male       18
    Prof       B           Female     10
               A           Female      8
    AssocProf  B           Female      6
    AsstProf   A           Female      6
               B           Female      5
    AssocProf  A           Female      4
    dtype: int64



# Loops


```python
# For Loop categorical variables
categorical = list(df.select_dtypes(include="object"))
for cat in categorical:
    print(df[cat].value_counts())

```

    Prof         266
    AsstProf      67
    AssocProf     64
    Name: rank, dtype: int64
    B    216
    A    181
    Name: discipline, dtype: int64
    Male      358
    Female     39
    Name: sex, dtype: int64



```python
# look at data set
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>discipline</th>
      <th>yrs.since.phd</th>
      <th>yrs.service</th>
      <th>sex</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Prof</td>
      <td>B</td>
      <td>19</td>
      <td>18</td>
      <td>Male</td>
      <td>139750</td>
    </tr>
  </tbody>
</table>
</div>




```python
# transform objects into dummies
df = pd.get_dummies(data=df, drop_first=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yrs.since.phd</th>
      <th>yrs.service</th>
      <th>salary</th>
      <th>rank_AsstProf</th>
      <th>rank_Prof</th>
      <th>discipline_B</th>
      <th>sex_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>18</td>
      <td>139750</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>16</td>
      <td>173200</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>3</td>
      <td>79750</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>39</td>
      <td>115000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>41</td>
      <td>141500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# isolate X and y
y = df.salary / 1000
X = df.drop(columns=["salary", "yrs.service"])
X.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yrs.since.phd</th>
      <th>rank_AsstProf</th>
      <th>rank_Prof</th>
      <th>discipline_B</th>
      <th>sex_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add a constant
X = sm.add_constant(X)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>yrs.since.phd</th>
      <th>rank_AsstProf</th>
      <th>rank_Prof</th>
      <th>discipline_B</th>
      <th>sex_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>392</th>
      <td>1.0</td>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>393</th>
      <td>1.0</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>394</th>
      <td>1.0</td>
      <td>42</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>395</th>
      <td>1.0</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>1.0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>397 rows × 6 columns</p>
</div>




```python
# Training and Test Set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1502)

X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>yrs.since.phd</th>
      <th>rank_AsstProf</th>
      <th>rank_Prof</th>
      <th>discipline_B</th>
      <th>sex_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>260</th>
      <td>1.0</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>253</th>
      <td>1.0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>256</th>
      <td>1.0</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>267</th>
      <td>1.0</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>235</th>
      <td>1.0</td>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Multilinear Regression

model = sm.OLS(y_train, X_train).fit()
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 salary   R-squared:                       0.483
    Model:                            OLS   Adj. R-squared:                  0.474
    Method:                 Least Squares   F-statistic:                     54.33
    Date:                Wed, 21 Dec 2022   Prob (F-statistic):           9.82e-40
    Time:                        15:49:53   Log-Likelihood:                -1332.1
    No. Observations:                 297   AIC:                             2676.
    Df Residuals:                     291   BIC:                             2698.
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const            79.1284      5.569     14.210      0.000      68.169      90.088
    yrs.since.phd    -0.0051      0.137     -0.037      0.970      -0.275       0.265
    rank_AsstProf   -13.4152      4.718     -2.843      0.005     -22.701      -4.129
    rank_Prof        34.9533      3.964      8.819      0.000      27.152      42.754
    discipline_B     15.3690      2.630      5.843      0.000      10.192      20.546
    sex_Male          5.7962      4.380      1.323      0.187      -2.825      14.417
    ==============================================================================
    Omnibus:                       40.745   Durbin-Watson:                   2.146
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               71.723
    Skew:                           0.775   Prob(JB):                     2.66e-16
    Kurtosis:                       4.842   Cond. No.                         146.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python

```
