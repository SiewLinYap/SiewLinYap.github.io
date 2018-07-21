


## Exploratory Data Analysis (EDA) on SAT scores

---

Exploratory data analysis is an important and essential part of any data science project. It helps to prepare, process and visualize the data for better preliminary judgement. A high quality and comprehensive EDA eases the modeling and lead to a more reliable and conclusive outcomes.

In this aspect, a set of SAT scores by state was used to practise various EDA techniques and attempt to use different plots to assist the understanding of the dataset and visualize them to gain insights and draw conclusions.



#### Package imports


```python
import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.rcParams["patch.force_edgecolor"] = True
sns.set(style='darkgrid')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```



## Part 1 : Data Loading Using csv Module vs Using Pandas

---

### 1.1 Load the file with the `csv` module and put it in a Python dictionary





```python
sat = './sat_scores.csv'
```


```python
with open(sat,'rU') as f:
    
    content = [line for line in f]
    keys = content[0].replace('\n','').split(',')

    temp_values = content[1:]
    k1_values = []
    k2_values = []
    k3_values = []
    k4_values = []
    for v_temp in temp_values:
        v_temp = v_temp.replace('\n','').split(',')
        k1_values.append(v_temp[0])
        k2_values.append(v_temp[1])
        k3_values.append(v_temp[2])
        k4_values.append(v_temp[3])


```

    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/ipykernel_launcher.py:1: DeprecationWarning: 'U' mode is deprecated
      """Entry point for launching an IPython kernel.



```python
sat_dict = dict(zip(keys, [k1_values,k2_values,k3_values,k4_values]))
```


```python
print(sat_dict)
```

    {'State': ['CT', 'NJ', 'MA', 'NY', 'NH', 'RI', 'PA', 'VT', 'ME', 'VA', 'DE', 'MD', 'NC', 'GA', 'IN', 'SC', 'DC', 'OR', 'FL', 'WA', 'TX', 'HI', 'AK', 'CA', 'AZ', 'NV', 'CO', 'OH', 'MT', 'WV', 'ID', 'TN', 'NM', 'IL', 'KY', 'WY', 'MI', 'MN', 'KS', 'AL', 'NE', 'OK', 'MO', 'LA', 'WI', 'AR', 'UT', 'IA', 'SD', 'ND', 'MS', 'All'], 'Rate': ['82', '81', '79', '77', '72', '71', '71', '69', '69', '68', '67', '65', '65', '63', '60', '57', '56', '55', '54', '53', '53', '52', '51', '51', '34', '33', '31', '26', '23', '18', '17', '13', '13', '12', '12', '11', '11', '9', '9', '9', '8', '8', '8', '7', '6', '6', '5', '5', '4', '4', '4', '45'], 'Verbal': ['509', '499', '511', '495', '520', '501', '500', '511', '506', '510', '501', '508', '493', '491', '499', '486', '482', '526', '498', '527', '493', '485', '514', '498', '523', '509', '539', '534', '539', '527', '543', '562', '551', '576', '550', '547', '561', '580', '577', '559', '562', '567', '577', '564', '584', '562', '575', '593', '577', '592', '566', '506'], 'Math': ['510', '513', '515', '505', '516', '499', '499', '506', '500', '501', '499', '510', '499', '489', '501', '488', '474', '526', '499', '527', '499', '515', '510', '517', '525', '515', '542', '439', '539', '512', '542', '553', '542', '589', '550', '545', '572', '589', '580', '554', '568', '561', '577', '562', '596', '550', '570', '603', '582', '599', '551', '514']}


### 1.2 Make a pandas DataFrame object with the SAT dictionary, and another with the pandas `.read_csv()` function



```python
# set up dataframe with raw data from dict derived earlier

df_sat_dict = pd.DataFrame(sat_dict)
df_sat_dict.head()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sat_dict.tail()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
    <tr>
      <th>51</th>
      <td>All</td>
      <td>45</td>
      <td>506</td>
      <td>514</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Last row is the sum of all rows above, need to remove this from analysis to avoid data being skewed
df_sat_dict.drop(index=51, inplace=True)
```


```python
df_sat_dict.tail()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sat_dict.dtypes 
# data type for all variables were found as object types (including those that supposed to be numeric)
```




    State     object
    Rate      object
    Verbal    object
    Math      object
    dtype: object




```python
# set up dataframe with raw data loaded directly using read_csv()

df_sat = pd.read_csv(sat)
df_sat.head()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sat.tail()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
    <tr>
      <th>51</th>
      <td>All</td>
      <td>45</td>
      <td>506</td>
      <td>514</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sat.drop(index=51, axis=0, inplace=True)
df_sat.tail()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sat.dtypes 
# numbers were found transformed as integer data type right away when using read_csv()
```




    State     object
    Rate       int64
    Verbal     int64
    Math       int64
    dtype: object



> #### Observation on the difference :
> - If we did not convert the string column values to float in dictionary, the columns in the DataFrame are of type `object` (which are string values, essentially). 



## Part 2 : Create a "Data Dictionary"

---

A data dictionary is an object that describes the data. It helps users to understand the data better and thus ease the EDA process. The data dictionary in this exercise is framed up to include the name of each variable (column), the type of the variable, description of what the variable is, and the shape (rows and columns) of the entire dataset.


```python
data_dict = pd.DataFrame(columns=['Column_name', 'Data_type', 'Description', 'No_of_records'])
```


```python
data_dict['Column_name'] = df_sat.columns
```


```python
data_dict['Data_type'] = list(df_sat.dtypes)
```


```python
data_dict['Description'] = ['US State of SAT scores', 'Rate of students that take the SAT in that state', 
                            'Mean verbal score in that state', 'Mean math score in that state']
```


```python
df_sat.info() # since all columns are with same no. of records, can use .shape[0] to get this info
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 51 entries, 0 to 50
    Data columns (total 4 columns):
    State     51 non-null object
    Rate      51 non-null int64
    Verbal    51 non-null int64
    Math      51 non-null int64
    dtypes: int64(3), object(1)
    memory usage: 2.0+ KB



```python
data_dict['No_of_records'] = [df_sat.shape[0] for v in data_dict['Column_name']]
```


```python
data_dict
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
      <th>Column_name</th>
      <th>Data_type</th>
      <th>Description</th>
      <th>No_of_records</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>State</td>
      <td>object</td>
      <td>US State of SAT scores</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rate</td>
      <td>int64</td>
      <td>Rate of students that take the SAT in that state</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Verbal</td>
      <td>int64</td>
      <td>Mean verbal score in that state</td>
      <td>51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Math</td>
      <td>int64</td>
      <td>Mean math score in that state</td>
      <td>51</td>
    </tr>
  </tbody>
</table>
</div>





## Part 3 : Data Analysis and Visualization

---

### 3.1 Check the distribution of numerical columns 

Data was examined to see if it was normally distributed




```python
fig, ax = plt.subplots(3,1, figsize=(14,12))
sns.distplot(df_sat['Rate'], bins=30, kde=False, ax=ax[0])
sns.distplot(df_sat['Verbal'], bins=30, kde=False, color='g', ax=ax[1])
sns.distplot(df_sat['Math'], bins=30, kde=False, color='r', ax=ax[2])
```

    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    <matplotlib.axes._subplots.AxesSubplot at 0x110a3ab70>




![png](output_27_2.png)


### 3.2 Get an overview of the correlation of each variable in the dataset



```python
sns.pairplot(df_sat, hue='State', size=3, aspect=1)
```




    <seaborn.axisgrid.PairGrid at 0x110b516a0>




![png](output_29_1.png)


***
> #### Observation on Pairplot :
> - Rate data is least normalized as compared to Verbal and Math data
> - Positive correlation observed for Math and Verbal
> - Negative correlation observed for Math & Rate, Verbal & Rate
> - SAT performance varied for various states

***

 



### 3.3 : Comparison of Math and Verbal Scores

---



```python
df_sat[['Math', 'Verbal']].plot.hist(bins=30, alpha=0.5, stacked=True, figsize=(14,5))
plt.title('Histogram of SAT Performance')
plt.xlabel('Score')
```




    Text(0.5,0,'Score')




![png](output_33_1.png)



```python
fig, ax = plt.subplots(2,3, figsize=(25,6))

sns.boxplot(data=df_sat['Rate'],orient='h', ax=ax[0][0])
sns.distplot(df_sat['Rate'], ax=ax[1][0],bins=30)

sns.boxplot(data=df_sat['Verbal'], orient='h', ax=ax[0][1], color='orange')
sns.distplot(df_sat['Verbal'], bins=30, ax=ax[1][1], color='orange')

sns.boxplot(data=df_sat['Math'], orient='h', ax=ax[0][2], color='purple')
sns.distplot(df_sat['Math'], bins=30, ax=ax[1][2], color='purple')
```

    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    <matplotlib.axes._subplots.AxesSubplot at 0x1154ffdd8>




![png](output_34_2.png)


> #### Observation :
> - Though SAT dataset doesn't very well distributed, there's no outlier point observed for all the 3 keys numerical variables
> - Outlier is defined as data points that are beyond 1.5 of interquartile range :
    * 1.5 below the 1st quartile
    * 1.5 above the 3rd quartile


```python
fig = plt.figure(figsize=(14,5))
ax = fig.gca()

ax = sns.boxplot(data=df_sat[['Rate','Verbal','Math']], orient='h', palette='Set2')
ax = plt.xlabel('Score')
ax = plt.title('Boxplot of SAT Performance')
ax = plt.text(1,1,"Shouldn't plot Rate together as it has different score scale", fontsize=14)
```


![png](output_36_0.png)


***
> #### Benefits of using boxplot vs scatterplot/histogram:
> - good way to summarize large amounts of data
> - provide information about the range and distribution of data set inclusive details such as :
    * minimum
    * 1st quartile
    * median
    * 3rd quartile
    * maximum 
    * outliers
> - give some indication of the data's symetry and skewness


***

> However, for boxploat above, Rate doesn't have the same scale as Verbal and Math, plotting the raw data of Rate within one chart with others makes the comparison not so relevant. To have the comparision more relevant, the data needs to be standardize on the same scale. It can be done through the standardization method below by factoring in the mean and standard deviation.



```python
# standardize variables

df_sat_std = (df_sat[['Rate','Verbal','Math']] - df_sat.mean()) / df_sat.std()
```


```python
fig = plt.figure(figsize=(14,5))
ax = fig.gca()

ax = sns.boxplot(data=df_sat_std, orient='h', palette='Set2')
ax = plt.title('Standardized Boxplot for SAT Performance')
```


![png](output_40_0.png)




### 3.4 : Analysis on Subset of Data

---
#### 3.4.a) States with Verbal scores greater than average Verbal scores across states



```python
df_sat['Verbal'].mean()
```




    532.5294117647059




```python
df_sat_verbal_abv_average = df_sat[df_sat['Verbal']>df_sat['Verbal'].mean()].sort_values(by='Verbal', ascending=False)
df_sat_verbal_abv_average
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>6</td>
      <td>584</td>
      <td>596</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>9</td>
      <td>580</td>
      <td>589</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>9</td>
      <td>577</td>
      <td>580</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>8</td>
      <td>577</td>
      <td>577</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>12</td>
      <td>576</td>
      <td>589</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>7</td>
      <td>564</td>
      <td>562</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>8</td>
      <td>562</td>
      <td>568</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>11</td>
      <td>561</td>
      <td>572</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>9</td>
      <td>559</td>
      <td>554</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>12</td>
      <td>550</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>11</td>
      <td>547</td>
      <td>545</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>17</td>
      <td>543</td>
      <td>542</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>23</td>
      <td>539</td>
      <td>539</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>31</td>
      <td>539</td>
      <td>542</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
    </tr>
  </tbody>
</table>
</div>




```python
round(len(df_sat_verbal_abv_average) / df_sat['State'].nunique()*100,2)
```




    47.06



> - 47.06% of states in US having Verbal score above average. 
> - Less than half of the states in US achieving average score

#### 3.4.b) States with Verbal scores greater than median Verbal scores across states



```python
df_sat['Verbal'].median()
```




    527.0




```python
df_sat_verbal_abv_median = df_sat[df_sat['Verbal']>df_sat['Verbal'].median()].sort_values(by='Verbal', ascending=False)
df_sat_verbal_abv_median
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>6</td>
      <td>584</td>
      <td>596</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>9</td>
      <td>580</td>
      <td>589</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>9</td>
      <td>577</td>
      <td>580</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>8</td>
      <td>577</td>
      <td>577</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>12</td>
      <td>576</td>
      <td>589</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>7</td>
      <td>564</td>
      <td>562</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>8</td>
      <td>562</td>
      <td>568</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>11</td>
      <td>561</td>
      <td>572</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>9</td>
      <td>559</td>
      <td>554</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>12</td>
      <td>550</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>11</td>
      <td>547</td>
      <td>545</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>17</td>
      <td>543</td>
      <td>542</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>23</td>
      <td>539</td>
      <td>539</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>31</td>
      <td>539</td>
      <td>542</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
    </tr>
  </tbody>
</table>
</div>




```python
round(len(df_sat_verbal_abv_median)/df_sat['Verbal'].nunique()*100,2)
```




    61.54



> - 61.54% of states in US having Verbal score above the median score
> - This percentage is slightly higher ( = more states included ) because the median score is slightly lower than the average score


```python
# visualize the difference of mean, median and distribution of Verval score

plt.figure(figsize=(15,6))
sns.distplot(df_sat['Verbal'], bins=30)
plt.axvline(x=df_sat['Verbal'].mean(), linewidth=2.5, linestyle='dashed', color='r', label='Average score')
plt.axvline(x=df_sat['Verbal'].median(), linewidth=2.5, linestyle='dashed', color='g', label='Median score')
plt.legend()
plt.title('Distribution of Verbal Score')
```

    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    Text(0.5,1,'Distribution of Verbal Score')




![png](output_51_2.png)


#### 3.4.c) Difference between the Verbal and Math scores 



```python
# set up new column to reflect the difference of Verbal vs Math score

df_sat['Difference_Verbal_Math'] = df_sat['Verbal'] - df_sat['Math']
df_sat.head()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Difference_Verbal_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
      <td>-14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
      <td>-10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 3 states with greatest difference between Verbal and Math

df_sat_verbal_greaterThanMath = df_sat.sort_values(by='Difference_Verbal_Math', ascending=False).head(10)
df_sat_verbal_greaterThanMath.head(3)
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Difference_Verbal_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
      <td>95</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
      <td>15</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WV</td>
      <td>18</td>
      <td>527</td>
      <td>512</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 3 states with greatest difference between Math and Verbal

df_sat_math_greaterThanVerbal = df_sat.sort_values(by='Difference_Verbal_Math', ascending=True).head(10)
df_sat_math_greaterThanVerbal.head(3)
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Difference_Verbal_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>HI</td>
      <td>52</td>
      <td>485</td>
      <td>515</td>
      <td>-30</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CA</td>
      <td>51</td>
      <td>498</td>
      <td>517</td>
      <td>-19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
      <td>-14</td>
    </tr>
  </tbody>
</table>
</div>



### 3.5. Examine Summary Statistics

---

Summary stats enables a quick overall on the data, checking for correlation, missing values and anomalies




```python
df_sat.drop(columns=['Difference_Verbal_Math']).describe(include='all')
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>51</td>
      <td>51.000000</td>
      <td>51.000000</td>
      <td>51.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>UT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>37.000000</td>
      <td>532.529412</td>
      <td>531.843137</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>27.550681</td>
      <td>33.360667</td>
      <td>36.287393</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>4.000000</td>
      <td>482.000000</td>
      <td>439.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>9.000000</td>
      <td>501.000000</td>
      <td>503.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>33.000000</td>
      <td>527.000000</td>
      <td>525.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>64.000000</td>
      <td>562.000000</td>
      <td>557.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>82.000000</td>
      <td>593.000000</td>
      <td>603.000000</td>
    </tr>
  </tbody>
</table>
</div>



Count => no. of records/population size
<br>
mean => average of the total population
<br>
std => standard deviation of the population
<br>
min => lowest value of the record
<br>
25% => value by 25% quartile of the population
<br>
50% => value by 50% quartile of the population
<br>
75% => value by 75% quartile of the population
<br>
max => highest value of the record

#### 3.5 a) Correlation with Pearson Coefficient


```python
# check correlation of numerical variables in dataset

df_sat_corr = df_sat.drop(columns=['State','Difference_Verbal_Math']).corr()
df_sat_corr
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
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rate</th>
      <td>1.000000</td>
      <td>-0.888121</td>
      <td>-0.773419</td>
    </tr>
    <tr>
      <th>Verbal</th>
      <td>-0.888121</td>
      <td>1.000000</td>
      <td>0.899909</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>-0.773419</td>
      <td>0.899909</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6))
sns.heatmap(df_sat_corr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1176d26d8>




![png](output_61_1.png)


> #### Observations:
> 1. Verbal and Math is positively correlated => If doing good in Verbal, also likely doing good in Math
> 2. Rate is negatively correlated to Verbal and Math => State with low SAT taking rate seems to get better score in Verbal / Math

#### 3.5 b) Correlation with Spearman Coefficient


```python
df_sat_spearman = df_sat.drop('Difference_Verbal_Math', axis=1).corr(method='spearman')
df_sat_spearman
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
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rate</th>
      <td>1.000000</td>
      <td>-0.836058</td>
      <td>-0.811662</td>
    </tr>
    <tr>
      <th>Verbal</th>
      <td>-0.836058</td>
      <td>1.000000</td>
      <td>0.909413</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>-0.811662</td>
      <td>0.909413</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



> #### Observations
> - spearman correlation coefficient was found higher than the pearson correlation coeffifient

> #### Process calculating the spearman rank correlation
> 1. Rank the values of each variable, with the largest value having rank 1.
> 2. Calculate the difference in ranks
> 3. Complete the rest of the calculation with the formula below:
$$ r = 1 - \frac {6 \sum d_i^2} {n^3 - n}$$

> where $d_i$ is the differences in paired ranks, $n$ is the number of rows


> #### Useful Notes :

> - The Spearman Rank correlation measures the rank of one variable against another.
> - A strong positive spearman rank correlation means that if one does well in math one will most likely also do well in verbal test.
> - A strong negative spearkman rank means that if one does well in one, one would most likely do poorly in the other. 
> - The Spearman Rank correlation is not as sensitive to outliers because it does not take into account how far away the outliers are from the average. 
> - It is only concerned about the order of those outliers versus everyone else. 
> - Outliers in this dataset were those states that did well in one subject but poorly in the other (the two biggest difference). But since these are ranked aka sorted according to how their scores are; doing very well in Math and then doing average in Verbal test will only result is a high rank difference which will then give us a negative spearman rank correlation (not zero linear pearson correlation).

#### 3.5 c) Covariance


```python
# check covariance of the dataset

df_sat_cov = df_sat.drop(columns=['Difference_Verbal_Math']).cov()
df_sat_cov
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
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rate</th>
      <td>759.04</td>
      <td>-816.280000</td>
      <td>-773.220000</td>
    </tr>
    <tr>
      <th>Verbal</th>
      <td>-816.28</td>
      <td>1112.934118</td>
      <td>1089.404706</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>-773.22</td>
      <td>1089.404706</td>
      <td>1316.774902</td>
    </tr>
  </tbody>
</table>
</div>



> #### Difference between covariance matrix and correlation matrix.
> - Covariance matrix is a measure of how related the variables are to each other, but it is measured on the variance of the variables. 
> - While correlation matrix is dimensionless, regardless of what the units are in the variables, it always return the same in between -1 and 1

> #### Process to convert the covariance into the correlation
> - Conversion can be done using the following formula:
> $$ corr(X, Y) = \frac{cov(X, Y)}{std(X)std(Y)} $$

> #### Reason correlation matrix is preferred over covariance matrix for examining relationships in data
> - Covariance is difficult to compared due to different dimension used whilst correlation is a scaled & standardized version of covariance in which the values are assured to be -1 and 1

#### 3.5 d) Percentile Scoring
Examine percentile scoring of data. In other words, the conversion of numeric data to their equivalent percentile scores.


```python
df_sat_percentile = df_sat.copy()
df_sat_percentile['Rate_percentile'] = df_sat_percentile['Rate'].apply(lambda x: stats.percentileofscore(df_sat_percentile['Rate'],x))
df_sat_percentile.head()
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
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
      <th>Difference_Verbal_Math</th>
      <th>Rate_percentile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
      <td>-1</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
      <td>-14</td>
      <td>98.039216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
      <td>-4</td>
      <td>96.078431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
      <td>-10</td>
      <td>94.117647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
      <td>4</td>
      <td>92.156863</td>
    </tr>
  </tbody>
</table>
</div>



> - We could also possibly rank the values in each variable by percentile when calculating using spearman method
> - Percentile scoring and the spearman rank correlation are related in the sense that they both take a value that is calculated by how they are ranked in relation to one another in some sorted order, and **not** based on the difference from the mean.
> - Percentile scoring ranks each value against a sorted order to find in which percentile do the value fall in. For example, take New York state. A 94.12 percentile score means that the state's participantion rate of 77 is in the top 94.12% of the entire dataset. That there were 3 other states that had a higher participation rate than it ( (100% - 94.12%) x 52 = approx. 3 ). 

#### 3.5 e) Percentiles vs outliers


```python
# 9.3.2 Distribution as per normal values

fig, ax = plt.subplots(1,2, sharey = True, figsize=(20,6))

sns.distplot(df_sat_percentile['Rate'], bins=30, ax=ax[0]) # Distribution as per normal values
ax[0].set_title('Distribution of Rate')
sns.distplot(df_sat_percentile['Rate_percentile'], bins=30, ax=ax[1], color='g')
ax[1].set_title('Distribution of Rate_in_Percentile')

plt.suptitle('Comparison of Distribution for Rate_Normal vs Rate_In_Percentile', size=14)
plt.show()

```

    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /Users/yapsiewlin/anaconda3/envs/py36/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](output_74_1.png)



```python
fig, ax = plt.subplots(2,1, figsize=(10,8))

sns.boxplot(data=df_sat_percentile[['Rate', 'Rate_percentile']], ax=ax[0])
sns.violinplot(data=df_sat_percentile[['Rate', 'Rate_percentile']], ax=ax[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1188a3828>




![png](output_75_1.png)


> Observations :
> - Percentile is not as sensitive to outliers because it does not take into account how far away the outliers are from the average.
> - Rate was observed more normally distributed using percentile 


## Key Learnings

---
There are various ways and techniques that we can use to analyze data. From loading data, data preparation, data analysis and data visualization, there are many options that we can use to achieve the similar purpose. Different techniques have their own strength in particular features and performance. Therefore, it is essential to evaluate which one serves the best in term of clarity and provides the best visualization that enables a better and simplified explanation to a complex observation. 

Sometimes, we can combine multiple approaches / different plots to complement the unique feature that demonstrates by one type of chart but not the other. Through this combination, it can help us to explain, to 're-imagine' and understand the dataset better, and assist us to discover patterns in different perspective to make our analysis more comprehensive.



