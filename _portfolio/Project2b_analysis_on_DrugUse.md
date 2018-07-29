---
layout: post
title: Analysis on Drug Usage
img: "assets/img/portfolio/Drug_logo3.png"
date: July, 29 2018

---

![image]({{ site.baseurl }}/{{ page.img }})


This is another exercise to practise EDA using various techniques. Dataset was obtained from https://www.github.com/fivethirtyeight/data/tree/master/drug-use-by-age. The challenge for this dataset is its size. It only has 17 rows but with 28 columns. Different thought process is required in order to process this type of data. 


#### Package imports


```python
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["patch.force_edgecolor"] = True
sns.set(style='darkgrid')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

## Part 1 : Data Loading and Cleaning

---

To check if there are any anomalies on the dataset, and if transformation is needed




```python
pd.set_option('max_columns',50)
```


```python
drug = pd.read_csv('./drug-use-by-age.csv')
print(drug.shape)
drug.head()
```

    (17, 28)





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
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.1</td>
      <td>35.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>1.6</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>2757</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>15.0</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.6</td>
      <td>6.0</td>
      <td>2.5</td>
      <td>12.0</td>
      <td>2.4</td>
      <td>14.0</td>
      <td>0.1</td>
      <td>41.0</td>
      <td>0.3</td>
      <td>25.5</td>
      <td>0.3</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2792</td>
      <td>18.1</td>
      <td>5.0</td>
      <td>8.7</td>
      <td>24.0</td>
      <td>0.1</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>1.6</td>
      <td>3.0</td>
      <td>2.6</td>
      <td>5.0</td>
      <td>3.9</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>5.0</td>
      <td>0.8</td>
      <td>12.0</td>
      <td>0.1</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2956</td>
      <td>29.2</td>
      <td>6.0</td>
      <td>14.5</td>
      <td>25.0</td>
      <td>0.5</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>9.5</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>2.1</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>5.5</td>
      <td>5.5</td>
      <td>10.0</td>
      <td>0.8</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>6.0</td>
      <td>0.3</td>
      <td>10.5</td>
      <td>0.4</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>3058</td>
      <td>40.1</td>
      <td>10.0</td>
      <td>22.5</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.1</td>
      <td>66.5</td>
      <td>3.4</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.2</td>
      <td>7.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>2.4</td>
      <td>11.0</td>
      <td>1.8</td>
      <td>9.5</td>
      <td>0.3</td>
      <td>36.0</td>
      <td>0.2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



> - This is a 'short & fat' dataset consists of only 17 rows but with 28 columns. 
> - cleaning is required
> - missing values are observed for some columns


```python
drug.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17 entries, 0 to 16
    Data columns (total 28 columns):
    age                        17 non-null object
    n                          17 non-null int64
    alcohol-use                17 non-null float64
    alcohol-frequency          17 non-null float64
    marijuana-use              17 non-null float64
    marijuana-frequency        17 non-null float64
    cocaine-use                17 non-null float64
    cocaine-frequency          17 non-null object
    crack-use                  17 non-null float64
    crack-frequency            17 non-null object
    heroin-use                 17 non-null float64
    heroin-frequency           17 non-null object
    hallucinogen-use           17 non-null float64
    hallucinogen-frequency     17 non-null float64
    inhalant-use               17 non-null float64
    inhalant-frequency         17 non-null object
    pain-releiver-use          17 non-null float64
    pain-releiver-frequency    17 non-null float64
    oxycontin-use              17 non-null float64
    oxycontin-frequency        17 non-null object
    tranquilizer-use           17 non-null float64
    tranquilizer-frequency     17 non-null float64
    stimulant-use              17 non-null float64
    stimulant-frequency        17 non-null float64
    meth-use                   17 non-null float64
    meth-frequency             17 non-null object
    sedative-use               17 non-null float64
    sedative-frequency         17 non-null float64
    dtypes: float64(20), int64(1), object(7)
    memory usage: 3.8+ KB


> - some numerical values are in wrong data types (object -> float)

Columns with data type=object are examined further in order to understand why it is turned to object data type whilst it actually contains numeric values


```python
print('Age : {}'.format(drug['age'].unique()))
```

    Age : ['12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22-23' '24-25' '26-29'
     '30-34' '35-49' '50-64' '65+']


> - some values in Age column are in discrete values whilst some in range (but inconsistent range interval), one is found having special character (+)


```python
print('cocaine-frequency : {}'.format(drug['cocaine-frequency'].unique()))
print('crack-frequency : {}'.format(drug['crack-frequency'].unique()))
print('heroin-frequency : {}'.format(drug['heroin-frequency'].unique()))
print('inhalant-frequency : {}'.format(drug['inhalant-frequency'].unique()))
print('oxycontin-frequency : {}'.format(drug['oxycontin-frequency'].unique()))
print('meth-frequency: {}'.format(drug['meth-frequency'].unique()))
```

    cocaine-frequency : ['5.0' '1.0' '5.5' '4.0' '7.0' '8.0' '6.0' '15.0' '36.0' '-']
    crack-frequency : ['-' '3.0' '9.5' '1.0' '21.0' '10.0' '2.0' '5.0' '17.0' '6.0' '15.0'
     '48.0' '62.0']
    heroin-frequency : ['35.5' '-' '2.0' '1.0' '66.5' '64.0' '46.0' '180.0' '45.0' '30.0' '57.5'
     '88.0' '50.0' '66.0' '280.0' '41.0' '120.0']
    inhalant-frequency : ['19.0' '12.0' '5.0' '5.5' '3.0' '4.0' '2.0' '3.5' '10.0' '13.5' '-']
    oxycontin-frequency : ['24.5' '41.0' '4.5' '3.0' '4.0' '6.0' '7.0' '7.5' '12.0' '13.5' '17.5'
     '20.0' '46.0' '5.0' '-']
    meth-frequency: ['-' '5.0' '24.0' '10.5' '36.0' '48.0' '12.0' '105.0' '2.0' '46.0' '21.0'
     '30.0' '54.0' '104.0']


> - some missing values are observed with '-'

Extraction of records with '-'


```python
drug[drug.values =='-']
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
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.1</td>
      <td>35.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>1.6</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.1</td>
      <td>35.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>1.6</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>2757</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>15.0</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.6</td>
      <td>6.0</td>
      <td>2.5</td>
      <td>12.0</td>
      <td>2.4</td>
      <td>14.0</td>
      <td>0.1</td>
      <td>41.0</td>
      <td>0.3</td>
      <td>25.5</td>
      <td>0.3</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2792</td>
      <td>18.1</td>
      <td>5.0</td>
      <td>8.7</td>
      <td>24.0</td>
      <td>0.1</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>1.6</td>
      <td>3.0</td>
      <td>2.6</td>
      <td>5.0</td>
      <td>3.9</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>5.0</td>
      <td>0.8</td>
      <td>12.0</td>
      <td>0.1</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65+</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>364.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65+</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>364.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65+</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>364.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65+</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>364.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65+</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>364.0</td>
      <td>0.0</td>
      <td>-</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>



Replacement of cells with '-' to NA values


```python
drug.replace('-', np.nan, inplace=True)
```


```python
drug.head()
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
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>35.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>1.6</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>2757</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>15.0</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.6</td>
      <td>6.0</td>
      <td>2.5</td>
      <td>12.0</td>
      <td>2.4</td>
      <td>14.0</td>
      <td>0.1</td>
      <td>41.0</td>
      <td>0.3</td>
      <td>25.5</td>
      <td>0.3</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2792</td>
      <td>18.1</td>
      <td>5.0</td>
      <td>8.7</td>
      <td>24.0</td>
      <td>0.1</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>1.6</td>
      <td>3.0</td>
      <td>2.6</td>
      <td>5.0</td>
      <td>3.9</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>5.0</td>
      <td>0.8</td>
      <td>12.0</td>
      <td>0.1</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2956</td>
      <td>29.2</td>
      <td>6.0</td>
      <td>14.5</td>
      <td>25.0</td>
      <td>0.5</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>9.5</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>2.1</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>5.5</td>
      <td>5.5</td>
      <td>10.0</td>
      <td>0.8</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>6.0</td>
      <td>0.3</td>
      <td>10.5</td>
      <td>0.4</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>3058</td>
      <td>40.1</td>
      <td>10.0</td>
      <td>22.5</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.1</td>
      <td>66.5</td>
      <td>3.4</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.2</td>
      <td>7.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>2.4</td>
      <td>11.0</td>
      <td>1.8</td>
      <td>9.5</td>
      <td>0.3</td>
      <td>36.0</td>
      <td>0.2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Examination of missing values and data type for each column


```python
drug.isnull().sum()
```




    age                        0
    n                          0
    alcohol-use                0
    alcohol-frequency          0
    marijuana-use              0
    marijuana-frequency        0
    cocaine-use                0
    cocaine-frequency          1
    crack-use                  0
    crack-frequency            3
    heroin-use                 0
    heroin-frequency           1
    hallucinogen-use           0
    hallucinogen-frequency     0
    inhalant-use               0
    inhalant-frequency         1
    pain-releiver-use          0
    pain-releiver-frequency    0
    oxycontin-use              0
    oxycontin-frequency        1
    tranquilizer-use           0
    tranquilizer-frequency     0
    stimulant-use              0
    stimulant-frequency        0
    meth-use                   0
    meth-frequency             2
    sedative-use               0
    sedative-frequency         0
    dtype: int64




```python
drug.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17 entries, 0 to 16
    Data columns (total 28 columns):
    age                        17 non-null object
    n                          17 non-null int64
    alcohol-use                17 non-null float64
    alcohol-frequency          17 non-null float64
    marijuana-use              17 non-null float64
    marijuana-frequency        17 non-null float64
    cocaine-use                17 non-null float64
    cocaine-frequency          16 non-null object
    crack-use                  17 non-null float64
    crack-frequency            14 non-null object
    heroin-use                 17 non-null float64
    heroin-frequency           16 non-null object
    hallucinogen-use           17 non-null float64
    hallucinogen-frequency     17 non-null float64
    inhalant-use               17 non-null float64
    inhalant-frequency         16 non-null object
    pain-releiver-use          17 non-null float64
    pain-releiver-frequency    17 non-null float64
    oxycontin-use              17 non-null float64
    oxycontin-frequency        16 non-null object
    tranquilizer-use           17 non-null float64
    tranquilizer-frequency     17 non-null float64
    stimulant-use              17 non-null float64
    stimulant-frequency        17 non-null float64
    meth-use                   17 non-null float64
    meth-frequency             15 non-null object
    sedative-use               17 non-null float64
    sedative-frequency         17 non-null float64
    dtypes: float64(20), int64(1), object(7)
    memory usage: 3.8+ KB


Adjustment of data type to numeric


```python
drug.iloc[:,2:] = drug.iloc[:,2:].astype(float)
drug.dtypes
```




    age                         object
    n                            int64
    alcohol-use                float64
    alcohol-frequency          float64
    marijuana-use              float64
    marijuana-frequency        float64
    cocaine-use                float64
    cocaine-frequency          float64
    crack-use                  float64
    crack-frequency            float64
    heroin-use                 float64
    heroin-frequency           float64
    hallucinogen-use           float64
    hallucinogen-frequency     float64
    inhalant-use               float64
    inhalant-frequency         float64
    pain-releiver-use          float64
    pain-releiver-frequency    float64
    oxycontin-use              float64
    oxycontin-frequency        float64
    tranquilizer-use           float64
    tranquilizer-frequency     float64
    stimulant-use              float64
    stimulant-frequency        float64
    meth-use                   float64
    meth-frequency             float64
    sedative-use               float64
    sedative-frequency         float64
    dtype: object




```python
drug.describe()
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
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>16.000000</td>
      <td>17.000000</td>
      <td>14.000000</td>
      <td>17.000000</td>
      <td>16.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>16.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>16.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3251.058824</td>
      <td>55.429412</td>
      <td>33.352941</td>
      <td>18.923529</td>
      <td>42.941176</td>
      <td>2.176471</td>
      <td>7.875000</td>
      <td>0.294118</td>
      <td>15.035714</td>
      <td>0.352941</td>
      <td>73.281250</td>
      <td>3.394118</td>
      <td>8.411765</td>
      <td>1.388235</td>
      <td>6.156250</td>
      <td>6.270588</td>
      <td>14.705882</td>
      <td>0.935294</td>
      <td>14.812500</td>
      <td>2.805882</td>
      <td>11.735294</td>
      <td>1.917647</td>
      <td>31.147059</td>
      <td>0.382353</td>
      <td>35.966667</td>
      <td>0.282353</td>
      <td>19.382353</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1297.890426</td>
      <td>26.878866</td>
      <td>21.318833</td>
      <td>11.959752</td>
      <td>18.362566</td>
      <td>1.816772</td>
      <td>8.038449</td>
      <td>0.235772</td>
      <td>18.111263</td>
      <td>0.333762</td>
      <td>70.090173</td>
      <td>2.792506</td>
      <td>15.000245</td>
      <td>0.927283</td>
      <td>4.860448</td>
      <td>3.166379</td>
      <td>6.935098</td>
      <td>0.608216</td>
      <td>12.798275</td>
      <td>1.753379</td>
      <td>11.485205</td>
      <td>1.407673</td>
      <td>85.973790</td>
      <td>0.262762</td>
      <td>31.974581</td>
      <td>0.138000</td>
      <td>24.833527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2223.000000</td>
      <td>3.900000</td>
      <td>3.000000</td>
      <td>1.100000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.600000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.200000</td>
      <td>4.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2469.000000</td>
      <td>40.100000</td>
      <td>10.000000</td>
      <td>8.700000</td>
      <td>30.000000</td>
      <td>0.500000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>0.100000</td>
      <td>39.625000</td>
      <td>0.600000</td>
      <td>3.000000</td>
      <td>0.600000</td>
      <td>3.375000</td>
      <td>3.900000</td>
      <td>12.000000</td>
      <td>0.400000</td>
      <td>5.750000</td>
      <td>1.400000</td>
      <td>6.000000</td>
      <td>0.600000</td>
      <td>7.000000</td>
      <td>0.200000</td>
      <td>12.000000</td>
      <td>0.200000</td>
      <td>6.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2798.000000</td>
      <td>64.600000</td>
      <td>48.000000</td>
      <td>20.800000</td>
      <td>52.000000</td>
      <td>2.000000</td>
      <td>5.250000</td>
      <td>0.400000</td>
      <td>7.750000</td>
      <td>0.200000</td>
      <td>53.750000</td>
      <td>3.200000</td>
      <td>3.000000</td>
      <td>1.400000</td>
      <td>4.000000</td>
      <td>6.200000</td>
      <td>12.000000</td>
      <td>1.100000</td>
      <td>12.000000</td>
      <td>3.500000</td>
      <td>10.000000</td>
      <td>1.800000</td>
      <td>10.000000</td>
      <td>0.400000</td>
      <td>30.000000</td>
      <td>0.300000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3058.000000</td>
      <td>77.500000</td>
      <td>52.000000</td>
      <td>28.400000</td>
      <td>52.000000</td>
      <td>4.000000</td>
      <td>7.250000</td>
      <td>0.500000</td>
      <td>16.500000</td>
      <td>0.600000</td>
      <td>71.875000</td>
      <td>5.200000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>6.625000</td>
      <td>9.000000</td>
      <td>15.000000</td>
      <td>1.400000</td>
      <td>18.125000</td>
      <td>4.200000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>0.600000</td>
      <td>47.000000</td>
      <td>0.400000</td>
      <td>17.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7391.000000</td>
      <td>84.200000</td>
      <td>52.000000</td>
      <td>34.000000</td>
      <td>72.000000</td>
      <td>4.900000</td>
      <td>36.000000</td>
      <td>0.600000</td>
      <td>62.000000</td>
      <td>1.100000</td>
      <td>280.000000</td>
      <td>8.600000</td>
      <td>52.000000</td>
      <td>3.000000</td>
      <td>19.000000</td>
      <td>10.000000</td>
      <td>36.000000</td>
      <td>1.700000</td>
      <td>46.000000</td>
      <td>5.400000</td>
      <td>52.000000</td>
      <td>4.100000</td>
      <td>364.000000</td>
      <td>0.900000</td>
      <td>105.000000</td>
      <td>0.500000</td>
      <td>104.000000</td>
    </tr>
  </tbody>
</table>
</div>



> - some columns show high standard deviation values

## Part 2 : High Level Overview of Data

---

#### 2.1 : Check age group vs sample size distribution


```python
drug.plot.bar(x='age', y='n', figsize=(15,6), color='grey')
plt.title('Distribution of Sample Size by Age')
plt.ylabel('sample size')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/SampleSize_by_age.png" width="1600" height="400">


> - Sample size for age from 12-21 is more stable ranging from 2000 to 3000+
     * for age from 12 - 17 is around +/-3000
     * for age from 18 - 21 is around +/-2500
> - Sample size for other age groups varies a lot. Data showed inconsistency in age group from 22 onwards

#### 2.2 : For better comparison and avoid misleading graph interpretation due to mixture of age vs age groups, convert age group into average


```python
def age_modified(age):
    if '+' in age:
        age = float(age.strip('+'))
    elif '-' in age:
        x = age.split('-')
        age = (float(x[1]) - float(x[0]))/2. + float(x[0])
    else:
        age = float(age)
    return age      
```


```python
drug['age'] = drug['age'].apply(age_modified)
drug
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
      <th>age</th>
      <th>n</th>
      <th>alcohol-use</th>
      <th>alcohol-frequency</th>
      <th>marijuana-use</th>
      <th>marijuana-frequency</th>
      <th>cocaine-use</th>
      <th>cocaine-frequency</th>
      <th>crack-use</th>
      <th>crack-frequency</th>
      <th>heroin-use</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-use</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-use</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-use</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-use</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-use</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-use</th>
      <th>stimulant-frequency</th>
      <th>meth-use</th>
      <th>meth-frequency</th>
      <th>sedative-use</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.0</td>
      <td>2798</td>
      <td>3.9</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>35.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>1.6</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>0.1</td>
      <td>24.5</td>
      <td>0.2</td>
      <td>52.0</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.2</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.0</td>
      <td>2757</td>
      <td>8.5</td>
      <td>6.0</td>
      <td>3.4</td>
      <td>15.0</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.6</td>
      <td>6.0</td>
      <td>2.5</td>
      <td>12.0</td>
      <td>2.4</td>
      <td>14.0</td>
      <td>0.1</td>
      <td>41.0</td>
      <td>0.3</td>
      <td>25.5</td>
      <td>0.3</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>2792</td>
      <td>18.1</td>
      <td>5.0</td>
      <td>8.7</td>
      <td>24.0</td>
      <td>0.1</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>1.6</td>
      <td>3.0</td>
      <td>2.6</td>
      <td>5.0</td>
      <td>3.9</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>4.5</td>
      <td>0.9</td>
      <td>5.0</td>
      <td>0.8</td>
      <td>12.0</td>
      <td>0.1</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>2956</td>
      <td>29.2</td>
      <td>6.0</td>
      <td>14.5</td>
      <td>25.0</td>
      <td>0.5</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>9.5</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>2.1</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>5.5</td>
      <td>5.5</td>
      <td>10.0</td>
      <td>0.8</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>6.0</td>
      <td>0.3</td>
      <td>10.5</td>
      <td>0.4</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>3058</td>
      <td>40.1</td>
      <td>10.0</td>
      <td>22.5</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.1</td>
      <td>66.5</td>
      <td>3.4</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.2</td>
      <td>7.0</td>
      <td>1.1</td>
      <td>4.0</td>
      <td>2.4</td>
      <td>11.0</td>
      <td>1.8</td>
      <td>9.5</td>
      <td>0.3</td>
      <td>36.0</td>
      <td>0.2</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17.0</td>
      <td>3038</td>
      <td>49.3</td>
      <td>13.0</td>
      <td>28.0</td>
      <td>36.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>21.0</td>
      <td>0.1</td>
      <td>64.0</td>
      <td>4.8</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>8.5</td>
      <td>9.0</td>
      <td>1.4</td>
      <td>6.0</td>
      <td>3.5</td>
      <td>7.0</td>
      <td>2.8</td>
      <td>9.0</td>
      <td>0.6</td>
      <td>48.0</td>
      <td>0.5</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18.0</td>
      <td>2469</td>
      <td>58.7</td>
      <td>24.0</td>
      <td>33.7</td>
      <td>52.0</td>
      <td>3.2</td>
      <td>5.0</td>
      <td>0.4</td>
      <td>10.0</td>
      <td>0.4</td>
      <td>46.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.8</td>
      <td>4.0</td>
      <td>9.2</td>
      <td>12.0</td>
      <td>1.7</td>
      <td>7.0</td>
      <td>4.9</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0.5</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>19.0</td>
      <td>2223</td>
      <td>64.6</td>
      <td>36.0</td>
      <td>33.4</td>
      <td>60.0</td>
      <td>4.1</td>
      <td>5.5</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>180.0</td>
      <td>8.6</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>3.0</td>
      <td>9.4</td>
      <td>12.0</td>
      <td>1.5</td>
      <td>7.5</td>
      <td>4.2</td>
      <td>4.5</td>
      <td>3.3</td>
      <td>6.0</td>
      <td>0.4</td>
      <td>105.0</td>
      <td>0.3</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20.0</td>
      <td>2271</td>
      <td>69.7</td>
      <td>48.0</td>
      <td>34.0</td>
      <td>60.0</td>
      <td>4.9</td>
      <td>8.0</td>
      <td>0.6</td>
      <td>5.0</td>
      <td>0.9</td>
      <td>45.0</td>
      <td>7.4</td>
      <td>2.0</td>
      <td>1.5</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.7</td>
      <td>12.0</td>
      <td>5.4</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>0.9</td>
      <td>12.0</td>
      <td>0.5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21.0</td>
      <td>2354</td>
      <td>83.2</td>
      <td>52.0</td>
      <td>33.0</td>
      <td>52.0</td>
      <td>4.8</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>17.0</td>
      <td>0.6</td>
      <td>30.0</td>
      <td>6.3</td>
      <td>4.0</td>
      <td>1.4</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>1.3</td>
      <td>13.5</td>
      <td>3.9</td>
      <td>7.0</td>
      <td>4.1</td>
      <td>10.0</td>
      <td>0.6</td>
      <td>2.0</td>
      <td>0.3</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>22.5</td>
      <td>4707</td>
      <td>84.2</td>
      <td>52.0</td>
      <td>28.4</td>
      <td>52.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>5.0</td>
      <td>1.1</td>
      <td>57.5</td>
      <td>5.2</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>1.7</td>
      <td>17.5</td>
      <td>4.4</td>
      <td>12.0</td>
      <td>3.6</td>
      <td>10.0</td>
      <td>0.6</td>
      <td>46.0</td>
      <td>0.2</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>24.5</td>
      <td>4591</td>
      <td>83.1</td>
      <td>52.0</td>
      <td>24.9</td>
      <td>60.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>0.5</td>
      <td>6.0</td>
      <td>0.7</td>
      <td>88.0</td>
      <td>4.5</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>1.3</td>
      <td>20.0</td>
      <td>4.3</td>
      <td>10.0</td>
      <td>2.6</td>
      <td>10.0</td>
      <td>0.7</td>
      <td>21.0</td>
      <td>0.2</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>27.5</td>
      <td>2628</td>
      <td>80.7</td>
      <td>52.0</td>
      <td>20.8</td>
      <td>52.0</td>
      <td>3.2</td>
      <td>5.0</td>
      <td>0.4</td>
      <td>6.0</td>
      <td>0.6</td>
      <td>50.0</td>
      <td>3.2</td>
      <td>3.0</td>
      <td>0.6</td>
      <td>4.0</td>
      <td>8.3</td>
      <td>13.0</td>
      <td>1.2</td>
      <td>13.5</td>
      <td>4.2</td>
      <td>10.0</td>
      <td>2.3</td>
      <td>7.0</td>
      <td>0.6</td>
      <td>30.0</td>
      <td>0.4</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>32.0</td>
      <td>2864</td>
      <td>77.5</td>
      <td>52.0</td>
      <td>16.4</td>
      <td>72.0</td>
      <td>2.1</td>
      <td>8.0</td>
      <td>0.5</td>
      <td>15.0</td>
      <td>0.4</td>
      <td>66.0</td>
      <td>1.8</td>
      <td>2.0</td>
      <td>0.4</td>
      <td>3.5</td>
      <td>5.9</td>
      <td>22.0</td>
      <td>0.9</td>
      <td>46.0</td>
      <td>3.6</td>
      <td>8.0</td>
      <td>1.4</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>54.0</td>
      <td>0.4</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>42.0</td>
      <td>7391</td>
      <td>75.0</td>
      <td>52.0</td>
      <td>10.4</td>
      <td>48.0</td>
      <td>1.5</td>
      <td>15.0</td>
      <td>0.5</td>
      <td>48.0</td>
      <td>0.1</td>
      <td>280.0</td>
      <td>0.6</td>
      <td>3.0</td>
      <td>0.3</td>
      <td>10.0</td>
      <td>4.2</td>
      <td>12.0</td>
      <td>0.3</td>
      <td>12.0</td>
      <td>1.9</td>
      <td>6.0</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>104.0</td>
      <td>0.3</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>57.0</td>
      <td>3923</td>
      <td>67.2</td>
      <td>52.0</td>
      <td>7.3</td>
      <td>52.0</td>
      <td>0.9</td>
      <td>36.0</td>
      <td>0.4</td>
      <td>62.0</td>
      <td>0.1</td>
      <td>41.0</td>
      <td>0.3</td>
      <td>44.0</td>
      <td>0.2</td>
      <td>13.5</td>
      <td>2.5</td>
      <td>12.0</td>
      <td>0.4</td>
      <td>5.0</td>
      <td>1.4</td>
      <td>10.0</td>
      <td>0.3</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>30.0</td>
      <td>0.2</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65.0</td>
      <td>2448</td>
      <td>49.3</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.6</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>364.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.3 : Set up 2 dataframes --> one by drugUse ; one by drugFrequency


```python
drug.columns
```




    Index(['age', 'n', 'alcohol-use', 'alcohol-frequency', 'marijuana-use',
           'marijuana-frequency', 'cocaine-use', 'cocaine-frequency', 'crack-use',
           'crack-frequency', 'heroin-use', 'heroin-frequency', 'hallucinogen-use',
           'hallucinogen-frequency', 'inhalant-use', 'inhalant-frequency',
           'pain-releiver-use', 'pain-releiver-frequency', 'oxycontin-use',
           'oxycontin-frequency', 'tranquilizer-use', 'tranquilizer-frequency',
           'stimulant-use', 'stimulant-frequency', 'meth-use', 'meth-frequency',
           'sedative-use', 'sedative-frequency'],
          dtype='object')




```python
use_columns = [col for col in drug.columns if 'use' in col]
frequency_columns = [col for col in drug.columns if 'frequency' in col]

df_drugUse = drug[use_columns]
df_drugFrequency = drug[frequency_columns]

# insert age column to the front of the new df
df_drugUse.insert(0, 'age', drug['age'] )
df_drugFrequency.insert(0, 'age', drug['age'])

```

Check if df by drugUse is correctly set up


```python
df_drugUse.head()
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
      <th>age</th>
      <th>alcohol-use</th>
      <th>marijuana-use</th>
      <th>cocaine-use</th>
      <th>crack-use</th>
      <th>heroin-use</th>
      <th>hallucinogen-use</th>
      <th>inhalant-use</th>
      <th>pain-releiver-use</th>
      <th>oxycontin-use</th>
      <th>tranquilizer-use</th>
      <th>stimulant-use</th>
      <th>meth-use</th>
      <th>sedative-use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.0</td>
      <td>3.9</td>
      <td>1.1</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>1.6</td>
      <td>2.0</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.0</td>
      <td>8.5</td>
      <td>3.4</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>2.5</td>
      <td>2.4</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>18.1</td>
      <td>8.7</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>1.6</td>
      <td>2.6</td>
      <td>3.9</td>
      <td>0.4</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>29.2</td>
      <td>14.5</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>2.1</td>
      <td>2.5</td>
      <td>5.5</td>
      <td>0.8</td>
      <td>2.0</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>40.1</td>
      <td>22.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>3.4</td>
      <td>3.0</td>
      <td>6.2</td>
      <td>1.1</td>
      <td>2.4</td>
      <td>1.8</td>
      <td>0.3</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



Check if df by drugFrequency is correctly set up


```python
df_drugFrequency.head()
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
      <th>age</th>
      <th>alcohol-frequency</th>
      <th>marijuana-frequency</th>
      <th>cocaine-frequency</th>
      <th>crack-frequency</th>
      <th>heroin-frequency</th>
      <th>hallucinogen-frequency</th>
      <th>inhalant-frequency</th>
      <th>pain-releiver-frequency</th>
      <th>oxycontin-frequency</th>
      <th>tranquilizer-frequency</th>
      <th>stimulant-frequency</th>
      <th>meth-frequency</th>
      <th>sedative-frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>35.5</td>
      <td>52.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>24.5</td>
      <td>52.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>25.5</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>5.5</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>4.0</td>
      <td>9.5</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.5</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>6.0</td>
      <td>10.5</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>10.0</td>
      <td>30.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>66.5</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>11.0</td>
      <td>9.5</td>
      <td>36.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4a : Visualize data by drugUse in stacked-bar chart


```python
df_drugUse.plot(x='age', kind='bar',stacked=True, figsize=(20,8), colormap='tab20',rot=1)
plt.ylabel('% of population taking the drug')
plt.title('Distribution of Population by Age for Various Drug Types')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Dist_stackedBar_drugUse.png" width="1600" height="400">


> - alcohol is the highest intake in all ages/age groups
> - marijuana is the 2nd highest drug intake among various ages/age groups. However, a reduction in marijuana use was observed for age 22 onwards
> - pain reliever is the 3rd popular drug with the % of people in the same age/age groups who used this drug remained stable for age 17-21

#### 2.4b : Another visualization of drugUse using line chart


```python
df_drugUse.plot('age', xticks=np.arange(10,70,5), figsize=(20,8))
plt.ylabel('% of age population')
plt.title('Distribution of Population by Age for Various Drug Types')
```


<img src="{{ site.baseurl }}/assets/img/portfolio/Dist_line_drugUse.png" width="1600" height="400">


#### 2.5a : Visualization of drug data by Frequency in stacked-bar chart


```python
df_drugFrequency.plot(x='age', figsize=(20,8), stacked=True, kind='bar', colormap='tab20')
plt.ylabel('Frequency')
plt.title('Distribution of Drug Intake Freqeuncy by Age')
```


<img src="{{ site.baseurl }}/assets/img/portfolio/Dist_stackedBar_drugFreq.png" width="1600" height="400">


> - heroin was the drug with highest frequency of intake age 19 and age group of 35-49
> - stimulant was found having a high spike of drug frequency in age group of 65+
> - marijuana frequency was found stable for age 18 till age group of 50-64

#### 2.5b : Another visualization of drugFrequency using line chart


```python
df_drugFrequency.plot('age', figsize=(20,8), xticks=np.arange(10,70,5))
plt.ylabel('Median Frequency of Drug Intake')
plt.title('Distribution of Drug Frequency by Age for Various Drug Types')
```


<img src="{{ site.baseurl }}/assets/img/portfolio/Dist_line_drugFreq.png" width="1600" height="400">



> - Visualization through stacked bar gave better comparison view as compared to line plot

#### 2.6 : Check the spread of the data for each category using boxplot
Since data range varies significantly, standardized data before boxplot to enable comparable values on same scale


```python
drugUse_bx = df_drugUse.drop('age', axis=1)
drugFrequency_bx = df_drugFrequency.drop('age', axis=1)

# standardize data on same scale
std_drugUse_bx = (drugUse_bx - drugUse_bx.mean())/drugUse_bx.std()
std_drugFrequency_bx = (drugFrequency_bx - drugFrequency_bx.mean())/drugFrequency_bx.std()

```


```python
std_drugUse_bx
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
      <th>alcohol-use</th>
      <th>marijuana-use</th>
      <th>cocaine-use</th>
      <th>crack-use</th>
      <th>heroin-use</th>
      <th>hallucinogen-use</th>
      <th>inhalant-use</th>
      <th>pain-releiver-use</th>
      <th>oxycontin-use</th>
      <th>tranquilizer-use</th>
      <th>stimulant-use</th>
      <th>meth-use</th>
      <th>sedative-use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.917098</td>
      <td>-1.490293</td>
      <td>-1.142945</td>
      <td>-1.247469</td>
      <td>-0.757849</td>
      <td>-1.143818</td>
      <td>0.228371</td>
      <td>-1.348729</td>
      <td>-1.373352</td>
      <td>-1.486206</td>
      <td>-1.220203</td>
      <td>-1.455128</td>
      <td>-0.596759</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.745959</td>
      <td>-1.297981</td>
      <td>-1.142945</td>
      <td>-1.247469</td>
      <td>-1.057464</td>
      <td>-1.000577</td>
      <td>1.198949</td>
      <td>-1.222402</td>
      <td>-1.373352</td>
      <td>-1.429173</td>
      <td>-1.149164</td>
      <td>-1.074556</td>
      <td>-1.321394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.388802</td>
      <td>-0.854828</td>
      <td>-1.142945</td>
      <td>-1.247469</td>
      <td>-0.757849</td>
      <td>-0.642476</td>
      <td>1.306791</td>
      <td>-0.748675</td>
      <td>-0.880106</td>
      <td>-1.086977</td>
      <td>-0.793968</td>
      <td>-1.074556</td>
      <td>-0.596759</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.975838</td>
      <td>-0.369868</td>
      <td>-0.922774</td>
      <td>-0.823329</td>
      <td>-0.458234</td>
      <td>-0.463425</td>
      <td>1.198949</td>
      <td>-0.243366</td>
      <td>-0.222444</td>
      <td>-0.459617</td>
      <td>-0.296693</td>
      <td>-0.313412</td>
      <td>0.852512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.570315</td>
      <td>0.299042</td>
      <td>-0.647561</td>
      <td>-1.247469</td>
      <td>-0.757849</td>
      <td>0.002106</td>
      <td>1.738159</td>
      <td>-0.022293</td>
      <td>0.270802</td>
      <td>-0.231486</td>
      <td>-0.083576</td>
      <td>-0.313412</td>
      <td>-0.596759</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.228038</td>
      <td>0.758918</td>
      <td>-0.097134</td>
      <td>-0.823329</td>
      <td>-0.757849</td>
      <td>0.503448</td>
      <td>0.659739</td>
      <td>0.704089</td>
      <td>0.764048</td>
      <td>0.395874</td>
      <td>0.626817</td>
      <td>0.828303</td>
      <td>1.577148</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.121679</td>
      <td>1.235516</td>
      <td>0.563378</td>
      <td>0.449089</td>
      <td>0.140995</td>
      <td>1.291271</td>
      <td>0.444055</td>
      <td>0.925161</td>
      <td>1.257294</td>
      <td>1.194333</td>
      <td>0.768895</td>
      <td>0.447732</td>
      <td>0.852512</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.341182</td>
      <td>1.210432</td>
      <td>1.058762</td>
      <td>0.873228</td>
      <td>0.440610</td>
      <td>1.864233</td>
      <td>0.012687</td>
      <td>0.988325</td>
      <td>0.928463</td>
      <td>0.795103</td>
      <td>0.982013</td>
      <td>0.067160</td>
      <td>0.127877</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.530922</td>
      <td>1.260601</td>
      <td>1.499103</td>
      <td>1.297367</td>
      <td>1.639069</td>
      <td>1.434512</td>
      <td>0.120529</td>
      <td>1.177816</td>
      <td>1.257294</td>
      <td>1.479496</td>
      <td>1.479287</td>
      <td>1.970019</td>
      <td>1.577148</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.033176</td>
      <td>1.176987</td>
      <td>1.444061</td>
      <td>0.873228</td>
      <td>0.740225</td>
      <td>1.040600</td>
      <td>0.012687</td>
      <td>0.861998</td>
      <td>0.599632</td>
      <td>0.624005</td>
      <td>1.550326</td>
      <td>0.828303</td>
      <td>0.127877</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.070380</td>
      <td>0.792363</td>
      <td>1.278933</td>
      <td>0.873228</td>
      <td>2.238298</td>
      <td>0.646689</td>
      <td>-0.418681</td>
      <td>1.177816</td>
      <td>1.257294</td>
      <td>0.909169</td>
      <td>1.195130</td>
      <td>0.828303</td>
      <td>-0.596759</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.029455</td>
      <td>0.499715</td>
      <td>1.003719</td>
      <td>0.873228</td>
      <td>1.039839</td>
      <td>0.396018</td>
      <td>-0.634365</td>
      <td>0.861998</td>
      <td>0.599632</td>
      <td>0.852136</td>
      <td>0.484738</td>
      <td>1.208875</td>
      <td>-0.596759</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.940166</td>
      <td>0.156899</td>
      <td>0.563378</td>
      <td>0.449089</td>
      <td>0.740225</td>
      <td>-0.069514</td>
      <td>-0.850049</td>
      <td>0.640925</td>
      <td>0.435217</td>
      <td>0.795103</td>
      <td>0.271621</td>
      <td>0.828303</td>
      <td>0.852512</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.821113</td>
      <td>-0.211002</td>
      <td>-0.042091</td>
      <td>0.873228</td>
      <td>0.140995</td>
      <td>-0.570856</td>
      <td>-1.065733</td>
      <td>-0.117038</td>
      <td>-0.058029</td>
      <td>0.452907</td>
      <td>-0.367732</td>
      <td>0.067160</td>
      <td>0.852512</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.728103</td>
      <td>-0.712684</td>
      <td>-0.372347</td>
      <td>0.873228</td>
      <td>-0.757849</td>
      <td>-1.000577</td>
      <td>-1.173575</td>
      <td>-0.653929</td>
      <td>-1.044521</td>
      <td>-0.516649</td>
      <td>-0.936046</td>
      <td>-0.693984</td>
      <td>0.127877</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.437912</td>
      <td>-0.971887</td>
      <td>-0.702603</td>
      <td>0.449089</td>
      <td>-0.757849</td>
      <td>-1.108008</td>
      <td>-1.281417</td>
      <td>-1.190820</td>
      <td>-0.880106</td>
      <td>-0.801813</td>
      <td>-1.149164</td>
      <td>-0.693984</td>
      <td>-0.596759</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.228038</td>
      <td>-1.481931</td>
      <td>-1.197988</td>
      <td>-1.247469</td>
      <td>-1.057464</td>
      <td>-1.179628</td>
      <td>-1.497101</td>
      <td>-1.790875</td>
      <td>-1.537767</td>
      <td>-1.486206</td>
      <td>-1.362281</td>
      <td>-1.455128</td>
      <td>-2.046029</td>
    </tr>
  </tbody>
</table>
</div>



Comparision of various drugUse spread on same scale


```python
plt.figure(figsize=(20,10))
sns.boxplot(data=std_drugUse_bx, orient='h',)
plt.title('Boxplot on Standardized Scale for DrugUse')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Boxplot_drugUse.png" width="1600" height="400">


Comparision of various drugFrequency spread on same scale


```python
plt.figure(figsize=(20,10))
sns.boxplot(data=std_drugFrequency_bx, orient='h')
plt.title('Boxplot on Standardized Scale for DrugFrequency')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Boxplot_drugFreq.png" width="1600" height="400">


> - drugFrequency data is more scatter and with more outliers as compared to drugUse data

#### 2.7a : Check the correlation of data in drugUse dataset


```python
drugUse_temp = df_drugUse.drop('age', axis=1)
drugFrequency_temp = df_drugFrequency.drop('age', axis=1)
```


```python
plt.figure(figsize=(12,8))
sns.heatmap(drugUse_temp.corr(),cmap='RdBu_r',annot=True)
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Correlation_drugUse.png" width="1600" height="500">


> - inhalant-use has almost no to negative correlation to the rest of the drug use
> - other drug-use are generally having positive correlation to each other in different levels

Visualization of the correlation of top 4 popular drugs (alcohol, marijuana, hallucinogen, and pain-reliever)


```python
sns.pairplot(drugUse_temp[['alcohol-use','marijuana-use','hallucinogen-use','pain-releiver-use']],kind='reg')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Top4_popular_drugUse.png" width="1600" height="800">


> - all 4 drugs showed positive correlation to each other
> - alcohol-use data was observed to behave more scatter-correlated with other drugs
> - whilst the other 3 drugs(marijuana, hallucinogen, and pain-reliever) were found quite well positively correlated

#### 2.7b : Check the correlation of data in drugFrequency dataset


```python
plt.figure(figsize=(12,8))
sns.heatmap(data=drugFrequency_temp.corr(), cmap='RdBu_r', annot=True)
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Correlation_drugFreq.png" width="1600" height="500">


> - mixture of positive and negative correlation among various drugFrequency
> - crack-frequeny and stimulant-frequency has upto 0.9 positive correlation

Visualization the correlation of top 4 popular drugFreq(alcohol, marijuana, hallucinogen, and pain-reliever)


```python
sns.pairplot(drugFrequency_temp[['alcohol-frequency','marijuana-frequency','hallucinogen-frequency','pain-releiver-frequency']],kind='reg')
```




<img src="{{ site.baseurl }}/assets/img/portfolio/Top4_popular_drugFreq.png" width="1600" height="800">


> - No significant correlation was observed for all the 4 drugs frequency
> - Hallucinogen-frequency was found having 2 extreme scale with majority of data at lower frequency level.
> - The high hallucinogen-frequency data points could be outliers/exceptional intakes

## Part 3 : Hypothesis Generation and Testing

---

In the data exploration process, it is common that we would need to generate some assumptions, testify and validate those assumptions before we can summarize the findings in order to make solid conclusions. For this session, observation in Part 2 was used to practise hypothesis generation and testing.

> #### Question to explore :
> - Correlation matrix showed significant correlation (0.98) of oxycontin-use vs pain-reliever-use.
> - Are the drug users in pain-reliever having the similar age group distribution as the drug users in oxycontine?
$$ H_0: Use_{pain-reliever} = Use_{oxycontin} $$
$$ H_1: Use_{pain-reliever} \neq Use_{oxycontin} $$
> - But for their frequencies, correlation was only at 0.56. Are these correlation statistically significant?
> - Among these 2 groups of drug users, are they taking the pain-reliever as frequent as oxycontine?
$$ H_0: frequency_{pain-reliever} = frequency_{oxycontin} $$
$$ H_1: frequency_{pain-reliever} \neq frequency_{oxycontin} $$

> #### Deliverables :
> - join-plot
> - stats summary to include p-values

#### 3.a : Comparison on Drug Use 
1st examination through graphical view


```python
sns.jointplot(x='pain-releiver-use', y='oxycontin-use', data=drugUse_temp, kind='reg')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Hypothesis_jointplot_drugUse.png" width="800" height="500">


2nd examination through stats library on p-value


```python
p_value_drugUse = stats.ttest_ind(drugUse_temp['pain-releiver-use'],drugUse_temp['oxycontin-use'])
p_value_drugUse
```




    Ttest_indResult(statistic=6.82263516475104, pvalue=1.0265878201430413e-07)



#### 3.b : Comparison on Drug Frequency
1st examination through graphical view
- Repeat the same workflow as in 3.a


```python
sns.jointplot(x='pain-releiver-frequency', y='oxycontin-frequency', data=drugFrequency_temp, kind='reg')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Hypothesis_jointplot_drugFreq.png" width="800" height="500">



```python
p_value_drugFrequency = stats.ttest_ind(drugFrequency_temp['pain-releiver-frequency'], drugFrequency_temp['oxycontin-frequency'],nan_policy='omit')
p_value_drugFrequency
# need to set nan_policy='omit' in order to ignore nan value in oxycontin-frequency for p-value calculation
```




    Ttest_indResult(statistic=-0.030003630957118617, pvalue=0.9762564938195634)



> #### Conclusion :

> #### for drug use
> Pearson correlation coefficient was close to 1, reported high at 0.98. 
> p-value is small at 1.0265878201430413e-07, therefore null hypothesis is rejected
> drug user age group in pain-reliever is positively correlated to the drug user age group in oxycontine
 
> #### for drug frequency
> Pearson correlation coefficient was only at 0.56 
> p-value is small at 0.9762564938195634, therefore null hypothesis is accepted 
> No conclusion can be made for drug frequency in pain reliever vs drug frequency in osycontine

## Part 4 : Outliers Handling

---

Outliers handling is common in data analysis. In this session, a subset of the data is extracted and used to outline the flow on how outliers could be examined and corrected.

- Pain-reliever-frequency is used to study outlier effect


```python
fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)

sns.boxplot(data=drugFrequency_temp['pain-releiver-frequency'], orient='h',ax=ax[0])
sns.distplot(drugFrequency_temp['pain-releiver-frequency'], bins=30, ax=ax[1])
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Outliers_handling.png" width="1600" height="400">


> 4 outlier data points were observed

#### 4.a : Extraction of outlier data points


```python
# Get the IQR
dataExamined = drugFrequency_temp['pain-releiver-frequency']
q25, q75 = np.percentile(dataExamined, [25,75])
IQR = q75 - q25

# Get outlier point below q25 and above q75
outliers_abv = dataExamined[dataExamined>(q75+1.5*IQR)] 
outliers_below = dataExamined[dataExamined<(q25-1.5*IQR)]

# List out all outlier points
outliers_list = list(outliers_below.append(outliers_abv))
outliers_list
```




    [7.0, 36.0, 22.0, 24.0]



#### 4.b : Removal of outlier data points from examined dataset


```python
dataExamined_clean = [v for v in dataExamined if v not in outliers_list]
print(len(dataExamined_clean))
dataExamined_clean
```

    13





    [14.0, 12.0, 10.0, 9.0, 12.0, 12.0, 10.0, 15.0, 15.0, 15.0, 13.0, 12.0, 12.0]



#### 4.c : Comparison of mean, median, std dev for dataset with outliers vs no outlier


```python
# set up dataframe for comparison
df_comparison = pd.DataFrame(dataExamined)
df_comparison.columns = ['With outliers']

# replace those identified as outlier point with nan in order to do stats calculation for scenario with no outliers
df_comparison['No outliers'] = df_comparison['With outliers'].map(lambda x: np.nan if x in outliers_list else x)
df_comparison.head()
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
      <th>With outliers</th>
      <th>No outliers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### 4.d : Tranpose dataframe for ease of comparison and calculation by columns


```python
df_comparison_T = df_comparison.T
df_comparison_T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>With outliers</th>
      <td>36.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>No outliers</th>
      <td>NaN</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_comparison_T['mean'] = df_comparison_T.mean(axis=1)
df_comparison_T['median'] = df_comparison_T.median(axis=1)
df_comparison_T['stdDev'] = df_comparison_T.std(axis=1)
df_comparison_T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>mean</th>
      <th>median</th>
      <th>stdDev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>With outliers</th>
      <td>36.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>14.705882</td>
      <td>12.5</td>
      <td>6.558028</td>
    </tr>
    <tr>
      <th>No outliers</th>
      <td>NaN</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>12.384615</td>
      <td>12.0</td>
      <td>1.836437</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_comparison_T[['mean','median','stdDev']]
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
      <th>mean</th>
      <th>median</th>
      <th>stdDev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>With outliers</th>
      <td>14.705882</td>
      <td>12.5</td>
      <td>6.558028</td>
    </tr>
    <tr>
      <th>No outliers</th>
      <td>12.384615</td>
      <td>12.0</td>
      <td>1.836437</td>
    </tr>
  </tbody>
</table>
</div>



> - all the 3 mean, median and stdDev are smaller when the outlier data points are excluded


```python
sns.boxplot(data=df_comparison,orient='h')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Outliers_vs_NoOutliers_boxplot.png" width="700" height="300">



```python
# addition plot to see the distribution of df with no outliers

fig, ax = plt.subplots(2,2,figsize=(15,6), sharex=True)

sns.boxplot(data=df_comparison['With outliers'], orient='h', ax=ax[0][0])
sns.distplot(df_comparison['With outliers'], bins=30, ax=ax[1][0])

sns.boxplot(data=df_comparison['No outliers'], orient='h', ax=ax[0][1], color='g')
sns.distplot(df_comparison.dropna()['No outliers'], bins=30, ax=ax[1][1], color='g')
```



<img src="{{ site.baseurl }}/assets/img/portfolio/Outliers_comparison_overview.png" width="1600" height="300">


## Key Learnings

---
There are many techniques that we can use to explore data. No one single best method to outline how to explore the data perfectly. It is very much data dependent and can have different creative thoughts to present and visualize the data. Most importantly, the presented method is able to level up the understanding of the data and therefore expand the usage of such data for higher order of processing or modeling.
