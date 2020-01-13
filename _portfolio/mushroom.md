---
layout: post
title: Analysis on Mushroom Edibility
img: "assets/img/portfolio/mushroom_icon.png"
date: April, 18 2018

---

![image]({{ site.baseurl }}/{{ page.img }})

Love to eat mushroom but not sure if it is ebible? There's a dataset in [Kaggle](https://www.kaggle.com/uciml/mushroom-classification) which can be used to build a model classifying if a mushroom is safe and consumable. 

The main goal of this little task is to attempt aswering 2 questions :
1. What types of machine learning models perform best on this dataset?
2. Which features are most indicative of a poisonous mushroom?

Also, to help people visualizing the performance of the model and its prediction outcome, a simple web application is built after model building. 


## __Part 1 : Raw Data Exploration__

Instaed of the conventional way of exploring the raw data with common pandas library, I decided to try an enhanced version of libraray called [pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/). After a quick look on the documentation, I found it was pretty straight forward to generate a nice and comprehensive report covering the key areas of data screening.

After reading in the dataframe (df), just execute simple line of code 
```python
import pandas_profiling as pp

pandas_report = pp.ProfileReport(df)
```

The pandas report is a html file outlining 5 key summaries [ Overview, Variable, Correlation, Missing values, Sample]. 

{% include pprofiling_demo_video.html %}

<video src="{{ site.baseurl }}/assets/img/portfolio/pprofling_report_demo.mp4" width="680" height="360" controls preload></video>



Based on the pandas report, I can now check the summarized statistic variable by variable in more details . Here are the observations and preliminary considerations for model building after studying the report :


> * 'vell-type' was observed to have only 1 unique variable, others with >=2 unique variables, max unique variable at 12
> * 'vell-type' to be excluded from modelling since it's a constant value
> * 'stock-root' was found having special character "?"
> * 'odor' and 'gill-color' were observed to have positive correlation in this preliminary analysis. To verify further.


## __Part 2 : Data Input Format Transformation__

Before working on model bulding, data was transformed into processable format as input to the model. As all variables were categorical data, label encoder was used on the `class` variable while other variables were transformed using one hot encoder via `pd.get_dummies()`.

Snippet of the transformed data after dropping `vell-type` column due to constant value as below :

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
      <th>cap-shape_b</th>
      <th>cap-shape_c</th>
      <th>cap-shape_f</th>
      <th>cap-shape_k</th>
      <th>cap-shape_s</th>
      <th>cap-shape_x</th>
      <th>cap-surface_f</th>
      <th>cap-surface_g</th>
      <th>cap-surface_s</th>
      <th>cap-surface_y</th>
      <th>cap-color_b</th>
      <th>cap-color_c</th>
      <th>cap-color_e</th>
      <th>cap-color_g</th>
      <th>cap-color_n</th>
      <th>cap-color_p</th>
      <th>cap-color_r</th>
      <th>cap-color_u</th>
      <th>cap-color_w</th>
      <th>cap-color_y</th>
      <th>bruises_f</th>
      <th>bruises_t</th>
      <th>odor_a</th>
      <th>odor_c</th>
      <th>odor_f</th>
      <th>odor_l</th>
      <th>odor_m</th>
      <th>odor_n</th>
      <th>odor_p</th>
      <th>odor_s</th>
      <th>odor_y</th>
      <th>gill-attachment_a</th>
      <th>gill-attachment_f</th>
      <th>gill-spacing_c</th>
      <th>gill-spacing_w</th>
      <th>gill-size_b</th>
      <th>gill-size_n</th>
      <th>gill-color_b</th>
      <th>gill-color_e</th>
      <th>gill-color_g</th>
      <th>gill-color_h</th>
      <th>gill-color_k</th>
      <th>gill-color_n</th>
      <th>gill-color_o</th>
      <th>gill-color_p</th>
      <th>gill-color_r</th>
      <th>gill-color_u</th>
      <th>gill-color_w</th>
      <th>gill-color_y</th>
      <th>stalk-shape_e</th>
      <th>stalk-shape_t</th>
      <th>stalk-root_?</th>
      <th>stalk-root_b</th>
      <th>stalk-root_c</th>
      <th>stalk-root_e</th>
      <th>stalk-root_r</th>
      <th>stalk-surface-above-ring_f</th>
      <th>stalk-surface-above-ring_k</th>
      <th>stalk-surface-above-ring_s</th>
      <th>stalk-surface-above-ring_y</th>
      <th>stalk-surface-below-ring_f</th>
      <th>stalk-surface-below-ring_k</th>
      <th>stalk-surface-below-ring_s</th>
      <th>stalk-surface-below-ring_y</th>
      <th>stalk-color-above-ring_b</th>
      <th>stalk-color-above-ring_c</th>
      <th>stalk-color-above-ring_e</th>
      <th>stalk-color-above-ring_g</th>
      <th>stalk-color-above-ring_n</th>
      <th>stalk-color-above-ring_o</th>
      <th>stalk-color-above-ring_p</th>
      <th>stalk-color-above-ring_w</th>
      <th>stalk-color-above-ring_y</th>
      <th>stalk-color-below-ring_b</th>
      <th>stalk-color-below-ring_c</th>
      <th>stalk-color-below-ring_e</th>
      <th>stalk-color-below-ring_g</th>
      <th>stalk-color-below-ring_n</th>
      <th>stalk-color-below-ring_o</th>
      <th>stalk-color-below-ring_p</th>
      <th>stalk-color-below-ring_w</th>
      <th>stalk-color-below-ring_y</th>
      <th>veil-color_n</th>
      <th>veil-color_o</th>
      <th>veil-color_w</th>
      <th>veil-color_y</th>
      <th>ring-number_n</th>
      <th>ring-number_o</th>
      <th>ring-number_t</th>
      <th>ring-type_e</th>
      <th>ring-type_f</th>
      <th>ring-type_l</th>
      <th>ring-type_n</th>
      <th>ring-type_p</th>
      <th>spore-print-color_b</th>
      <th>spore-print-color_h</th>
      <th>spore-print-color_k</th>
      <th>spore-print-color_n</th>
      <th>spore-print-color_o</th>
      <th>spore-print-color_r</th>
      <th>spore-print-color_u</th>
      <th>spore-print-color_w</th>
      <th>spore-print-color_y</th>
      <th>population_a</th>
      <th>population_c</th>
      <th>population_n</th>
      <th>population_s</th>
      <th>population_v</th>
      <th>population_y</th>
      <th>habitat_d</th>
      <th>habitat_g</th>
      <th>habitat_l</th>
      <th>habitat_m</th>
      <th>habitat_p</th>
      <th>habitat_u</th>
      <th>habitat_w</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Based on this transformed dataframe, data was then further splitted into train and test set using split ratio of 0.3

```python
# split data to train and test set

data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=42)
```

A preliminary check on the correlation of all the features to `class` was also done to have an overview if features were positively or negatively correlated to the target prediction variable `class`. 

```python
# check correlation of features to class
correlation_overview = data_all.corr()['class'].reset_index()
correlation_overview.rename(columns={'index':'features', 'class':'correlation_to_class'}, inplace=True)
df_feature_correlation = correlation_overview.sort_values(by='correlation_to_class', ascending=False)
df_feature_correlation
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
      <th>features</th>
      <th>correlation_to_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>116</th>
      <td>class</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>odor_f</td>
      <td>0.623842</td>
    </tr>
    <tr>
      <th>57</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.587658</td>
    </tr>
    <tr>
      <th>61</th>
      <td>stalk-surface-below-ring_k</td>
      <td>0.573524</td>
    </tr>
    <tr>
      <th>36</th>
      <td>gill-size_n</td>
      <td>0.540024</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>stalk-surface-above-ring_s</td>
      <td>-0.491314</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bruises_t</td>
      <td>-0.501530</td>
    </tr>
    <tr>
      <th>35</th>
      <td>gill-size_b</td>
      <td>-0.540024</td>
    </tr>
    <tr>
      <th>93</th>
      <td>ring-type_p</td>
      <td>-0.540469</td>
    </tr>
    <tr>
      <th>27</th>
      <td>odor_n</td>
      <td>-0.785557</td>
    </tr>
  </tbody>
</table>
<p>117 rows × 2 columns</p>
</div>


## __Part 3 : Model Building & Performance Check__

Since this was a classification problem, 3 different algorithms were chosen to check prediction outcomes and their performance. Models chosen were :

> * logistic_regressionCV
> * random_forest_classifier
> * kneighbors_classifier

To make the entire train and performance check process iterable for 3 chosen models, a function was written to serve this process.


```python
# define few models to train
model_lr = LogisticRegressionCV(random_state=42)
model_rf = RandomForestClassifier(random_state=42)
model_kn = KNeighborsClassifier()

models_name = ['logistic_regressionCV', 'random_forest_classifier', 'kneighbors_classifier']
models = [model_lr, model_rf, model_kn]
```


```python
def model_building(model, model_name, X_train, X_test, y_train, y_test):
    """
    Function to train model and output model performance
    """
    
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    y_prob_predict = model.predict_proba(X_test)
    
    # save trained model
    filename = f'./pretrained_models/saved_model_{model_name}.sav'
    joblib.dump(model, filename)
    

    print('\n')
    print(colored(model_name, color='red', attrs=['bold']))
    print('-'*80)
    print('Model Parameters :')
    print(model)
    print('-'*80)
    print('\n')
    
    # To plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test, display_labels=['Edible','Poisonous'], cmap='magma')
    plt.show(block=False)
    
    # To display classification report showing precision, recall, f1
    print('-'*80)
    print('Classification report :\n')
    print(classification_report(y_test, y_predict))
    print('-'*80)

    # To plot roc_auc graph to show area under the curve
    print('ROC_AUC score :')
    print(round(roc_auc_score(y_test, y_prob_predict[:,1]),4))


    fpr, tpr, _ = roc_curve(y_test, y_prob_predict[:,1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=[6,6])
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Mushroom Classifier : Edible or Poisonous', fontsize=12)
    plt.legend(loc="lower right")
    plt.show()

    
```


```python
for i, model in enumerate(models):   
    
    model_building(model, models_name[i], data_train, data_test, label_train, label_test)
```

For an initial trial, default parameters were used for all 3 models. The model performances were found good even with the default parameters.


3.1 Model Performance for Logistic RegressionCV


    --------------------------------------------------------------------------------
    Model Parameters :
    LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                         fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                         max_iter=100, multi_class='auto', n_jobs=None,
                         penalty='l2', random_state=42, refit=True, scoring=None,
                         solver='lbfgs', tol=0.0001, verbose=0)
    --------------------------------------------------------------------------------
    
    
<img src="{{ site.baseurl }}/assets/img/portfolio/confMat_logR.png">    


    --------------------------------------------------------------------------------
    Classification report :
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1257
               1       1.00      1.00      1.00      1181
    
        accuracy                           1.00      2438
       macro avg       1.00      1.00      1.00      2438
    weighted avg       1.00      1.00      1.00      2438
    
    --------------------------------------------------------------------------------
    ROC_AUC score :
    1.0



<img src="{{ site.baseurl }}/assets/img/portfolio/roc_logR.png">  

3.2 Model Performance for Random Forest Classifier
 

    --------------------------------------------------------------------------------
    Model Parameters :
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)
    --------------------------------------------------------------------------------
    
    
<img src="{{ site.baseurl }}/assets/img/portfolio/confMat_randomF.png"> 


    --------------------------------------------------------------------------------
    Classification report :
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1257
               1       1.00      1.00      1.00      1181
    
        accuracy                           1.00      2438
       macro avg       1.00      1.00      1.00      2438
    weighted avg       1.00      1.00      1.00      2438
    
    --------------------------------------------------------------------------------
    ROC_AUC score :
    1.0



<img src="{{ site.baseurl }}/assets/img/portfolio/roc_randomF.png">  

3.3 Model Performance for KNeighbors Classifier


    --------------------------------------------------------------------------------
    Model Parameters :
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')
    --------------------------------------------------------------------------------
    
    
<img src="{{ site.baseurl }}/assets/img/portfolio/confMat_kNeighbors.png"> 


    --------------------------------------------------------------------------------
    Classification report :
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1257
               1       1.00      1.00      1.00      1181
    
        accuracy                           1.00      2438
       macro avg       1.00      1.00      1.00      2438
    weighted avg       1.00      1.00      1.00      2438
    
    --------------------------------------------------------------------------------
    ROC_AUC score :
    1.0


<img src="{{ site.baseurl }}/assets/img/portfolio/roc_kNeighbors.png">  


## __Part 4 : Analysis of Feature Importance__

To check if there was any variable dominating in predicting the mushroom class and how much the dominance level was, analysis was carried out on the trained models.

### 4.1 Coefficients derived from Logistic RegressionCV Model

For Logistic RegressionCV, features impact on the model can be extracted by checking the coefficient.


```python

trained_model_lr = joblib.load('./pretrained_models/saved_model_logistic_regressionCV.sav')

# extract feature importance from trained model
feat_importance_lr = trained_model_lr.coef_.tolist()
feat_importance_lr = [round(v, 6) for v in feat_importance_lr[0]]

# set up df for better comparison view
df_feat_importance_lr = pd.DataFrame(data_train.columns, columns=['features'])
df_feat_importance_lr['feature_importance'] = feat_importance_lr
df_feat_importance_lr = df_feat_importance_lr.sort_values('feature_importance', ascending=False)
df_feat_importance_lr
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
      <th>features</th>
      <th>feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>spore-print-color_r</td>
      <td>6.427334</td>
    </tr>
    <tr>
      <th>23</th>
      <td>odor_c</td>
      <td>4.933256</td>
    </tr>
    <tr>
      <th>24</th>
      <td>odor_f</td>
      <td>4.160369</td>
    </tr>
    <tr>
      <th>52</th>
      <td>stalk-root_b</td>
      <td>3.802192</td>
    </tr>
    <tr>
      <th>28</th>
      <td>odor_p</td>
      <td>3.558643</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>100</th>
      <td>spore-print-color_u</td>
      <td>-2.710225</td>
    </tr>
    <tr>
      <th>35</th>
      <td>gill-size_b</td>
      <td>-3.353126</td>
    </tr>
    <tr>
      <th>25</th>
      <td>odor_l</td>
      <td>-4.993461</td>
    </tr>
    <tr>
      <th>22</th>
      <td>odor_a</td>
      <td>-5.036676</td>
    </tr>
    <tr>
      <th>27</th>
      <td>odor_n</td>
      <td>-6.073253</td>
    </tr>
  </tbody>
</table>
<p>116 rows × 2 columns</p>
</div>

To get the top 10 important features, concentation was done to merge the top 5 positive coefficient with top 5 negative coefficient.

```python
df_feat_coef_top10_lr = pd.concat([df_feat_importance_lr.head(5), df_feat_importance_lr.tail(5)])
df_feat_coef_top10_lr.sort_values('feature_importance', ascending=False)
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
      <th>features</th>
      <th>feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>spore-print-color_r</td>
      <td>6.427334</td>
    </tr>
    <tr>
      <th>23</th>
      <td>odor_c</td>
      <td>4.933256</td>
    </tr>
    <tr>
      <th>24</th>
      <td>odor_f</td>
      <td>4.160369</td>
    </tr>
    <tr>
      <th>52</th>
      <td>stalk-root_b</td>
      <td>3.802192</td>
    </tr>
    <tr>
      <th>28</th>
      <td>odor_p</td>
      <td>3.558643</td>
    </tr>
    <tr>
      <th>100</th>
      <td>spore-print-color_u</td>
      <td>-2.710225</td>
    </tr>
    <tr>
      <th>35</th>
      <td>gill-size_b</td>
      <td>-3.353126</td>
    </tr>
    <tr>
      <th>25</th>
      <td>odor_l</td>
      <td>-4.993461</td>
    </tr>
    <tr>
      <th>22</th>
      <td>odor_a</td>
      <td>-5.036676</td>
    </tr>
    <tr>
      <th>27</th>
      <td>odor_n</td>
      <td>-6.073253</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_feat_coef_top10_lr.plot(x='features', y='feature_importance', kind='barh', figsize=(12,6))
plt.title('Top 10 Feature for Mushroom Class Prediction based on Logistic_Regression_CV')
plt.show()
```


<img src="{{ site.baseurl }}/assets/img/portfolio/featimportance_logR.png">  


> * No particular feature showing dominant effect on mushroom class prediction
> * In general, combination of odor, gill-size and spore-print-color demonstrated higher effects on final mushroom class prediction


### 4.2 Feature Importance derived from Random Forest Classifier Model

Different from Logistic RegressionCV, to identify contributing features, feature importance was extracted out from Random Forest model. 

```python
trained_model_rf = joblib.load('./pretrained_models/saved_model_random_forest_classifier.sav')

# extract feature importance from trained model
feat_importance_rf = list(trained_model_rf.feature_importances_)
feat_importance_rf = [round(v, 6) for v in feat_importance_rf]

# set up df for better comparison view
df_feat_importance_rf = pd.DataFrame(data_train.columns, columns=['features'])
df_feat_importance_rf['feature_importance'] = feat_importance_rf
df_feat_importance_rf.sort_values('feature_importance', ascending=False)
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
      <th>features</th>
      <th>feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>odor_n</td>
      <td>0.096636</td>
    </tr>
    <tr>
      <th>35</th>
      <td>gill-size_b</td>
      <td>0.071668</td>
    </tr>
    <tr>
      <th>57</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.070451</td>
    </tr>
    <tr>
      <th>24</th>
      <td>odor_f</td>
      <td>0.069137</td>
    </tr>
    <tr>
      <th>36</th>
      <td>gill-size_n</td>
      <td>0.053173</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>stalk-color-above-ring_e</td>
      <td>0.000009</td>
    </tr>
    <tr>
      <th>98</th>
      <td>spore-print-color_o</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>102</th>
      <td>spore-print-color_y</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>94</th>
      <td>spore-print-color_b</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>gill-color_o</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>116 rows × 2 columns</p>
</div>


```python
df_feat_imp_top10 = df_feat_importance_rf.sort_values('feature_importance', ascending=False).head(10)
df_feat_imp_top10
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
      <th>features</th>
      <th>feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>odor_n</td>
      <td>0.096636</td>
    </tr>
    <tr>
      <th>35</th>
      <td>gill-size_b</td>
      <td>0.071668</td>
    </tr>
    <tr>
      <th>57</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.070451</td>
    </tr>
    <tr>
      <th>24</th>
      <td>odor_f</td>
      <td>0.069137</td>
    </tr>
    <tr>
      <th>36</th>
      <td>gill-size_n</td>
      <td>0.053173</td>
    </tr>
    <tr>
      <th>93</th>
      <td>ring-type_p</td>
      <td>0.039158</td>
    </tr>
    <tr>
      <th>91</th>
      <td>ring-type_l</td>
      <td>0.035865</td>
    </tr>
    <tr>
      <th>37</th>
      <td>gill-color_b</td>
      <td>0.035587</td>
    </tr>
    <tr>
      <th>95</th>
      <td>spore-print-color_h</td>
      <td>0.033872</td>
    </tr>
    <tr>
      <th>61</th>
      <td>stalk-surface-below-ring_k</td>
      <td>0.031121</td>
    </tr>
  </tbody>
</table>
</div>

These values were found standard-scaled to the range of 0 to 1. Even after a selection of the top 10 feature, as there was no sign of +ve or -ve from these feature importance values, we would not be able to gauge if these top 10 feature was positively or negatively affecting the mushroom class prediction. To solve this problem, the correlation table generated earlier in Part 2 was used as a reference. 


```python
# re-setup df to reflect which feature importance is positively or negatively affecting the mushroom class prediction
df_feat_imp_correlation_top10_rf = df_feat_imp_top10.merge(df_feature_correlation, how='inner', left_on='features', right_on='features')
df_feat_imp_correlation_top10_rf
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
      <th>features</th>
      <th>feature_importance</th>
      <th>correlation_to_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>odor_n</td>
      <td>0.096636</td>
      <td>-0.785557</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gill-size_b</td>
      <td>0.071668</td>
      <td>-0.540024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.070451</td>
      <td>0.587658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>odor_f</td>
      <td>0.069137</td>
      <td>0.623842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gill-size_n</td>
      <td>0.053173</td>
      <td>0.540024</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ring-type_p</td>
      <td>0.039158</td>
      <td>-0.540469</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ring-type_l</td>
      <td>0.035865</td>
      <td>0.451619</td>
    </tr>
    <tr>
      <th>7</th>
      <td>gill-color_b</td>
      <td>0.035587</td>
      <td>0.538808</td>
    </tr>
    <tr>
      <th>8</th>
      <td>spore-print-color_h</td>
      <td>0.033872</td>
      <td>0.490229</td>
    </tr>
    <tr>
      <th>9</th>
      <td>stalk-surface-below-ring_k</td>
      <td>0.031121</td>
      <td>0.573524</td>
    </tr>
  </tbody>
</table>
</div>


By utilizing the +ve / -ve sign from the `correlation_to_class` column, it was then used as a reference sign to turn the original `feature_important` values into either a positive or negative value. WIth this method, the original weight of the feature important was retained and it was further enhanced to have an indication sign to show whether these features increased or decreased their importance affecting the `class` prediction.

```python
df_feat_imp_correlation_top10_rf['revised_feature_importance'] = np.where(df_feat_imp_correlation_top10_rf['correlation_to_class']<0, -1*df_feat_imp_correlation_top10_rf['feature_importance'], df_feat_imp_correlation_top10_rf['feature_importance'])
df_feat_imp_correlation_top10_rf = df_feat_imp_correlation_top10_rf.sort_values(by='revised_feature_importance', ascending=False)
df_feat_imp_correlation_top10_rf
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
      <th>features</th>
      <th>feature_importance</th>
      <th>correlation_to_class</th>
      <th>revised_feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.070451</td>
      <td>0.587658</td>
      <td>0.070451</td>
    </tr>
    <tr>
      <th>3</th>
      <td>odor_f</td>
      <td>0.069137</td>
      <td>0.623842</td>
      <td>0.069137</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gill-size_n</td>
      <td>0.053173</td>
      <td>0.540024</td>
      <td>0.053173</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ring-type_l</td>
      <td>0.035865</td>
      <td>0.451619</td>
      <td>0.035865</td>
    </tr>
    <tr>
      <th>7</th>
      <td>gill-color_b</td>
      <td>0.035587</td>
      <td>0.538808</td>
      <td>0.035587</td>
    </tr>
    <tr>
      <th>8</th>
      <td>spore-print-color_h</td>
      <td>0.033872</td>
      <td>0.490229</td>
      <td>0.033872</td>
    </tr>
    <tr>
      <th>9</th>
      <td>stalk-surface-below-ring_k</td>
      <td>0.031121</td>
      <td>0.573524</td>
      <td>0.031121</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ring-type_p</td>
      <td>0.039158</td>
      <td>-0.540469</td>
      <td>-0.039158</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gill-size_b</td>
      <td>0.071668</td>
      <td>-0.540024</td>
      <td>-0.071668</td>
    </tr>
    <tr>
      <th>0</th>
      <td>odor_n</td>
      <td>0.096636</td>
      <td>-0.785557</td>
      <td>-0.096636</td>
    </tr>
  </tbody>
</table>
</div>


```python
df_feat_imp_correlation_top10_rf.plot(x='features', y='revised_feature_importance', kind='barh', figsize=(12,6))
plt.title('Top 10 Feature for Mushroom Class Prediction based on Random_Forest_Classifier')
plt.show()
```

<img src="{{ site.baseurl }}/assets/img/portfolio/featImportance_randomF.png">


> * No particular feature showing dominant effect on mushroom class prediction
> * In general, combination of odor, gill-size, ring-type and stalk-surface demonstrated higher effects on final mushroom class prediction


### 4.3 Feature Importance derived from KNeighbors Classifier

As compared to Logistic RegressionCV and Random Forest Classifier, there's no coefficient or feature importance that can be extracted easily from the trained model for Kneighbors Classifier. One way to quantitatively check which feature has greater impact is to perform n_features classification using ONE single feature at a time.


```python
trained_model_kn = joblib.load('./pretrained_models/saved_model_kneighbors_classifier.sav')

features = data_test.columns

master_score_list = []

# iterate through each feature to check cross_val_score to find which features having higher score
for i, feature in enumerate(features):
    data_single_feature = np.array(data_test.iloc[:, i]).reshape(-1, 1)
    score_single_feature = cross_val_score(model_kn, data_single_feature, label_test, cv=3)
    mean_score = np.mean(score_single_feature)
    master_score_list.append(mean_score)
    
df_kNeighbors_feature_score = pd.DataFrame()
df_kNeighbors_feature_score['features'] = data_test.columns
df_kNeighbors_feature_score['cross_val_score_single_feature'] = master_score_list
df_kNeighbors_feature_score_top10 = df_kNeighbors_feature_score.sort_values('cross_val_score_single_feature', ascending=False).head(10)
df_kNeighbors_feature_score_top10
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
      <th>features</th>
      <th>cross_val_score_single_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>odor_n</td>
      <td>0.876126</td>
    </tr>
    <tr>
      <th>24</th>
      <td>odor_f</td>
      <td>0.779325</td>
    </tr>
    <tr>
      <th>61</th>
      <td>stalk-surface-below-ring_k</td>
      <td>0.763336</td>
    </tr>
    <tr>
      <th>57</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>36</th>
      <td>gill-size_n</td>
      <td>0.752266</td>
    </tr>
    <tr>
      <th>35</th>
      <td>gill-size_b</td>
      <td>0.752266</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bruises_t</td>
      <td>0.735438</td>
    </tr>
    <tr>
      <th>58</th>
      <td>stalk-surface-above-ring_s</td>
      <td>0.733388</td>
    </tr>
    <tr>
      <th>37</th>
      <td>gill-color_b</td>
      <td>0.725599</td>
    </tr>
    <tr>
      <th>93</th>
      <td>ring-type_p</td>
      <td>0.677935</td>
    </tr>
  </tbody>
</table>
</div>

</br>

Similar as Random Forest Classifier feature importanct study in Part 4.2, the cross validation score doesn't have a +ve or -ve values. Therefore, the `correlation_to_class` column was merged in order to utilize its sign to convert the cross-val-score values.


```python
# re-setup df to reflect which feature importance is carrying a postive or negative effect
df_feat_score_correlation_top10_kn = df_kNeighbors_feature_score_top10.merge(df_feature_correlation, how='inner', left_on='features', right_on='features')
df_feat_score_correlation_top10_kn
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
      <th>features</th>
      <th>cross_val_score_single_feature</th>
      <th>correlation_to_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>odor_n</td>
      <td>0.876126</td>
      <td>-0.785557</td>
    </tr>
    <tr>
      <th>1</th>
      <td>odor_f</td>
      <td>0.779325</td>
      <td>0.623842</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stalk-surface-below-ring_k</td>
      <td>0.763336</td>
      <td>0.573524</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.763333</td>
      <td>0.587658</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gill-size_n</td>
      <td>0.752266</td>
      <td>0.540024</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gill-size_b</td>
      <td>0.752266</td>
      <td>-0.540024</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bruises_t</td>
      <td>0.735438</td>
      <td>-0.501530</td>
    </tr>
    <tr>
      <th>7</th>
      <td>stalk-surface-above-ring_s</td>
      <td>0.733388</td>
      <td>-0.491314</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gill-color_b</td>
      <td>0.725599</td>
      <td>0.538808</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ring-type_p</td>
      <td>0.677935</td>
      <td>-0.540469</td>
    </tr>
  </tbody>
</table>
</div>



```python
# utilize the correlation values to generate revised feature importances that reflect either having positive or negative effect

df_feat_score_correlation_top10_kn['revised_feature_importance'] = np.where(df_feat_score_correlation_top10_kn['correlation_to_class']<0, -1*df_feat_score_correlation_top10_kn['cross_val_score_single_feature'], df_feat_score_correlation_top10_kn['cross_val_score_single_feature'])
df_feat_score_correlation_top10_kn = df_feat_score_correlation_top10_kn.sort_values(by='revised_feature_importance', ascending=False)
df_feat_score_correlation_top10_kn
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
      <th>features</th>
      <th>cross_val_score_single_feature</th>
      <th>correlation_to_class</th>
      <th>revised_feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>odor_f</td>
      <td>0.779325</td>
      <td>0.623842</td>
      <td>0.779325</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stalk-surface-below-ring_k</td>
      <td>0.763336</td>
      <td>0.573524</td>
      <td>0.763336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stalk-surface-above-ring_k</td>
      <td>0.763333</td>
      <td>0.587658</td>
      <td>0.763333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gill-size_n</td>
      <td>0.752266</td>
      <td>0.540024</td>
      <td>0.752266</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gill-color_b</td>
      <td>0.725599</td>
      <td>0.538808</td>
      <td>0.725599</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ring-type_p</td>
      <td>0.677935</td>
      <td>-0.540469</td>
      <td>-0.677935</td>
    </tr>
    <tr>
      <th>7</th>
      <td>stalk-surface-above-ring_s</td>
      <td>0.733388</td>
      <td>-0.491314</td>
      <td>-0.733388</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bruises_t</td>
      <td>0.735438</td>
      <td>-0.501530</td>
      <td>-0.735438</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gill-size_b</td>
      <td>0.752266</td>
      <td>-0.540024</td>
      <td>-0.752266</td>
    </tr>
    <tr>
      <th>0</th>
      <td>odor_n</td>
      <td>0.876126</td>
      <td>-0.785557</td>
      <td>-0.876126</td>
    </tr>
  </tbody>
</table>
</div>


```python
df_feat_score_correlation_top10_kn.plot(x='features', y='revised_feature_importance', kind='barh', figsize=(12,6))
plt.title('Top 10 Feature for Mushroom Class Prediction based on KNeighbors_Classifier')
plt.show()
```
 
<img src="{{ site.baseurl }}/assets/img/portfolio/featImportance_kNeighbors.png">


> * No particular feature showing dominant effect on mushroom class prediction
> * In general, combination of odor, gill-size and stalk-surface demonstrated higher effects on final mushroom class prediction


### 4.4 Common Top Feature across Difference Models

To find common top features that appear in all 3 models, simply iterate through features and feature-related dataframe

```python
# find common top feature that appears in all 3 models

for feat in list(df_feat_coef_top10_lr['features']):
    if (feat in list(df_feat_imp_correlation_top10_rf['features'])) and \
        (feat in list(df_feat_score_correlation_top10_kn['features'])):
        print(feat)
```


The print results showed common Top Features for all the evaluated models as follows:

    odor_f
    gill-size_b
    odor_n



## __Part 5 : Conclusion__

From the model preformance, all 3 evaluated models were found having good accuracy and F1 score. However, as these models were computed based on different algorithms, the feature importance extracted / identified from the models were therefore different. Consistency was observed in both odor and gill-size features in term of their influencing patterns :

> * odor_f (foul) => positively influece the class prediction [ high tendency output as 'poisonous' ]   
> * gill-size_b (broad) and odor_n (none) => negatively influence the class prediction [ high tendency output as 'edible' ]

For highest safety and precautionary steps, it is still advisable to compare the mushroom class prediction for all 3 models. Only if all 3 models showing the same output predictions, the edibility of the mushroom is then best classified.

<img src="{{ site.baseurl }}/assets/img/portfolio/side_by_side_comparison_3_models_featImp.png">



## __Extras : Deployment__

For fast check on the models and its class prediction, you may check the link [HERE](http://3.14.149.187:8501/) 


