---
layout: post
title: Where's the UNICORN ?
img: "assets/img/portfolio/unicorn.png"
date: May, 18 2018
tags: [Lorem]
---

![image]({{ site.baseurl }}/{{ page.img }})

In the domain of analysing company growth and its potential, the conventional way is normally conducted by checking the comprehensive financial reports and operation status. However, to many who do not really hold lots of business insights and sufficient financial figures, is there a simpler but yet statistically studied method that they can use as a baseline reference ?

The objective of the project is to attempt using limited information and small set of features to predict the company potential if the company will eventually go for IPO or acquired by other market players. It is not meant to serve as any financial or investment advice but more a project to exercise and integrate various modelling trials to assist the preliminary assessment decision in a more structure and statistically explainable approach.

The outcomes of the data analysis and model prediction provide moderate information on the competitive landscape across various market sectors while modelling the growth potential of various companies that were in focus by the Top 10 key investors. 



## Goal
---

To identify high growth company & predict its potential for successful IPO / acquisition
![Goal_ipo_acquired logo]({{ site.baseurl }}/assets/img/portfolio/Goal_ipo_acquired.jpeg)


<a id="success"></a>

## Success Metrics
---

* Focusing on True Positive Rate [ Sensitivity ] 
    * Company predicted as Acquired/IPO does get acquired/IPO in real world
    * Target [Recall Score](https://en.wikipedia.org/wiki/Precision_and_recall) for True Positive > 70%


* Dianogstic of Binary Classification Performance
    * To assess using Receiver Operating Characteristic curve ( [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) ) , a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied
    * Target ROC_AUC score > 70%


![Success_metrics logo]({{ site.baseurl }}/assets/img/portfolio/Success_metrics.jpeg)


<a id="data_source"></a>

## Data Source
---

* Raw dataset from Crunchbase @ 2015
* Missing values handling via Web-scrapped data 
* Validation via Web-scapped / direct webpage data @ 2018


<a id="analytical_approach"></a>

## Analytical Approach
---

#### 1. Scope Definition to Focus on Market Sectors of High Interest
Since available data set was back to 2015, start up companies to focus for the analysis shall not be funded too many years back.

* Criteria 1: Limit scope to companies with last round of funding obtained at 2010 and beyond
* Criteria 2: Restrict analysis to only funding sources coming from:
    * venture capitalist
    * seed funding
    * angel investor
    * private_equity
* Criteria 3: Extraction of data involving investments done by top 10 investors

#### 2. Thorough Data Mining & Cleaning
Data cleansing and filtration based on scopes defined and perform preliminary basic analysis to get the most relevant data out for next level of details analysis

#### 3. Exploratory Data Analysis
* Extraction of Top10 Key Investors based on the Total Investment Amount in USD
* Reclassification of Market Sector and group scatterred sectors into the closest business fields
* Similarity of Investment Porfolio via Network Graph
* Missing Values Handling and Replacement via Web-Scrapping
* Preliminary Statistical Analysis for
    * Acquired companies
    * IPO companies
    * Closed down companies
                            
#### 4. Sampling Methods & Predictive Model Selection
* Evaluation of various sampling methods for Imbalanced datasets
* Evaluation of various classification models for best predictive model selection

#### 5. Data Validation for Predictive Model Improvement
* Validation of False Positive data versus latest state in 2018
* Recalculation with latest state to check the actual predictive model performance


<a id="analytical_approach"></a>

## Analytical Outcomes
---

#### 1. Finalization of Top10 Key Investors

<img src="{{ site.baseurl }}/assets/img/portfolio/Top10_investors_table.jpeg" width="600" height="420">


{% highlight js %}
// top 10 investors by amount raised_usd

df_top10_investors.sort_values('Raised Amount Usd', ascending=True).plot(x='Investor Name', y='Raised Amount Usd', kind='barh', figsize=(15,4), colormap='plasma')
plt.title('Total Investment Amount in USD by Top 10 Investors')
plt.show()
{% endhighlight %}


<img src="{{ site.baseurl }}/assets/img/portfolio/Top10_investors_graph.jpeg" width="5000" height="400">


#### 2. Market Sectors In Focus by Top10 Key Investors

{% highlight js %}
// attempt to use TF-IDF vectorizer on company category to check which key words have higher importance/appearance

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tvec = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1,2), max_features=1000)
// use ngram(1,2) because category is more relevant when mentioning in single (software), or pair words (consumer electronics) 

tvec.fit(df_top10investor_choice['company_category_list_2'])
df_catModified_tvec = pd.DataFrame(tvec.transform(df_top10investor_choice['company_category_list_2']).todense(), columns=['Category_'+ v for v in tvec.get_feature_names()], index=df_top10investor_choice['company_category_list_2'].index)

wordcount = df_catModified_tvec.sum().sort_values(ascending=False)
df_wordcount = pd.DataFrame(wordcount)
df_wordcount = df_wordcount.reset_index()
df_wordcount.rename(columns={'index': 'Feature', 0: 'VectorizerCount'}, inplace=True)
df_wordcount
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Vectorizer_counts.jpeg" width="600" height="420">


{% highlight js %}
// market sector sorted by amount raised_usd from top 10 investors

df_MarketSector.sort_values('Raised Amount Usd', ascending=True).plot(y='Raised Amount Usd', x='Market_Sector', kind='barh', figsize=(15,10),colormap='tab20c')
plt.title('Total Investment Amount in USD by Market Sector')
plt.show()
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Market_sector_in_focus.jpeg" width="800" height="420">


#### 3. Similarity of Investment Portfolio via Network Graph
* 1st Visualization via networkx library

{% highlight js %}
// refine network graph to have visibility on Market Sectors & the common interest among various top 10 investors. An overview that includes investors' path.

import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()

// set up nodes
for node in nodes:
    G.add_node(node)

// set up edges
edlabel = []
graph_tuple = []
for i,r in edges_df.iterrows():
    G.add_edge(r['Source'],r['Target'])
    edlabel.append(r['Edge_labels'])
    graph_tuple.append((r['Source'], r['Target']))
    
// set up edges_labels
edge_labels = dict(zip(graph_tuple, edlabel))

// network graph plot
plt.figure(figsize=(18,12))
pos = nx.shell_layout(G)

nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos,labels=nodes_label)
nx.draw_networkx_edge_labels(G,pos, edge_labels=edge_labels)
plt.show()
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Network_graph.jpeg" width="2000" height="600">


* 2nd Visualization via Neo4j Graph Platform

{% highlight js %}
// codes used in Neo4j for interactive graph plotting

CREATE CONSTRAINT ON (i:Investor) ASSERT i.inv_name IS UNIQUE;
CREATE CONSTRAINT ON (m:MktSector) ASSERT m.sector IS UNIQUE;

LOAD CSV WITH HEADERS FROM "file:///15_df_neo_final.csv" AS line
WITH line
MERGE (i:Investor {inv_name: line.Investor_name})
MERGE (m:MktSector {sector: line.Market_Sector_neo})
CREATE (i)-[:INVESTED_IN]->(m);
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Neo4j_graph.jpeg" width="2000" height="600">


## Analysis on Acquired Companies

{% highlight js %}
sns.set(style="ticks")

// Funding Total in USD
x1 = startup_2010onwards[~(startup_2010onwards['funding_total_usd']==0)]['funding_total_usd']
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(x1, ax=ax_box, color='#e77c7c')
sns.distplot(x1, ax=ax_hist, color='#e77c7c')

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

// funding rounds
x2 = startup_2010onwards['funding_rounds']
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(x2, ax=ax_box)
sns.distplot(x2, ax=ax_hist)

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

// No.of Investors from Top10 Investors
x3 = startup_2010onwards['Investedby Top10']
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(x3, ax=ax_box, color='#388E3C')
sns.distplot(x3, ax=ax_hist, color='#388E3C')

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

// years_to_exit
x4 = startup_2010onwards['years_to_exit']
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                    gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(x4, ax=ax_box, color='#bd7ee1')
sns.distplot(x4, ax=ax_hist, color='#bd7ee1')

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Stats_acquired_comp_overview.jpeg" width="1000" height="600">


{% highlight js %}
// to check which market sector having the highest chance of being acquired

df_acquired_sortby_MS.sort_values('No_acquired_comp',ascending=True).plot(x='Market_Sector',y='No_acquired_comp', kind='barh', figsize=(15,7), colormap='tab10')
plt.title('No. of Company Exited due to Acquisition')
plt.show()
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Acquired_by_mktSec_graph.jpeg" width="1000" height="450">

{% highlight js %}
df_acquired_sortby_MS = startup_2010onwards.groupby('Market_Sector')['name'].count().to_frame()
df_acquired_sortby_MS.reset_index(inplace=True)
df_acquired_sortby_MS.rename(columns={'name':'No_acquired_comp'}, inplace=True)
df_acquired_sortby_MS.sort_values('No_acquired_comp', ascending=False, inplace=True)
df_acquired_sortby_MS
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Acquired_by_mktSec_table.jpeg" width="600" height="420">


### Summary of companies under Top10 investor list that were exited due to acquisition :

* 196 / 1675 companies from top 10 investors list was acquired 
* 11.70% acquisition successful rate

#### For companies founded since 1994 

Average | Median
------- | ------
5.63 years to exit                 | 5 years
3.09 funding_rounds                | 3
1.56 investor from Top10 investors | 1
\$66.67M total funding             | $30M


---
#### For start-ups founded 2010 onwards  [88 companies]

Average | Median
------- | ------
2.95 years to exit                   | 3 years
2.25 funding_rounds                  | 2
1.42 investor from Top10 investors   | 1
\$35.79M total funding               | $8.25M

---
#### Top 3 market sectors with higher no. of companies being acquired 

* Software\\Apps
* Content Creation\\Entertainment\Curated Web\Design
* Internet\\Web\Search\Communication\Social Media

## Analysis on IPO Companies
Similar stats analysis approach as done for Acquired companies was repeated and used to study IPO companies. Below are the graphical analytic outcomes for IPO companies :


<img src="{{ site.baseurl }}/assets/img/portfolio/Stats_ipo_comp_overview.jpeg" width="1000" height="600">

{% highlight js %}
// to check which market sector having the highest chance of going for IPO

df_ipo_sortby_MS.sort_values('No._IPO_Company',ascending=True).plot(x='Market_Sector',y='No._IPO_Company', kind='barh', figsize=(15,6), colormap='tab10')
plt.title('No. of IPO Companies by Market Sector')
plt.show()
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Ipo_by_mktSec_graph.jpeg" width="1000" height="450">

{% highlight js %}
df_ipo_sortby_MS = df_ipo.groupby('Market_Sector')['Company Name'].count().to_frame()
df_ipo_sortby_MS.reset_index(inplace=True)
df_ipo_sortby_MS.rename(columns={'Company Name':'No._IPO_Company'}, inplace=True)
df_ipo_sortby_MS.sort_values('No._IPO_Company', ascending=False, inplace=True)
df_ipo_sortby_MS
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Ipo_by_mktSec_table.jpeg" width="600" height="420">

{% highlight js %}
// to check which IPO companies having the highest funding_total

df_ipo_sortby_MS.sort_values('Funding_Total_Usd',ascending=True).plot(x='Company Name',y='Funding_Total_Usd', kind='barh', figsize=(15,30), colormap='tab10')
plt.title('Funding Total in USD by IPO Company')
plt.show()
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Ipo_by_comp_graph.jpeg" width="1200" height="380">

<img src="{{ site.baseurl }}/assets/img/portfolio/Ipo_by_comp_table.jpeg" width="560" height="380">

### Summary of companies under Top10 investor list that were IPO successfully :

* 71 / 1674 companies were successfully listed
* 4.24% successful IPO rate

#### From 71 IPO companies :

Average | Median
------- | ------
\$332.91M total funding              | $112.79M
5.51 funding_rounds                  | 5
1.94 investor from Top10 investors   | 1
2005 years to exit                   | 2006

---
#### Top 3 Market Sectors having the highest no. of IPO companies:

- BioTech\\Health Care    : 25
- Software\\Apps          : 9
- eCommerce\\Marketplace  : 7

---
#### Top 3 IPO companies having the highest funding total:



Company | Total_Funding
------- | -------------
1st Alibaba   | \$4.81 Billion
2nd Facebook  | \$2.43 Billion
3rd Twitter   | \$1.16 Billion

## Analysis on Closed Down Companies
Similar stats analysis approach was again repeated and used to study Closed Down companies. Below are the graphical analytic outcomes for Closed Down companies :

<img src="{{ site.baseurl }}/assets/img/portfolio/Stats_closed_comp_overview.jpeg" width="1000" height="600">

{% highlight js %}
// to check which market sector having the highest no. of closed companies

df_closed_sortby_MS.sort_values('No._closed_Company',ascending=True).plot(x='Market_Sector',y='No._closed_Company', kind='barh', figsize=(15,6), colormap='tab10')
plt.title('No. of Closed Down Companies by Market Sector')
plt.show()

{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Closed_by_mktSec_graph.jpeg" width="1000" height="450">

{% highlight js %}
df_closed_sortby_MS = df_closed.groupby('Market_Sector')['Company Name'].count().to_frame()
df_closed_sortby_MS.reset_index(inplace=True)
df_closed_sortby_MS.rename(columns={'Company Name':'No._closed_Company'}, inplace=True)
df_closed_sortby_MS.sort_values('No._closed_Company', ascending=False, inplace=True)
df_closed_sortby_MS
{% endhighlight %}

<img src="{{ site.baseurl }}/assets/img/portfolio/Closed_by_mktSec_table.jpeg" width="600" height="380">

### Summary of companies under Top10 investor list that were closed down :


* 52 / 1674 companies were closed down
* 3.11% failure rate
   
---
#### From 52 closed down companies :

Average | Median
------- | ------
\$28.66M total funding              | $12.15M
2.56 funding_rounds                 | 2
1.42 investor from Top10 investors  | 1
2008 years to exit                  | 2009

[ 2008 Subprime Mortgage Financial Crisis ]

---
#### Top 3 Market Sectors having the highest no. of Closed Down companies:

- eCommerce\\Marketplace                               : 10
- ContentCreation\\Entertainment\Curated Web|Design    : 7
- BioTech\\Health Care                                 : 6
- Software\\Apps                                       : 6


<a id="ModelSelection"></a>

## Sampling Methods & Predictive Model Selection
---

Current data set was found not suitable for a comprehensive Time Series Analysis due to the lack of important key figure of Date/Time and its completeness to support continuous time-dependent analysis. With this, pior to starting next core modelling, a new dataframe was set up to exclude companies with Closed Down status and companies that were either acquired or IPO were then combined as one category. The restructure of such dataframe was done in preparation for classification modelling purpose. 

* Restructure 1: Exclude companies with Closed Down status
* Restructure 2: Combine acquired companies and IPO companies as one new category


### 1. Dataframe Readiness

{% highlight js %}
// Create X matrix

X = pd.concat([df_model_2clusters.drop(['Company Name','Market_Sector','Is_Acquired_IPO'], axis=1),mkt_sec], axis=1)
print(X.shape)
X
{% endhighlight %}


{% highlight js %}
// create y

y = df_model_2clusters['Is_Acquired_IPO']
y.value_counts()

'''
0    1337
1     266
Name: Is_Acquired_IPO, dtype: int64
'''
// imbalance dataset observed

{% endhighlight %}


