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

The outcomes of the data analysis and model prediction provide moderate information on the competitive landscape across various market sectors to mitigate risks while modelling the growth potential of various companies in focus by the Top 10 key investors. 



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

### 1. Scope Definition to Limit Exploration towards Market Sectors of High Interest Among Key Investors
Since available data set was back to 2015, start up companies to focus for the analysis shall not be funded too many years back.
* Criteria 1: Limit scope to companies with last round of funding obtained at 2010 and beyond
* Criteria 2: Restrict analysis to only funding sources coming from:
              * venture capitalist
              * seed funding
              * angel investor
              * private_equity
* Criteria 3: Extraction of data involving investments done by top 10 investors

### 2. Thorough Data Mining & Cleaning
Data cleansing and filtration based on scopes defined and perform preliminary basic analysis to get the most relevant data out for next level of details analysis

### 3. Exploratory Data Analysis
* Extration of Top10 Key Investors based on the Total Investment Amount in USD
* Reclassification of Market Sector to simplify and group scatterred sectors into the closest business fields
* Similarity of Investment Porfolio via Network Graph
* Missing Values Handling and Replacement via Web-Scrapping
* Analytics and Stats for
              * Acquired companies
              * IPO companies
              * Closed down companies
                            
### 4. Sampling Methods & Predictive Model Selection
* Evaluation of various sampling methods for Imbalanced datasets
* Evaluation of various classification models for best predictive model selection

### 5. Data Validation for Predictive Model Improvement
* Validation of False Positive data versus latest state in 2018
* Recalculation with latest state to check the actual predictive model performance


<a id="analytical_approach"></a>

## Analytical Outcomes
---

### 1. Finalization of Top10 Key Investors

{% highlight js %}

df_top10_investors = df_top10_investors[['Investor Name', 'Raised Amount Usd']]
df_top10_investors

{% endhighlight %}


<img src="{{ site.baseurl }}/assets/img/portfolio/Top10_investors_table.jpeg" width="660" height="580">


{% highlight js %}
// top 10 investors by amount raised_usd

df_top10_investors.sort_values('Raised Amount Usd', ascending=True).plot(x='Investor Name', y='Raised Amount Usd', kind='barh', figsize=(15,4), colormap='plasma')
plt.title('Total Investment Amount in USD by Top 10 Investors')
plt.show()
{% endhighlight %}

![Top10_investors_graph logo]({{ site.baseurl }}/assets/img/portfolio/Top10_investors_graph.jpeg)

### 2. Market Sectors In Focus by Top10 Key Investors


{% highlight js %}
// attempt to use TF-IDF vectorizer on company category to check which key words are higher importance/appearance
// for better decision making to map the most relevant ones together as one new market sector

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tvec = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1,2), max_features=1000)
// use ngram(1,2) because category more relevant when mentioning in single (software), or pair words (consumer electronics) 

tvec.fit(df_top10investor_choice['company_category_list_2'])
df_catModified_tvec = pd.DataFrame(tvec.transform(df_top10investor_choice['company_category_list_2']).todense(), columns=['Category_'+ v for v in tvec.get_feature_names()], index=df_top10investor_choice['company_category_list_2'].index)
// to add prefix of Category in column name, so it is clearer that these features are originally from category column
// when putting the post TF-IDF processed data into new dataframe later on combining with other post TF_IDF data
// to include index=df_top10investor_choice['company_category_list_2'].index in order to have the index consistent and avoid mismatch of index
// when using pd.concat with other df later on

df_catModified_tvec.sum().sort_values(ascending=False)

wordcount = df_catModified_tvec.sum().sort_values(ascending=False)
df_wordcount = pd.DataFrame(wordcount)
df_wordcount = df_wordcount.reset_index()
df_wordcount.rename(columns={'index': 'Feature', 0: 'VectorizerCount'}, inplace=True)
df_wordcount
{% endhighlight %}

![Vectorizer_counts logo]({{ site.baseurl }}/assets/img/portfolio/Vectorizer_counts.jpeg)


{% highlight js %}
// market sector sorted by amount raised_usd from top 10 investors

df_MarketSector.sort_values('Raised Amount Usd', ascending=True).plot(y='Raised Amount Usd', x='Market_Sector', kind='barh', figsize=(15,10),colormap='tab20c')
plt.title('Total Investment Amount in USD by Market Sector')
plt.show()
{% endhighlight %}

![Market_sector_in_focus logo]({{ site.baseurl }}/assets/img/portfolio/Market_sector_in_focus.jpeg)

### 3. Similarity of Investment Portfolio via Network Graph


{% highlight js %}
// refine network graph to have visibility on Market Sectors & the common interest among various top 10 investors
// overview including investor path

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

![Network_graph logo]({{ site.baseurl }}/assets/img/portfolio/Network_graph.jpeg)
