---
layout: post
title: Analysis on Job Postings
img: "assets/img/portfolio/job_icon.png"
date: May, 26 2018

---

![image]({{ site.baseurl }}/{{ page.img }})

Every year, there are thousands of graduates from different institutions coming fresh into the job market, looking for 
the jobs that best fit them in terms of qualification, aspiration and salary expectation. There are also people changing 
their career paths, looking for the ones that they can leverage their previous experience and knowledge into better use.
Similarly, employers are also hunting for the right candidates to fill the positions hoping to get them on board to help
growing the business together with an aligned visions. In view of these bi-directional demands and needs, how to have the consolidated solutions in order to stay competitive for both parties to get the right fit for each other become an interesting topic to study.

This project outlined the focus by studying two main aspects:
### __Part 1 : Factors that impact salary__

   * which are significant decisive factors for both job seeker and employer during the consideration of an acceptance / offer


### __Part 2 : Factors that distinguish the job category__

   - which are essential for getting the right candidates with the most fit qualification and experiences
, matching appropriately to the job requirements or roles and responsibilities



<a id="data_source"></a>

## Data Source
---

Job Postings @ [MyCareersFuture](https://www.mycareersfuture.sg)



<a id="analytical_approach"></a>

## Analytical Approach
---

### 1. Data Collection via Web-Scrapping
* Collect job postings that are data-related and scrap at least 1000 data to get the relevant information that is needed for subsequent analysis and prediction


### 2. Data Wrangling & Preparation
* Parse web-scrapped data and prepare them into dataframe for easy processing
* Perform data cleaning to clear doubtful entries and tranform into standardized types


### 3. Exploratory Data Analysis
* Set mean salary as the key predictive feature
* Check salary distribution and remove outliers >15k for senior positions like HOD, director in order to have normally distributed data for better prediction outcomes as general norms

### 4. Pre-Processing & Predictive Model Selection
* Analysing and transforming textual information using Natural Language Processing packages
* Evaluation of various classification models for best predictive model selection

### 5. Outlining Features Importance
* Summarizing the overall features that hold greatest significance in terms of Salary Prediction and Job Category Classification


<a id="analytical_approach"></a>

## Analytical Outcomes :
---

##  Web-Scrapping
Using BeautifulSoup and Selenium, relevant job postings linked were collected

Part 1 : to get basic job data info

{% highlight js %}
driver = webdriver.Firefox(executable_path='./geckodriver')
compiled_data = []
for page in range(0,20):   
    url = "https://www.mycareersfuture.sg/search?search=data&page={}".format(page)
    
    # Visit relevant page.b
    driver.get(url)

    # Wait few second.
    sleep(3)

    # Grab the page source.
    html = driver.page_source
    # print(html)

    soup = BeautifulSoup(html, 'lxml')
    compiled_data.append(list(jobPostingInfo(soup)))
driver.close()
{% endhighlight %}


Part 2 : to get job desc + role & responsibility info

{% highlight js %}
driver = webdriver.Firefox(executable_path='./geckodriver')
addOn_data = []
for i in range(0, len(compiled_data)):
    page_temp = []
    for j in range(20): # one page only has 20 records
        
        temp = []
        joblink = compiled_data[i][8][j]
        temp.append(joblink)
        
        joblink_url = "https://www.mycareersfuture.sg"+joblink
     
        # Visit relevant page.
        driver.get(joblink_url)

        # Wait few second.
        sleep(3)

        # Grab the page source.
        html = driver.page_source
        # print(html)

        soup2 = BeautifulSoup(html, 'lxml')
        
        postDate = soup2.find('span',{'id':'last_posted_date'}).text
        temp.append(postDate)       
        closeDate = soup2.find('span',{'id':'expiry_date'}).text
        temp.append(closeDate)
        job_content = soup2.find_all('div',{'id':'content'})
        role_resp = job_content[0].text
        temp.append(role_resp)
        
        try:
            requirements = job_content[1].text
            temp.append(requirements)
        except:
            temp.append('-')
   
        page_temp.append(temp)
    
    addOn_data.append(page_temp)

driver.close()
{% endhighlight %}

Example of the consolidated dataframe storing partial web-scrapped data

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>JobTitle</th>
      <th>Location</th>
      <th>EmploymentType</th>
      <th>Seniority</th>
      <th>Category</th>
      <th>GovSupport</th>
      <th>SalaryRange</th>
      <th>JobLink</th>
      <th>PostedDate</th>
      <th>ClosingDate</th>
      <th>RoleResponsibility</th>
      <th>Requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ASPIRE GLOBAL NETWORK PTE. LTD.</td>
      <td>Regional Head, Ad Operations</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Senior Management</td>
      <td>Admin / Secretarial</td>
      <td></td>
      <td>$9,000to$13,500Monthly</td>
      <td>/job/bde211a5a9f2b9cef2115aa6e8104a36</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>A global broadcast and entertainment giant is ...</td>
      <td>Requirements  Min 6 years’ experience with Str...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CYRUS TECHNOLOGY (S) PTE. LTD.</td>
      <td>program executive</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Senior Management</td>
      <td>Admin / Secretarial</td>
      <td></td>
      <td>$2,000to$2,400Monthly</td>
      <td>/job/0662cb940442cd31a25989972255c676</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Handle daily enquiries and requests from clie...</td>
      <td>Candidate possesses Diploma in any discipline...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NATIONAL UNIVERSITY HOSPITAL (SINGAPORE) PTE LTD</td>
      <td>Case Management Officer_RCCM (Contract)</td>
      <td>East, Central</td>
      <td>Permanent ...</td>
      <td>Executive</td>
      <td>Admin / Secretarial ...</td>
      <td>Government support available</td>
      <td>$2,800to$5,600Monthly</td>
      <td>/job/ef766282d386e151e6b0b863dbbf1d25</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>The case management officer reviews, assessing...</td>
      <td>Qualification:  Diploma or Degree in nursing o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A*STAR RESEARCH ENTITIES</td>
      <td>Scientist / Senior Scientist (ARTC / A*STAR)</td>
      <td>East, Central</td>
      <td>Permanent ...</td>
      <td>Executive</td>
      <td>Admin / Secretarial ...</td>
      <td></td>
      <td>$5,900to$11,800Monthly</td>
      <td>/job/1f70879985e4d7b0c506434c2beb82ee</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>The Agency for Science, Technology and Researc...</td>
      <td>Data Scientist (SMG) Senior (at the level of...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A*STAR RESEARCH ENTITIES</td>
      <td>IMCB - Research Manager (JEC)</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Executive</td>
      <td>Healthcare / Pharmaceutical</td>
      <td></td>
      <td>$6,300to$12,600Monthly</td>
      <td>/job/c9c34298cd0dc645e3d6edb902945a4c</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>About the Institute of Molecular and Cell Biol...</td>
      <td>Possess MSc in Medical Biochemistry Minimum 5...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TEEKAY MARINE (SINGAPORE) PTE. LTD.</td>
      <td>Marine Personnel Officer</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Executive</td>
      <td>Healthcare / Pharmaceutical</td>
      <td>Government support available</td>
      <td>Salary undisclosed</td>
      <td>/job/53701b418e11e3c29cacdc01b483df97</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Position Summary The Marine Personnel Officer ...</td>
      <td>Diploma in Maritime/Business Administration w...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ST RECRUITMENT CENTRE</td>
      <td>Admin Assistant</td>
      <td>West</td>
      <td>Contract ...</td>
      <td>Senior Executive</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td></td>
      <td>$1,800to$2,500Monthly</td>
      <td>/job/1ac5d9d42402ff2938608515ebf9923f</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>To issue purchase. Do data entry. Perform sto...</td>
      <td>Minimum GCE 'O' level. Knowledge in Excel Spr...</td>
    </tr>
  </tbody>
</table>







{% highlight js %}

{% endhighlight %}
<img src="{{ site.baseurl }}/assets/img/portfolio/Vectorizer_counts.jpeg" width="600" height="420">
