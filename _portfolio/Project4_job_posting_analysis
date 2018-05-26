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



/***

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
    <tr>
      <th>7</th>
      <td>RANDSTAD PTE. LIMITED</td>
      <td>Senior Software Engineer (.NET, C#, Azure)</td>
      <td>West</td>
      <td>Contract ...</td>
      <td>Senior Executive</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td>Government support available</td>
      <td>$5,000to$8,000Monthly</td>
      <td>/job/49be922389893c2a27034c6370419087</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>about the company Our client is a large US MNC...</td>
      <td>skills and experience required  Degree in Soft...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>INFOCHOLA SOLUTIONS PTE. LTD.</td>
      <td>Office &amp; Admin Executive</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Non-executive</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td></td>
      <td>$1,800to$2,400Monthly</td>
      <td>/job/0765e093bbefa05ff55b916328989ef6</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>- Data entry of information - Filing, checking...</td>
      <td>At least 3 years experience in office adminis...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BRITISH-AMERICAN TOBACCO (SINGAPORE) PRIVATE L...</td>
      <td>Supply Planner</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Non-executive</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td>Government support available</td>
      <td>$3,500to$5,500Monthly</td>
      <td>/job/6d790c1132ed11235b0c362f3e6ac407</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Develop and execute following plans by optimis...</td>
      <td>Continuously engage with end markets and Fact...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BRADY ASIA HOLDING PTE. LTD.</td>
      <td>Senior  /  Project Manager</td>
      <td>South</td>
      <td>Permanent</td>
      <td>Senior Executive</td>
      <td>Human Resources</td>
      <td>Government support available</td>
      <td>Salary undisclosed</td>
      <td>/job/e971f1328ea5ca0a255922d93296cddf</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Position Summary   The Project Manager is resp...</td>
      <td>•       Bachelor’s degree in a technical field...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GUILFORD INTERNATIONAL PTE. LTD.</td>
      <td>Administrative Executive</td>
      <td>South</td>
      <td>Permanent</td>
      <td>Senior Executive</td>
      <td>Human Resources</td>
      <td>Government support available</td>
      <td>$3,000to$4,000Monthly</td>
      <td>/job/da21d276f69f2102c4c37b29f615cbd3</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Receiving, Issuing and Dispatching stocks Han...</td>
      <td>At least 3 years in food trading industry Abl...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RGF TALENT SOLUTIONS SINGAPORE PTE. LTD.</td>
      <td>Senior / Recruitment Consultant</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Non-executive</td>
      <td>Admin / Secretarial</td>
      <td></td>
      <td>$3,600to$7,000Monthly</td>
      <td>/job/5cd1465f05ea0cbee85fc3603836c04e</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Are you ready to take the next step in your re...</td>
      <td>What makes a successful RGF Recruitment Consul...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>A*STAR RESEARCH ENTITIES</td>
      <td>Research Engineer / Senior Research Engineer (...</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Non-executive</td>
      <td>Admin / Secretarial</td>
      <td></td>
      <td>$2,500to$5,000Monthly</td>
      <td>/job/5fa2f7fa28285f766f505f3e6f422a59</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>The Agency for Science, Technology and Researc...</td>
      <td>Embedded Systems, Masters or Bachelors   Min...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ROBERT HALF INTERNATIONAL PTE. LTD.</td>
      <td>Research Analyst (1 year contract)</td>
      <td>Central</td>
      <td>Permanent</td>
      <td>Middle Management ...</td>
      <td>Information Technology</td>
      <td></td>
      <td>$4,000to$5,500Monthly</td>
      <td>/job/eb3952661b08815ac0cc7b5e64d4980c</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>The Company Our client is a well-known Consult...</td>
      <td>Your Profile    2+ years of experience in an a...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EIG DERMAL WELLNESS (S) PTE. LTD.</td>
      <td>Marketing Manager (Professional Trade)</td>
      <td>Central</td>
      <td>Permanent</td>
      <td>Middle Management ...</td>
      <td>Information Technology</td>
      <td></td>
      <td>$4,500to$5,500Monthly</td>
      <td>/job/273b55268d8accd7b64105abfcc2926d</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Job Responsibility:       Direct the marketing...</td>
      <td>Candidate must possess minimum Bachelors’ Deg...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>A*STAR RESEARCH ENTITIES</td>
      <td>Senior Research Engineer (ARTC / A*STAR)</td>
      <td>West</td>
      <td>Full Time</td>
      <td>Junior Executive</td>
      <td>Admin / Secretarial</td>
      <td></td>
      <td>$3,400to$6,800Monthly</td>
      <td>/job/3f7def4df899b28b315d1a2b4e856bbb</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>The Agency for Science, Technology and Researc...</td>
      <td>Software Engineering or Computer Engineering...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A*STAR RESEARCH ENTITIES</td>
      <td>Senior Laboratory Officer / Research Engineer ...</td>
      <td>West</td>
      <td>Full Time</td>
      <td>Junior Executive</td>
      <td>Admin / Secretarial</td>
      <td></td>
      <td>$2,100to$4,200Monthly</td>
      <td>/job/928c0ebf2b1dec21d1f2e8e640c05bcc</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>The Agency for Science, Technology and Researc...</td>
      <td>Degree or Diploma in Engineering or Business...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PARK HOTEL ALEXANDRA</td>
      <td>Reservations Sales Executive</td>
      <td>North</td>
      <td>Permanent</td>
      <td>Senior Executive</td>
      <td>Logistics / Supply Chain</td>
      <td></td>
      <td>Salary undisclosed</td>
      <td>/job/5ac945fc414698546efce4edcb67df5e</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>To receive, confirm and process reservations ...</td>
      <td>Excellent communication skills (written &amp; ora...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PKF-CAP CORPORATE SERVICES PTE. LTD.</td>
      <td>Executive Assessment</td>
      <td>North</td>
      <td>Permanent</td>
      <td>Senior Executive</td>
      <td>Logistics / Supply Chain</td>
      <td></td>
      <td>$2,800to$3,500Monthly</td>
      <td>/job/cd8cc2c0e913f64837d10d1c3bb5bfd3</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>You will be responsible for supporting Manager...</td>
      <td>Education and qualifications: • Diploma or equ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TAKASAGO INTERNATIONAL (SINGAPORE) PTE LTD</td>
      <td>Regulatory Affairs Executive</td>
      <td>West</td>
      <td>Full Time</td>
      <td>Executive</td>
      <td>Manufacturing</td>
      <td></td>
      <td>Salary undisclosed</td>
      <td>/job/f0cf73effea1262ad5690c884c1e0cae</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Ensure SDS for products and samples submissio...</td>
      <td>We are looking for a talented candidate and he...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PKF-CAP CORPORATE SERVICES PTE. LTD.</td>
      <td>Assistant Manager</td>
      <td>West</td>
      <td>Full Time</td>
      <td>Executive</td>
      <td>Manufacturing</td>
      <td></td>
      <td>$4,500to$5,000Monthly</td>
      <td>/job/34e8300806014d6450db1b4ffbe73f28</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>You will report to the Head of Assessment and ...</td>
      <td>Education and qualifications: • University deg...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CHERRYLOFT RESORTS AND HOTELS PTE. LTD.</td>
      <td>Accounts Assistant / Executive / Manager</td>
      <td>East</td>
      <td>Permanent</td>
      <td>Manager</td>
      <td>Human Resources</td>
      <td>Government support available</td>
      <td>$1,800to$5,000Monthly</td>
      <td>/job/55363c60ad11ea14be80b3de1caa9609</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Kick start your career in the hospitality indu...</td>
      <td>Minimum GCE “O” Levels/SPM for the Accounts A...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>HR-PRO RECRUITMENT SERVICES PTE. LTD.</td>
      <td>Website Administrator  /  Webmaster</td>
      <td>East</td>
      <td>Permanent</td>
      <td>Manager</td>
      <td>Human Resources</td>
      <td></td>
      <td>$2,300to$2,500Monthly</td>
      <td>/job/29da15c8ac095bf638aba3f804c55e12</td>
      <td>12 Apr 2018</td>
      <td>27 Apr 2018</td>
      <td>Work as part of the team to develop, launch a...</td>
      <td>Diploma in Web Design/ Publishing/ Project Ma...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>OLIVER WYMAN PTE. LTD.</td>
      <td>ITS Security and Risk Analyst</td>
      <td>Islandwide</td>
      <td>Full Time</td>
      <td>Manager ...</td>
      <td>Accounting / Auditing / Taxation</td>
      <td></td>
      <td>$3,000to$6,000Monthly</td>
      <td>/job/c15644a2c0df1498329987ed9a7d6da1</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>As a trusted member of the Information Technol...</td>
      <td>Complete security and technology risk related...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>SANTEN PHARMACEUTICAL ASIA PTE. LTD.</td>
      <td>Regional Senior Logistics &amp; Distribution Execu...</td>
      <td>Islandwide</td>
      <td>Full Time</td>
      <td>Manager ...</td>
      <td>Accounting / Auditing / Taxation</td>
      <td>Government support available</td>
      <td>Salary undisclosed</td>
      <td>/job/e8fcb536afc092d37d47d957479eeb90</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>The Senior Executive will perform operational ...</td>
      <td>Tertiary qualification in a Supply Chain rela...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>A &amp; ONE PRECISION ENGINEERING PTE. LTD.</td>
      <td>SALES MANAGER</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Non-executive</td>
      <td>Design</td>
      <td></td>
      <td>$4,000to$5,000Monthly</td>
      <td>/job/e308a67c9c8545e12bdfb9c7bc59edde</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Formulate business planning, management strat...</td>
      <td>Preferred to have Diploma / Degree in mechani...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>YOJEE PTE. LTD.</td>
      <td>Regional Manager</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Non-executive</td>
      <td>Design</td>
      <td></td>
      <td>$7,000to$10,000Monthly</td>
      <td>/job/1a0a74de21f13e3c84f66bec86a6f318</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>-Manage the day-to-day operations and grow the...</td>
      <td>-Bachelor's degree in a related field -Minimum...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MARITIME TECHNOLOGIES (R&amp;D) PTE. LTD.</td>
      <td>Tech Lead</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Executive</td>
      <td>Information Technology</td>
      <td>Government support available</td>
      <td>$5,000to$8,000Monthly</td>
      <td>/job/a65c945fe3bcd5c14db1ff645ff52c91</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>You think more as a boss than as an employee o...</td>
      <td>BS in Computer Science or a similar field or ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>FLUIDIGM SINGAPORE PTE. LTD.</td>
      <td>Technical Specialist</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Executive</td>
      <td>Information Technology</td>
      <td></td>
      <td>$2,300to$2,800Monthly</td>
      <td>/job/b578e9264ddea5fdc928cc759dd49523</td>
      <td>12 Apr 2018</td>
      <td>12 May 2018</td>
      <td>Title: QC Technical Specialist Summary: The QC...</td>
      <td>Diploma or Bachelor's degree in Science or En...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>374</th>
      <td>EXCELLENCE SINGAPORE PTE. LTD.</td>
      <td>Accounts Executive (FULL SET) (ACCOUNTING FIRM)</td>
      <td>North</td>
      <td>Permanent ...</td>
      <td>Fresh/entry level ...</td>
      <td>Engineering ...</td>
      <td></td>
      <td>$2,200to$3,000Monthly</td>
      <td>/job/45561218d92231d23adb77c771aeb671</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Responsibilities:  Preparing of financial data...</td>
      <td>Requirements:   Training provided Diploma in a...</td>
    </tr>
    <tr>
      <th>375</th>
      <td>WEB PROFESSIONAL HOUSE PTE LTD</td>
      <td>Senior System Analyst</td>
      <td>North</td>
      <td>Permanent ...</td>
      <td>Fresh/entry level ...</td>
      <td>Engineering ...</td>
      <td>Government support available</td>
      <td>$4,500to$7,500Monthly</td>
      <td>/job/b313e830dd4af8c98e3df8558c0abc14</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Main duties and responsibilities of the succes...</td>
      <td>The candidate(s) should possess a Degree in C...</td>
    </tr>
    <tr>
      <th>376</th>
      <td>BRITISH-AMERICAN TOBACCO (SINGAPORE) PRIVATE L...</td>
      <td>Area Supply Planning Manager</td>
      <td>Islandwide</td>
      <td>Contract</td>
      <td>Executive</td>
      <td>Information Technology</td>
      <td>Government support available</td>
      <td>$10,000to$12,500Monthly</td>
      <td>/job/c1cd2ccda25826997db4c1887b9907ef</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Develop and execute a production plan for the ...</td>
      <td>Carry out production planning to develop a th...</td>
    </tr>
    <tr>
      <th>377</th>
      <td>PAIN CLINIC@ WELLNESS PHILOSOPHY PTE. LTD.</td>
      <td>Marketing Assistant</td>
      <td>Islandwide</td>
      <td>Contract</td>
      <td>Executive</td>
      <td>Information Technology</td>
      <td></td>
      <td>$2,300to$2,800Monthly</td>
      <td>/job/fe676feba73ab74950f30c017da912aa</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Assists and supports the marketing team with ...</td>
      <td>Candidate must possess relevant qualification...</td>
    </tr>
    <tr>
      <th>378</th>
      <td>PAIN CLINIC@ WELLNESS PHILOSOPHY PTE. LTD.</td>
      <td>Marketing Assistant</td>
      <td>West</td>
      <td>Permanent ...</td>
      <td>Executive ...</td>
      <td>Accounting / Auditing / Taxation</td>
      <td></td>
      <td>$2,300to$2,800Monthly</td>
      <td>/job/77d31f5e40d5fc1320f979532d015549</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Assists and supports the marketing team with ...</td>
      <td>Candidate must possess relevant qualification...</td>
    </tr>
    <tr>
      <th>379</th>
      <td>GOOGLE ASIA PACIFIC PTE. LTD.</td>
      <td>APAC Strategy and Operations Manager, Google M...</td>
      <td>West</td>
      <td>Permanent ...</td>
      <td>Executive ...</td>
      <td>Accounting / Auditing / Taxation</td>
      <td>Government support available</td>
      <td>$13,500to$22,500Monthly</td>
      <td>/job/9023635a7915ac2852d33d70ab78075f</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Company overview: Google is not a conventional...</td>
      <td>Minimum qualifications:  Bachelor's degree or ...</td>
    </tr>
    <tr>
      <th>380</th>
      <td>A*STAR RESEARCH ENTITIES</td>
      <td>SIgN - Research Fellow (Ren Ee Chee Lab)</td>
      <td>South</td>
      <td>Full Time</td>
      <td>Non-executive</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td>Government support available</td>
      <td>$4,500to$9,000Monthly</td>
      <td>/job/ce7a2bf297f7bf0f661e9edb5a02e729</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>A post-doctoral Research Fellow (Genomics and ...</td>
      <td>PhD in Bioinformatics, Computational Biology ...</td>
    </tr>
    <tr>
      <th>381</th>
      <td>NANYANG TECHNOLOGICAL UNIVERSITY</td>
      <td>Executive</td>
      <td>South</td>
      <td>Full Time</td>
      <td>Non-executive</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td></td>
      <td>Salary undisclosed</td>
      <td>/job/39981f61a7d32f3c864674847a8eaa5e</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Assist the Centre Manager in budget planning ...</td>
      <td>Diploma in Business/Accounting. Must be accou...</td>
    </tr>
    <tr>
      <th>382</th>
      <td>CRIMSONLOGIC PTE LTD</td>
      <td>Lead Systems Engineer</td>
      <td>Central</td>
      <td>Permanent ...</td>
      <td>Junior Executive</td>
      <td>Human Resources</td>
      <td>Government support available</td>
      <td>$4,500to$6,800Monthly</td>
      <td>/job/78245be8a3b2e040efb86bc33ebfcdd3</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Job Purpose   Evaluates, tests, installs, adm...</td>
      <td>Minimum Years/Type of Experience   Minimum 6 ...</td>
    </tr>
    <tr>
      <th>383</th>
      <td>INFO-TECH SYSTEMS INTEGRATORS PTE. LTD.</td>
      <td>Senior Test Engineer</td>
      <td>Central</td>
      <td>Permanent ...</td>
      <td>Junior Executive</td>
      <td>Human Resources</td>
      <td>Government support available</td>
      <td>$4,000to$6,500Monthly</td>
      <td>/job/fc7fa6e74df1651ee490bc8196427a16</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Strong expertise in Test Strategy, Test Data p...</td>
      <td>Candidate must possess at least Bachelor's De...</td>
    </tr>
    <tr>
      <th>384</th>
      <td>AALST CHOCOLATE PTE. LTD.</td>
      <td>Senior VP</td>
      <td>West</td>
      <td>Full Time</td>
      <td>Senior Management ...</td>
      <td>Sales / Retail</td>
      <td>Government support available</td>
      <td>$3,500to$7,000Monthly</td>
      <td>/job/064af390f4d7450f5c46680232524df6</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Definition regional industrial business devel...</td>
      <td>Candidate must possess at least a Bachelor's ...</td>
    </tr>
    <tr>
      <th>385</th>
      <td>MURATA ENERGY DEVICE SINGAPORE PTE. LTD.</td>
      <td>Assistant Engineer (Production)</td>
      <td>West</td>
      <td>Full Time</td>
      <td>Senior Management ...</td>
      <td>Sales / Retail</td>
      <td></td>
      <td>$2,000to$3,000Monthly</td>
      <td>/job/af054704dcbc6028c6c86e6648c66943</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Carry out machine improvement activities to e...</td>
      <td>Diploma in Mechatronics ITE with 3 years expe...</td>
    </tr>
    <tr>
      <th>386</th>
      <td>ETH SINGAPORE SEC LTD.</td>
      <td>Researcher (Natural Capital Project)</td>
      <td>West</td>
      <td>Permanent</td>
      <td>Executive</td>
      <td>Manufacturing</td>
      <td>Government support available</td>
      <td>$6,000to$9,000Monthly</td>
      <td>/job/ac5bdbe1b5887e274ea3f65289fd8d79</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Researcher: Natural Capital Singapore, Future ...</td>
      <td>Key skills The candidate should have/be  MSc, ...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>GAO JI FOOD (S) PTE LTD</td>
      <td>Administrative Executive</td>
      <td>West</td>
      <td>Permanent</td>
      <td>Executive</td>
      <td>Manufacturing</td>
      <td></td>
      <td>$1,800to$2,600Monthly</td>
      <td>/job/28b03596eeddc77f9fe0a66d5d7123cd</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Main Duties &amp; Responsibilities  To assist in t...</td>
      <td>Pre-requisites and Skills:  A team player with...</td>
    </tr>
    <tr>
      <th>388</th>
      <td>JGC SINGAPORE PTE LTD</td>
      <td>LRCC / RCC TURNAROUND PLANNER (1-YEAR CONTRACT...</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Professional</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td>Government support available</td>
      <td>$3,000to$5,000Monthly</td>
      <td>/job/42c8d0293718916b52fb314f59a35ff1</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Job Summary  Reporting to Project Control Man...</td>
      <td>Requirements  Must possess Long residue cataly...</td>
    </tr>
    <tr>
      <th>389</th>
      <td>CONCORDE HOTEL SINGAPORE</td>
      <td>Sales Manager</td>
      <td>South</td>
      <td>Contract ...</td>
      <td>Professional</td>
      <td>Sciences / Laboratory / R&amp;D</td>
      <td></td>
      <td>Salary undisclosed</td>
      <td>/job/5797ee3e437bc01d348fb0825b279337</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>The primary objective of Sales Manager is to b...</td>
      <td>GCE ‘A’ level holder or equivalent qualificat...</td>
    </tr>
    <tr>
      <th>390</th>
      <td>90 SECONDS INTERNATIONAL PTE. LTD.</td>
      <td>Financial Planning &amp; Analysis (FP&amp;A) Manager</td>
      <td>South</td>
      <td>Full Time</td>
      <td>Junior Executive</td>
      <td>F&amp;B</td>
      <td>Government support available</td>
      <td>$8,000to$16,000Monthly</td>
      <td>/job/fc5bceb29d99875c5a94784f42b863f4</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>We are looking for an experienced Financial Pl...</td>
      <td>Someone with a strong sense of urgency, excep...</td>
    </tr>
    <tr>
      <th>391</th>
      <td>AGILITY PROJECT LOGISTICS PTE. LTD.</td>
      <td>Operations Support Executive</td>
      <td>South</td>
      <td>Full Time</td>
      <td>Junior Executive</td>
      <td>F&amp;B</td>
      <td></td>
      <td>$2,200to$3,000Monthly</td>
      <td>/job/d8a0beed24acd792d206e11a9cad4733</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Job Summary This position is to follow establi...</td>
      <td>Requirements  With at least 2 years’ operation...</td>
    </tr>
    <tr>
      <th>392</th>
      <td>ELLIOTT MOSS CONSULTING PTE. LTD.</td>
      <td>SAP Analytics-HANA / SLT Consultant</td>
      <td>West</td>
      <td>Contract</td>
      <td>Executive</td>
      <td>Building and Construction</td>
      <td>Government support available</td>
      <td>$9,000to$12,500Monthly</td>
      <td>/job/8758435958f07c162b4c3e9efcaf608a</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Responsibility for BW work stream as part of ...</td>
      <td>6+ years SAP BW, BW on HANA, BW 7.5 and 2+ ye...</td>
    </tr>
    <tr>
      <th>393</th>
      <td>THE RITZ-CARLTON, MILLENIA SINGAPORE</td>
      <td>Director of Sales - MICE</td>
      <td>West</td>
      <td>Contract</td>
      <td>Executive</td>
      <td>Building and Construction</td>
      <td>Government support available</td>
      <td>Salary undisclosed</td>
      <td>/job/fba2e7d95fca52daf1361fd986f52bda</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>JOB SUMMARY Leads and manages all day-to-day a...</td>
      <td>CANDIDATE PROFILE Education and Experience • 2...</td>
    </tr>
    <tr>
      <th>394</th>
      <td>PESKO ENGINEERING PTE LTD</td>
      <td>Account Assistant</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Manager</td>
      <td>Hospitality</td>
      <td></td>
      <td>$2,000to$3,000Monthly</td>
      <td>/job/66f19cf196547210f970bf38331767f0</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>·       Data entry of financial records in t...</td>
      <td>·       Diploma, LCCI, CAT or equivalent. · ...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>WORLDQUANT (SINGAPORE) PTE. LTD.</td>
      <td>Execution Trader</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Manager</td>
      <td>Hospitality</td>
      <td></td>
      <td>$8,000to$12,000Monthly</td>
      <td>/job/6e88ce6feb6c4d2537999ae156550cce</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Handle execution across Fixed Income Products,...</td>
      <td> Degree in a quantitative or technical discip...</td>
    </tr>
    <tr>
      <th>396</th>
      <td>ALMAC PHARMACEUTICAL SERVICES PTE. LTD.</td>
      <td>Pharmaceutical Technical Support Representative</td>
      <td>Central</td>
      <td>Permanent ...</td>
      <td>Middle Management</td>
      <td>Banking and Finance</td>
      <td>Government support available</td>
      <td>Salary undisclosed</td>
      <td>/job/ece72799bd4afd2501988f08f5223efb</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>The Technical Support Representative will prov...</td>
      <td>Criteria  Service-oriented Knowledge of MS su...</td>
    </tr>
    <tr>
      <th>397</th>
      <td>SOFTLAYER TECHNOLOGIES ASIA PRIVATE LIMITED</td>
      <td>Inventory Technician</td>
      <td>Central</td>
      <td>Permanent ...</td>
      <td>Middle Management</td>
      <td>Banking and Finance</td>
      <td></td>
      <td>$2,300to$2,500Monthly</td>
      <td>/job/c555d461f8e975739aaf54c8bbb5280c</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Computer hardware component identification En...</td>
      <td>Focused work ethic in a team environment Abil...</td>
    </tr>
    <tr>
      <th>398</th>
      <td>NEWMEDIA EXPRESS PTE. LTD.</td>
      <td>Internet Engineer</td>
      <td>East</td>
      <td>Permanent</td>
      <td>Executive</td>
      <td>Logistics / Supply Chain</td>
      <td></td>
      <td>$1,800to$4,000Monthly</td>
      <td>/job/0cf4ec37ba723b4e2233e8c6d658d121</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Develop and administer internal web-based app...</td>
      <td>Diploma / Degree in Computer Science / Inform...</td>
    </tr>
    <tr>
      <th>399</th>
      <td>MURATA ENERGY DEVICE SINGAPORE PTE. LTD.</td>
      <td>Assistant Engineer</td>
      <td>East</td>
      <td>Permanent</td>
      <td>Executive</td>
      <td>Logistics / Supply Chain</td>
      <td></td>
      <td>$2,200to$3,200Monthly</td>
      <td>/job/3093c549409f4854009b380ebd0b359a</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Carry out machine maintenance and improvement...</td>
      <td>Assistant Engineer level – Diploma in Electri...</td>
    </tr>
    <tr>
      <th>400</th>
      <td>MEDIACORP PTE. LTD.</td>
      <td>UX Designer</td>
      <td>North, Central</td>
      <td>Contract</td>
      <td>Professional</td>
      <td>Information Technology</td>
      <td></td>
      <td>$4,500to$6,500Monthly</td>
      <td>/job/08a2b2f8b2e6b43974db4a050bc2c8b7</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>You will be deeply involved in our company wid...</td>
      <td>A Bachelor's Degree in a related field, such ...</td>
    </tr>
    <tr>
      <th>401</th>
      <td>APBA PTE. LTD.</td>
      <td>Senior Build and Release Engineer</td>
      <td>North, Central</td>
      <td>Contract</td>
      <td>Professional</td>
      <td>Information Technology</td>
      <td>Government support available</td>
      <td>$5,000to$8,000Monthly</td>
      <td>/job/aa1480415af5146f5312ac7145e61aa4</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Partner with project leaders and delivery tea...</td>
      <td>Degree/Diploma in IT or relevant discipline. ...</td>
    </tr>
    <tr>
      <th>402</th>
      <td>DELIVEROO SINGAPORE PTE. LTD.</td>
      <td>Corporate Account Manager</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Manager</td>
      <td>Hospitality</td>
      <td>Government support available</td>
      <td>$2,500to$5,000Monthly</td>
      <td>/job/a5abb52963d0e60831e1afd05543c41a</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Our mission is to bring the world's best-loved...</td>
      <td>The ideal candidate has a natural commercial a...</td>
    </tr>
    <tr>
      <th>403</th>
      <td>APBA PTE. LTD.</td>
      <td>Software QA Engineer</td>
      <td>Central</td>
      <td>Full Time</td>
      <td>Manager</td>
      <td>Hospitality</td>
      <td></td>
      <td>$4,000to$6,000Monthly</td>
      <td>/job/b6f923e9c2353305eae03eceb80ce2dd</td>
      <td>09 Apr 2018</td>
      <td>09 May 2018</td>
      <td>Collaborate with product managers, software d...</td>
      <td>3+ years relevant experience in software test...</td>
    </tr>
  </tbody>
</table>
***/






{% highlight js %}

{% endhighlight %}
<img src="{{ site.baseurl }}/assets/img/portfolio/Vectorizer_counts.jpeg" width="600" height="420">
