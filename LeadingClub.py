# coding=utf-8


import tensorflow as tf 
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 

from plotly import tools 
import plotly.plotly as py 
import plotly.figure_factory as ff 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected = True )

from imblearn.over_sampling import SMOTE 	#SMOTE对少数类样本进行分析并合成新的样本加入dataset，来防止过拟合  Smote_Algo.py中有
from collections import Counter

import time 

df = pd.read_csv(r"C:\Users\msi\Desktop\loan.csv", low_memory = False)

original_df = df.copy() 

#df.head()
#df.info()

#先概览整体数据分布

fig,ax = plt.subplots(1,3,figsize=(16,5))

loan_amount  = df["loan_amount"].values 
funded_amount = df["funded_amount"].values 
investor_funds = df["investor_funds"].values

sns.displot(loan_amount, ax = ax[0], color = "#7522F")
ax[0].set_title("Loan Applied by the Borrower", fontsize = 14)	#fontsize字体大小
sns.displot(funded_amount, ax = ax[1], color = "2F8FF7")
ax[1].set_title("Amount Funded by the Lender",fontsize = 14)
sns.displot(investor_funds,ax =ax[2], color = "#2EAD46")
ax[2].set_title("Total committed by Investors", fontsize=14)

#上面三图观察发现三者分布相似


#年份与贷款之间的关系
dt_series = pd.to_datetime(df['issue_d'])
df['year'] = dt_series.dt.year	# 新创年份属性


plt.figure(figsize = (12,8))
sns.barplot('year','loan_amount',data = df,palette = 'tab10')		#第一个为x，第二个为y 
plt.title("Issuance of Loans", fontsize = 14)	
plt.xlabel('year',fontsize = 14)
plt.y_label("Average loan amount issued", fontsize = 14)

#贷款数量逐年升高


df["loan_status"].value_counts() 
'''
Current                                                601779
Fully Paid                                             207723
Charged Off                                             45248
Late (31-120 days)                                      11591
Issued                                                   8460
In Grace Period                                          6253
Late (16-30 days)                                        2357
Does not meet the credit policy. Status:Fully Paid       1988
Default                                                  1219
Does not meet the credit policy. Status:Charged Off       761
Name: loan_status, dtype: int64
'''

#对贷款好坏进行二分值变化
bad_loan= ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]

df['loan_condition'] = np.nan

def loan_condition(status):
	if status in bad_loan:
		return "Bad Loan"
	else:
		return "Good Loan"

df['loan_condition'] = df['loan_status'].apply(loan_condition)

#查看好坏贷款分布与好坏贷款随年份变化的关系

f,ax = plt.subplots(1,2, figsize = (16,8))
colors = ["#3791D7", "#D72626"]
labels = "Good Loans","Bad Loans"

plt.suptitle('Information on Loan Conditions', fontsize=20)
df['loan_condition'].value_counts().plot.pie(exploede = [0,0.1], autopct = "%1.2f%%", ax =ax[0],shadow = True, colors = colors, labels = label, fontsize = 12, startangle = 70)

ax[0].set_ylabel(r"% of Condition of Loans", fontsize = 4)

palette = ["#3791D7", "#E01E1B"]
sns.barplot(x = "year",y = "loan_amount",hue = "loan_condition", data = df,palette = palette, estimator = lambda x : len(x) / len(df) * 100)
ax[1].set_ylabel("%",fontsize = 4)


#Loan and Region 
#Translate different places into five parts
df['addr_state'].unique() 
west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
south_west = ['AZ', 'TX', 'NM', 'OK']
south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

df['region'] = np.nan 

def finding_regions(state):
	if state in west:
		return 'West'
	elif state in south_west:
		return 'SouthWest'
	elif state in south_east:
		return 'SouthEast'
	elif state in mid_west:
		return 'MidWest'
	elif state in north_east:
		return 'NorthWest'

df['region'] = df['addr_state'].apply(finding_regions)


#统计每个区域随年份变化的贷款数量
df['complete_date'] = pd.to_datetime(df['issue_d'])
group_dates = df.groupby(['complete_date', 'region'], as_index = False).sum()

group_dates['issue_d'] = [month.to_period('M') for month in group_dates['complete_date']]		#变为以年-月为索引的表格
group_dates = group_dates.groupby(['issue_d','region'], as_index = False).sum()		#以年月与区域为分组的对每个年月每个地域的loan进行求和
group_dates['loan_amount'] = group_dates['loan_amount']/1000 

df_dates = pd.DataFrame(data = group_dates[['issue_d','region','loan_amount']])
plt.style.use('dark_background')
cmap = plt.cm.Set3
by_issued_amount = df_dates.groupby(['issue_d','region']).loan_amount.sum()
by_issued_amount.unstack().plot(stacked = False, colormap = cmap, legend = True, figsize = (15,6))
plt.set_title('Loans issued by Region', fontsize = 16)		# 又等于 plt.title('Loans issued by Region', fontsize = 16)


#将工作年限属性变为int属性 emp_length
employment_length = ['10+ years', '< 1 year','1 year','3 years','8 years', '9 years','4 years','5 years', '6 years','2 years','7 years','n/a']


df['emp_length_int'] = np.nan 
df.loc[df['emp_length'] == '10+ years', "emp_length_int"] = 10
df.loc[df['emp_length'] == '9 years', "emp_length_int"] = 9
df.loc[df['emp_length'] == '8 years', "emp_length_int"] = 8
df.loc[df['emp_length'] == '7 years', "emp_length_int"] = 7
df.loc[df['emp_length'] == '6 years', "emp_length_int"] = 6
df.loc[df['emp_length'] == '5 years', "emp_length_int"] = 5
df.loc[df['emp_length'] == '4 years', "emp_length_int"] = 4
df.loc[df['emp_length'] == '3 years', "emp_length_int"] = 3
df.loc[df['emp_length'] == '2 years', "emp_length_int"] = 2 
df.loc[df['emp_length'] == '1 years', "emp_length_int"] = 1 
df.loc[df['emp_length'] == '< 1 years', "emp_length_int"] = 0.5
df.loc[df['emp_length'] == 'n/a', "emp_length_int"] = 0


#各州各变量随年份变化(利率，工作长度，平均债与收入比，平均年收入)
sns.set_style('whitegrid')
figure , axx = plt.subplots(2,2) 
cmap = plt.cm.inferno 

by_interest_rate = df.groupby(['year','region']).interest_rate.mean() 
by_interest_rate.unstack().plot(kind = 'area', stacked = True, colormap = cmap, grid = False, legend = False, ax = axx[0], figsize = (16,12))
axx[0].set_title('Average Interest Rate by Region', fontsize=14)

by_employment_length = df.groupby(['year', ' region']).emp_length_int.mean() 
by_employment_length.unstack().plot(kind = 'area', stacked = True, colormap = cmap, grid = False, legend=False, ax =axx[1], figsize = (16,12))
axx[1].set_title('Average Employment Length by Region', fontsize=14)
	
by_dti = df.groupby(['year','region']).dti.mean() 
by_dti.unstack().plot(kind = 'area', stacked = True, colormap=cmap, grid = False, legend = False, ax = axx[2],figsize=(16,12))
axx[2].set_title('Average Debt-to-Income by Region', fontsize=14)

by_income = df.groupby(['year','region']).annual_income.mean() 
by_income.unstack().plot(kind='area', stacked = Trye, colormap=cmap, grid = False, ax = axx[3], figsize = (16,12))
axx[3].set_title('Average Annual Income by Region', fontsize=14)
axx[3].legend(bbox_to_anchor = (-1.0,-0.5,1.8,0.1), loc = 10, prop = {'size':12}, ncol = 5, mode = "expand", borderaxespad = 0.)

#每个州在四个图的位置分布一样


#对bad loan的数据进行具体分析
badloans_df = df.loc[df['loan_condition'] == "Bad Loan"]

loan_status_cross = pd.crosstab(badloans_df['region'],badloans_df['loan_status']).apply(lambda x : x/ x.sum() * 100 )
number_of_loanstatus = pd.crosstab(badloans_df['region'], badloans_df['loan_status'])

loan_status_cross['Charged Off'] = loan_status_cross['Charged Off'].apply(lambda x : round(x,2)) 
loan_status_cross['Default'] = loan_status_cross['Default'].apply(lambda x : round(x,2))
loan_status_cross['Does not meet the credit policy. Status:Charged Off'] = loan_status_cross['Does not meet the credit policy. Status:Charged Off'].apply(lambda x: round(x, 2))
loan_status_cross['In Grace Period'] = loan_status_cross['In Grace Period'].apply(lambda x : round(x,2)) 
loan_status_cross['Late (16-30 days)'] = loan_status_cross['Late (16-30 days)'].apply(lambda x: round(x, 2))
loan_status_cross['Late (31-120 days)'] = loan_status_cross['Late (31-120 days)'].apply(lambda x: round(x, 2))
 
number_of_loanstatus['Total'] = number_of_loanstatus.sum(axis = 1) 	#对前面的属性值求和
#print(number_of_loanstatus)


#先将各贷款还款情况做成对应的表
#先将其值变为List便于迭代
charged_off = loan_status_cross['Charged Off'].values.tolist() 
default = loan_status_cross['Default'].values.tolist() 
not_meet_credit = loan_status_cross['Does not meet the credit policy. Status:Charged Off'].values.tolist()
grace_period = loan_status_cross['In Grace Period'].values.tolist() 
grace_period = loan_status_cross['In Grace Period'].values.tolist()
short_pay = loan_status_cross['Late (16-30 days)'] .values.tolist()
long_pay = loan_status_cross['Late (31-120 days)'].values.tolist()
	
charged = go.Bar(
	x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= charged_off,													#传入x与y的为其值的list
    name='Charged Off',
    marker=dict(
        color='rgb(192, 148, 246)'
    ),
    text = '%'
)

defaults = go.Bar(
	x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=default,
    name='Defaults',
    marker=dict(
        color='rgb(176, 26, 26)'
    ),
    text = '%'
)

grace = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= grace_period,
    name='Grace Period',
    marker = dict(
        color='rgb(147, 147, 147)'
    ),
    text = '%'
)

short_pays = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= short_pay,
    name='Late Payment (16-30 days)', 
    marker = dict(
        color='rgb(246, 157, 135)'
    ),
    text = '%'
)

long_pays = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= long_pay,
    name='Late Payment (31-120 days)',
    marker = dict(
        color = 'rgb(238, 76, 73)'
        ),
    text = '%'
)

data = [charged, defaults, credit_policy, grace, short_pays, long_pays]
layout = go.layout(
	barmode = 'stack',
	title =  '% of Bad Loan Status by Region',
	xaxis = dict(title = 'US Regions')
	)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'stacked-bar')

#平均利率
ir = df['interest_rate'].mean() 
#平均年收入
ai = df['annual_income'].mean()


#three key metrics: Loans issued by state (Total Sum), 
#Average interest rates charged to customers and average annual income of all customers by state.
#to see states that give high returns at a descent risk

by_loan_amout = df.groupby(['region','addr_state'], as_index = False).loan_amount.sum()
by_interest_rate = df.groupby(['region','addr_state'], as_index = False).interest_rate.mean()
by_income = df.groupby(['region','addr_state'], as_index = False).annual_income.mean()

states = by_loan_amout['addr_state'].values.tolist() 
average_loan_amounts = by_loan_amout['loan_amount'].values.tolist() 
average_interest_rates = by_interest_rate['interest_rate'].values.tolist() 
average_annual_income = by_income['annual_income'].values.tolist()

from collections import OrderedDict

metrics_data = OrderedDict([('state_codes', states),
                            ('issued_loans', average_loan_amounts),
                            ('interest_rate', average_interest_rates),
                            ('annual_income', average_annual_income)])

metrics_df = pd.DataFrame.from_dict(metrics_data)		# dict -> dataframe 
metrics_df = metrics_df.round(decimals = 2)


















