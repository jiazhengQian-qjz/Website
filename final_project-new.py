import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
import sys


st.title("Loan Prediction Based on Customer Behavior")
st.write("Predict who possible Defaulters are for the Consumer Loans Product")

# Data Processing
df = pd.read_csv('final_data.csv')
df.drop_duplicates(inplace=True)      
df.fillna(df.mean(),inplace=True)

# Outlier Handling
df['Income_value_z'] = stats.zscore(df.Income)
df = df[(df.Income_value_z > -3) & (df.Income_value_z < 3)]
df['Age_value_z'] = stats.zscore(df.Age)
df = df[(df.Age_value_z > -3) &  (df.Age_value_z < 3)]
df['Experience_value_z'] = stats.zscore(df.Experience)
df = df[(df.Experience_value_z > -3) &  (df.Experience_value_z < 3)]
df['CURRENT_JOB_YRS_value_z'] = stats.zscore(df.CURRENT_JOB_YRS)
df = df[(df.CURRENT_JOB_YRS_value_z > -3) & (df.CURRENT_JOB_YRS_value_z < 3)]
df['CURRENT_HOUSE_YRS_value_z'] = stats.zscore(df.CURRENT_HOUSE_YRS)
df = df[(df.CURRENT_HOUSE_YRS_value_z > -3) & (df.CURRENT_HOUSE_YRS_value_z < 3)]
df.replace("Uttar_Pradesh[5]","Uttar_Pradesh",inplace=True)

# create a multi select
House_Ownership_filter = st.sidebar.multiselect(
     'House Ownership Selector',
     df.House_Ownership.unique(),  # options
     df.House_Ownership.unique())  # defaults

# create a input form
form = st.sidebar.form("Profession_form")
Profession_filter = form.text_input('Profession Type (enter ALL to reset)', 'ALL')
form.form_submit_button("Apply")

# filter by House_Ownership
df = df[df.House_Ownership.isin(House_Ownership_filter)]

# filter by Profession_filter
if Profession_filter!='ALL':
    df = df[df.Profession == Profession_filter]


# data set description
image1 = plt.imread("微信图片_20221028205848.jpg")
st.image(image1,use_column_width=True)
st.subheader("·Introduction of Research")
st.write("An organization wants to predict who possible defaulters are for the consumer loans product. They have data about historic customer behavior based on what they have observed. Hence when they acquire new customers they want to predict who is riskier and who is not.")

st.subheader("·Dataset Description")
st.markdown("***The resource of dataset:***\n<https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior>")
st.markdown("***Click the check box to view the data:***")
if st.checkbox("show original data"):
    st.write(df)   
st.subheader('Fig1: Geographical Scope of Study')
fig, ax = plt.subplots()
df2 = df.STATE.value_counts()
a = list(df2.index)
b = list(df2.values)
ax.set_xlabel("State")
ax.set_ylabel("Amount of Residence")
plt.bar(a,b,width = 0.5)
plt.xticks(rotation = 270)
st.pyplot(fig)
st.write("The research covers 26 states around the world.")

st.subheader('Fig2: Income Situation of Research Sample')   
fig, ax = plt.subplots()
df_sorted_Income = df.sort_values(by='Income', ignore_index=True, ascending=False)
df_sorted_Income = df_sorted_Income["Income"]/1000
df_sorted_Income.plot()
ax.set_ylabel('Income (unit:thousand)')
ax.set_xlabel("Sample")
st.pyplot(fig)
st.write("The research covers groups with the yearly income < $10,000,000.")

st.markdown("**Content**")
st.write("All values were provided at the time of the loan application.")
st.markdown("***Attributes Description:***")
image = plt.imread("微信图片_20221028205907.jpg")
st.image(image,width=600,caption = "Attributes of Samples",use_column_width=True)


# Research Problem
st.subheader("·Problem 1")
st.markdown("")
st.subheader('Fig3: Relationship Between Property Ownership and Default Possibility')
df_house= df.groupby('House_Ownership').sum()
try:
    rent = df_house.loc['rented','Risk_Flag']/len(df[df.House_Ownership == 'rented'])
except:
    rent = 0
try:
    own = df_house.loc['owned','Risk_Flag']/len(df[df.House_Ownership == 'owned'])
except:
    own = 0
try:
    neither = df_house.loc['norent_noown','Risk_Flag']/len(df[df.House_Ownership == 'norent_noown'])
except:
    neither = 0
fig, ax = plt.subplots(figsize=(10, 5))
a = [rent,own,neither]
b = ['rent','own','neither']
plt.bar(b,a,width = 0.5)
ax.set_xlabel('Default rates for three types of housing')
ax.set_ylabel('rate')
st.pyplot(fig)
st.write(f"The default rates of the three groups \(\"rent\",\"own\",\"neither\") are {rent*100:.3f}%,{own*100:.3f}% and {neither*100:.3f}% respectively")
st.markdown("**Data Description**")
st.write(f"The figure (Fig3) shows that the default rate of rental groups ({rent*100:.3f}%) is higher than that of homeowners ({own*100:.3f}%).")


st.subheader('Fig4: Default Ratio of Car_Ownership')
df3 = df.groupby("Car_Ownership").sum()
fig, ax = plt.subplots(figsize=(20, 5))
rate = df3.iloc[1].Risk_Flag/(df3.iloc[0].Risk_Flag+df3.iloc[1].Risk_Flag)
plt.pie([rate,1-rate],autopct="%.1f%%",labels=["Car_Ownership = Yes","Car_Ownership = No"])
st.pyplot(fig)
st.markdown("**Data Description**")
st.write(f"The figure (Fig4) shows that the default rate of the group that does not own the car ({rate:.3f}%) is about 2.5 times higher than that of the group who owns the car ({1-rate:.3f}%). The conclusion is that there is a correlation between car ownership and default risk, and the default rate of the group without car ownership is higher.")


st.subheader('Fig5: Default Ratio of Groups with Different Marital Status')
fig, ax = plt.subplots(figsize=(15,5))
default_rate_single = df[(df["Married/Single"] == "single")&(df["Risk_Flag"]==1)].shape[0]/df[df["Married/Single"] == "single"].shape[0]
default_rate_married = df[(df["Married/Single"] == "married")&(df["Risk_Flag"]==1)].shape[0]/df[df["Married/Single"] == "married"].shape[0]
title = ["default_of_single","default_of_married"]
data = [default_rate_single,default_rate_married]
ax.bar(title,data)
ax.set_xlabel("marital_status")
ax.set_ylabel("default_rate")
st.pyplot(fig)
st.write(f"The difference of default ratio between married and single groups is {abs(default_rate_single-default_rate_married):.3f}")
st.markdown("**Data Description**")
st.write(f'The figure (Fig5) shows that the default rate of single people is higher than that of married people (the gap is about {abs(default_rate_single-default_rate_married):.3f}%).')

st.subheader("·Problem 2")
st.subheader('Fig6: Relationship Between Profession Experience and Default Possibility')
group_ratio0 = (df[(df["Experience"]<4)&(df["Risk_Flag"]!=0)].shape[0])/(df[df["Experience"]<4].shape[0])
group_ratio1 = (df[(df["Experience"]>=4)&(df["Experience"]<8)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Experience"]>=4)&(df["Experience"]<8)].shape[0])
group_ratio2 = (df[(df["Experience"]>=8)&(df["Experience"]<12)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Experience"]>=8)&(df["Experience"]<12)].shape[0])
group_ratio3 = (df[(df["Experience"]>=12)&(df["Experience"]<16)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Experience"]>=12)&(df["Experience"]<16)].shape[0])
group_ratio4 = (df[(df["Experience"]>=16)&(df["Experience"]<20)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Experience"]>=16)&(df["Experience"]<20)].shape[0])
ratio_list = [group_ratio0,group_ratio1,group_ratio2,group_ratio3,group_ratio4]
age_range = ["0-4","4-8","8-12","12-16","16-20"]                                # has no linear correlationship
fig, ax = plt.subplots()                                                                     # set the x-label/y-label
ax.plot(age_range, ratio_list)
ax.set_xlabel("range of professional experience in years")
ax.set_ylabel("potential of default")
st.pyplot(fig)
st.markdown("***Group default rate with different work experience ranges:***")
st.write(f'Group default rate with work experience in the range of 0-4 years is {group_ratio0*100:.3f}%')
st.write(f'Group default rate with work experience in the range of 4-8 years is {group_ratio1*100:.3f}%')
st.write(f'Group default rate with work experience in the range of 8-12 years is {group_ratio2*100:.3f}%')
st.write(f'Group default rate with work experience in the range of 12-16 years is {group_ratio3*100:.3f}%')
st.write(f'Group default rate with work experience in the range of 16-20 years is {group_ratio4*100:.3f}%')
st.markdown("**Data Description**")
st.write(f"The figure (Fig6) shows that there is a significant negative correlation between years of work experience and default rate. Among them, the group with 0-4 years of work experience had the highest default rate ({group_ratio0*100:.3f}%), and the group with 16-20 years of work experience had the lowest default rate ({group_ratio4*100:.3f}%).")

st.subheader('Fig7: Default Ratio of Different Age Groups')
group_ratio0 = (df[(df["Age"]<30)&(df["Risk_Flag"]!=0)].shape[0])/(df[df["Age"]<30].shape[0])
group_ratio1 = (df[(df["Age"]>=30)&(df["Age"]<40)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Age"]>=30)&(df["Age"]<40)].shape[0])
group_ratio2 = (df[(df["Age"]>=40)&(df["Age"]<50)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Age"]>=40)&(df["Age"]<50)].shape[0])
group_ratio3 = (df[(df["Age"]>=50)&(df["Age"]<60)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Age"]>=50)&(df["Age"]<60)].shape[0])
group_ratio4 = (df[(df["Age"]>=60)&(df["Age"]<70)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Age"]>=60)&(df["Age"]<70)].shape[0])
group_ratio5 = (df[(df["Age"]>=70)&(df["Age"]<80)&(df["Risk_Flag"]!=0)].shape[0])/(df[(df["Age"]>=70)&(df["Age"]<80)].shape[0])
ratio_list = [group_ratio0,group_ratio1,group_ratio2,group_ratio3,group_ratio4,group_ratio5]
age_range = ["20-30","30-40","40-50","50-60","60-70","70-80"]                                # has no linear correlationship
fig, ax = plt.subplots()                                                                     # set the x-label/y-label
ax.plot(age_range, ratio_list)
ax.set_xlabel("age range")
ax.set_ylabel("potential of default")
difference = abs(group_ratio2-group_ratio3)
st.pyplot(fig)
st.write(f"The default rate of group with age range of 20-30 is: {group_ratio0*100:.3f}%")
st.write(f"The default rate of group with age range of 30-40 is: {group_ratio1*100:.3f}%")
st.write(f"The default rate of group with age range of 40-50 is: {group_ratio2*100:.3f}%")
st.write(f"The default rate of group with age range of 50-60 is: {group_ratio3*100:.3f}%")
st.write(f"The default rate of group with age range of 60-70 is: {group_ratio4*100:.3f}%")
st.write(f"The default rate of group with age range of 70-80 is: {group_ratio5*100:.3f}%")
st.markdown("**Data Description**")
st.write(f"The figure (Fig7) shows that the default rate is the lowest in 60-70 age group ({group_ratio4*100:.3f}%) and the highest in 20-30 age group ({group_ratio0*100:.3f}%). The gap between the default rate of 40-50 and 50-60 age groups is small (about{difference:.7f}).")



st.subheader("·Further study")
st.subheader('Fig8: Importance Ranking of Features')
image3 = plt.imread("e912c02d147696b840bcba3158c4a8f.png")
st.image(image3,use_column_width=True)
st.markdown("**Data Description**")
st.write(f"We use the random forest algorithm to rank the importance of some features, and it is found that the importance of these features on the impact of loan default results is ranked as follows:1.Income 2.Age 3.Experience 4.Current_job_years 5.Current_house_years")
st.markdown("**The final results:**")
st.write('Score for the training data set:98.29%')
st.write('Score for the testing data set:87.87%')