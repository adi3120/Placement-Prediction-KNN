import streamlit as st
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Read Data from csv file
data=pd.read_csv("collegePlace.csv")

#Encode Categorical Data
Stream=data['Stream'].map({"Civil":1,"Computer Science":2,"Electrical":3,"Electronics And Communication":4,"Information Technology":5,"Mechanical":6})
data_new=data.copy()
data_new["Stream"]=Stream
data_new["Gender"]=pd.get_dummies(data["Gender"],drop_first=True)

#Select Neccesary Categorical Attributes from the dataset 
x=data_new[["Age","Gender","Stream","Internships","CGPA","Hostel","HistoryOfBacklogs"]]

#Select Final Result Column
y=data_new["PlacedOrNot"]

#Split Data into Test and Train Data
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.8,random_state=6)

#Initialize Classifier Model - Here KNN
knn = KNeighborsClassifier(n_neighbors=3)

#Fit Model 
knn.fit(xtrain,ytrain)


#Lets start UI with Streamlit
st.title("Campus Placement Predicter")

#Take user input into different variables
ip_Age=int(st.number_input("Enter Age : "))
ip_Gender=st.selectbox("Select Gender : ",["Male","Female"])

ip_options_streams=["Civil","Computer Science","Electrical","Electronics And Communication","Information Technology","Mechanical"]

ip_Stream=st.selectbox("Select Stream : ",ip_options_streams)
ip_Internships=int(st.number_input("Enter Number of Internships Completed : "))
ip_CGPA=int(st.number_input("Enter CGPA :"))
ip_Hostel=st.selectbox("Hosteler ?",["Yes","No"])
ip_HOB=st.selectbox("Had History of Backlog ?",["Yes","No"])


#Convert input values to a form that our model can understand
if ip_Gender=="Male":
	ip_Gender=1
else:
	ip_Gender=0

if ip_Stream=="Civil":
	ip_Stream=1
elif ip_Stream=="Computer Science":
	ip_Stream=2
elif ip_Stream=="Electrical":
	ip_Stream=3
elif ip_Stream=="Electronics And Communication":
	ip_Stream=4
elif ip_Stream=="Information Technology":
	ip_Stream=5
elif ip_Stream=="Mechanical":
	ip_Stream=6


if ip_Hostel=="Yes":
	ip_Hostel=1
else:
	ip_Hostel=0

if ip_HOB=="Yes":
	ip_HOB=1
else:
	ip_HOB=0


#Put inputs in an array

inputs=[ip_Age,ip_Gender,ip_Stream,ip_Internships,ip_CGPA,ip_Hostel,ip_HOB]


#make a submit button
submit=st.button("Predict Placement !!")

if(submit):

	#Calculate prediction using KNN classifier
	knn_yp = knn.predict([inputs])

	if(knn_yp==1):
		st.header("Good Chances of Placement :)")
	else:
		st.header("Need to work hard, weak Placement Chances ! ")


#Thankyou
st.write("Made with ❤️ By Aditya Yadav")









