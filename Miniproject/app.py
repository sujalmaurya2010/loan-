
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score # add kiya

st.set_page_config(page_title="Loan",page_icon="😊")
st.title("😊 Loan Approval Prediction System")

df = pd.read_csv("loan.csv")
st.dataframe(df) # fix 1

x = df[["Income","CIBIL_Score","Loan_Amount","Employment_Years"]]
y = df["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

model = DecisionTreeClassifier(max_depth=3,min_samples_split=8,min_samples_leaf=4,random_state=42)
model.fit(x_train, y_train) # fix 2

prediction = model.predict(x_test) # fix 3,4
accuracy = accuracy_score(y_test, prediction) # fix 5

st.success(f"Our model accuracy is {accuracy*100:.2f}%") # fix 5

income = st.sidebar.number_input("Enter your income :",min_value=10000,max_value=500000,value=50000,step=5000)
cibil = st.sidebar.number_input("Enter your cibil :",min_value=500,max_value=900,value=620,step=10)
loan = st.sidebar.number_input("Enter your loan amount :",min_value=50000,max_value=1000000,value=250000,step=10000)
experience = st.sidebar.number_input("Enter your experience :",min_value=0,max_value=40,value=5,step=5)

if st.sidebar.button("Predict Loan"):
    result = model.predict([[income,cibil,loan,experience]])
    if result[0]==1:
        approved = min(loan,int(income*8+(cibil-650)*400+experience*1000))
        if approved<0:
            approved=0

        st.success("Loan Approved")
        st.info(f"Approved Loan Amount :$ {approved:,}")
    else:
        st.error("Loan Rejected")
