import streamlit as st
import numpy as np
import pandas as pd
import scikit learn
from scikit learn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""Simple Iris Flower Prediction""")

st.sidebar.header('User Input parameter')

def User_Input_Features():
	sepal_length = st.sidebar.slider('sepal_length',4.3, 7.9, 5.4)
	sepal_width = st.sidebar.slider('sepal_width',3.5, 7.9, 5.4)
	petal_length = st.sidebar.slider('petal_length',1.4, 7.9, 5.4)
	petal_width = st.sidebar.slider('petal_width',0.2, 7.9, 5.4)

	data = { 'sepal_length' :sepal_length , 
	'sepal_width':sepal_width,
	'petal_length' : petal_length,
	'petal_width'  : petal_width
	  }

	features = pd.DataFrame(data,index = [0])
	return features

df = User_Input_Features()
st.subheader('User Input Parameter')
st.write(df)

iris = datasets.load_iris()
x= iris.data 
y = iris.target

classify = RandomForestClassifier()
classify.fit(x,y)

prediction = classify.predict(df)
prediction_probability = classify.predict_proba(df)


st.subheader('class label and their corresponding index number')
st.write(iris.target_names)


st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_probability)
st.bar_chart(prediction_probability)
