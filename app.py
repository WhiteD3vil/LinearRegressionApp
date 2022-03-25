import numpy as np
import pandas as pd
import pickle
import streamlit as st

pickle_model = open("model/linear_v1.pkl", "rb")
regressor = pickle.load(pickle_model)

def predict_chance(GREScore,TOEFLScore,UniversityRanking,CGPA):
    """
    This function is used to predict the chance of admission based on the input parameters.
    """
    return regressor.predict([[GREScore,TOEFLScore,UniversityRanking,CGPA]])

def main():
    """
    This function is used to display the app.
    """
    st.title("Admission Prediction App")
    html_temp="""
        <div style="background-color:tomato;padding:10px">
        <h2 style ="color:black;text-align:center;">Admission Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)
    GREScore = st.text_input("GRE Score")
    TOEFLScore = st.text_input("TOEFL Score")
    UniversityRanking = st.text_input("University Ranking")
    CGPA = st.text_input("CGPA")
    result = ""
    if st.button("Predict"):
        result = predict_chance(GREScore,TOEFLScore,UniversityRanking,CGPA)
    st.success(f"The chance of admission is {int(result[][])*100}%")

if __name__ == "__main__":
    main()