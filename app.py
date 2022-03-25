import numpy as np
import pandas as pd
import pickle
import streamlit as st

pickle_model = open("model/linear_v1.pkl", "rb")
regressor = pickle.load(pickle_model)

def predict_chance(GREScore,TOEFLScore,UniversityRating,CGPA):
    """
    This function is used to predict the chance of admission based on the input parameters.
    """
    return regressor.predict([[GREScore,TOEFLScore,UniversityRating,CGPA]])

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
    GREScore=st.text_input("GRE Score(0-340)")
    TOEFLScore=st.text_input("TOEFL Score(0-120)")
    UniversityRating=st.text_input("University Rating(1-5)")
    CGPA=st.text_input("CGPA(1-10)")
    result = ""
    if st.button("Predict"):
        result = predict_chance(GREScore,TOEFLScore,UniversityRating,CGPA)
        st.success(f"The chance of admission is {round(float(result[0][0])*100,2)}%")
    

if __name__ == "__main__":
    main()