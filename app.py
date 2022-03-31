import numpy as np
import pandas as pd
import pickle
import streamlit as st

pickle_model = open("model/linear_v1.pkl", "rb")
regressor = pickle.load(pickle_model)

def predict_admission(GREScore,TOEFLScore,UniversityRating,CGPA):
    """
    This function is used to predict the chance of admission based on the input parameters.
    """
    return regressor.predict([[GREScore,TOEFLScore,UniversityRating,CGPA]])

def predict_price_of_diamond(Diamond_length,Diamond_width,Diamond_depth):
    """
    This function is used to predict the chance of admission based on the input parameters.
    """
    return regressor.predict([[Diamond_length,Diamond_width,Diamond_depth]])

@st.cache
def load_admission_data(nrows):
    return pd.read_csv("datasets/Admission_Predict.csv", nrows=nrows)
@st.cache
def load_diamonds_data(nrows):
    return pd.read_csv('datasets/diamonds.csv',index_col=0, nrows=nrows)

def admission_st_app():
    st.title("Admission Prediction App")
    html_temp="""
        <div style="background-color:lightyellow;padding:10px">
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
        result = predict_admission(GREScore,TOEFLScore,UniversityRating,CGPA)
        st.success(f"The chance of admission is {round(float(result[0][0])*100,2)}%")


def diamonds_st_app():
    st.title("Diamonds Price Prediction App")
    html_temp="""
        <div style="background-color:skyblue;padding:10px">
        <h2 style ="color:black;text-align:center;">Admission Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)
    Diamond_length=st.text_input("Diamond length (0-15)")
    Diamond_width=st.text_input("Diamond Width (0-50)")
    Diamond_depth=st.text_input("Diamond depth (1-35)")
    result = ""
    if st.button("Predict"):
        result = predict_price_of_diamond(Diamond_length,Diamond_width,Diamond_depth)
        st.success(f"The price of diamond is {round(float(result[0][0]))}$ in US")

def main():
    """
    This function is used to display the app.
    """
    st.set_page_config(layout="wide")
    col1, col2, col3 = st.columns((0.5,1,1))
    with col1:
        option = st.selectbox('Select your APP',('None','Admission Prediction App', 'Predict Diamonds Prices'))
        if option == 'Admission Prediction App':
            choice = st.selectbox('Do you want to look at ?',('Dataset','Model Performance', 'Application Demo'))
            number_of_students = st.slider('Select Number of Students?', 0, 400, 5)
            st.write("Selected ", number_of_students, 'Students')
        elif option == "Predict Diamonds Prices":
            choice = st.selectbox('Do you want to look at ?',('Dataset','Model Performance', 'Application Demo'))
            row_number = st.slider('Select number of rows', 0,400, 5)
            st.write("Rows Selected: ", row_number)
    with col3:
        if option == 'None':
            st.title("Choose any App!")
        elif option == 'Admission Prediction App' and choice == 'Application Demo':
            admission_st_app()
        elif option == 'Admission Prediction App' and choice == 'Dataset':
            data_load_state = st.text('Loading data...')
            admission_data = load_admission_data(number_of_students)
            st.subheader('Admission Data')
            st.write(admission_data)
            with col2:
                #Bar Chart
                st.bar_chart(admission_data['CGPA'])
        elif option == "Predict Diamonds Prices" and choice == 'Application Demo':
            diamonds_st_app()
        elif option == 'Predict Diamonds Prices' and choice == 'Dataset':
            data_load_state = st.text('Loading data...')
            diamonds_data = load_diamonds_data(row_number)
            st.subheader('Diamonds Data')
            st.write(diamonds_data)
            with col2:
                #Bar Chart
                st.bar_chart(diamonds_data['carat'])

if __name__ == "__main__":
    main()