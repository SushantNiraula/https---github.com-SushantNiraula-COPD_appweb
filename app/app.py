import pandas as pd
import joblib 
import streamlit as st

with open('model\pipe.sav', 'rb') as f:
    pipe = joblib.load(f)

with open('model\Best_Random_Forest_Model.sav', 'rb') as f:
    model = joblib.load(f)


def main():
    st.title("COPD Prediction Dashboard")

    # User input
    st.sidebar.header("User Input")


    Smoking_Status=st.sidebar.selectbox("Smoking Status",["Current","Former","Never"])
    Biomass_Fuel_Exposure=st.sidebar.selectbox("Biomass Fuel Exposure",["Yes","No"])
    Occupational_Exposure=st.sidebar.selectbox("Occupational Exposure",["Yes","No"])
    Family_History_COPD=st.sidebar.selectbox("Family History",["Yes","No"])
    Location=st.sidebar.selectbox("Location",["Kathmandu","Pokhara","Biratnagar","Lalitpur","Birgunj","Chitwan","Hetauda","Dharan","Butwal","Bhaktapur"])
    Respiratory_Infections_Childhood=st.sidebar.selectbox("Respiratory Infections in Childhood",["Yes","No"])
    Age_Category=st.sidebar.selectbox("Age Category",['young','adult','middle_aged','old','too_old'])
    BMI_category=st.sidebar.selectbox("BMI Category",['underweight','normal','overweight','obese','too_obese'])
    Air_Pollution_Level_category=st.sidebar.selectbox("Air Pollution Level Category",['Good','Satisfactory','Moderate','Poor','Very_Poor','Severe'])
    Gender_encoded=st.sidebar.selectbox("Gender",['Male','Female'])
   
    # Process the input data
    input_data = {
        'Smoking_Status': [Smoking_Status],
        'Biomass_Fuel_Exposure': [Biomass_Fuel_Exposure],
        'Occupational_Exposure': [Occupational_Exposure],
        'Family_History_COPD': [Family_History_COPD],
        'Location': [Location],
        'Respiratory_Infections_Childhood': [Respiratory_Infections_Childhood],
        'Age_Category': [Age_Category],
        'BMI_category': [BMI_category],
        'Air_Pollution_Level_category': [Air_Pollution_Level_category],
        'Gender_encoded':[Gender_encoded]
    }
    
    # Convert the data to a dataframe
    input_df = pd.DataFrame(input_data)

    # Encoding
    input_df['Biomass_Fuel_Exposure']=input_df['Biomass_Fuel_Exposure'].map({'Yes':1,'No':0})
    input_df['Occupational_Exposure']=input_df['Occupational_Exposure'].map({'Yes':1,'No':0})
    input_df['Family_History_COPD']=input_df['Family_History_COPD'].map({'Yes':1,'No':0})
    input_df['Respiratory_Infections_Childhood']=input_df['Respiratory_Infections_Childhood'].map({'Yes':1,'No':0})
    input_df['Gender_encoded']=input_df['Gender_encoded'].map({'Male':1,'Female':0})
    # now for the fitting in pipe we need to convert the data frame to numpy array.
    num_arr=input_df.to_numpy()
    num_arr=num_arr.reshape(1,10)


    transformed_ip=pipe.transform(num_arr)
    
    if st.button("Predict"):
        prediction = model.predict(transformed_ip)
        if prediction[0]==1:
            st.write("You have COPD")
        else:
            st.write("You don't have COPD")





if __name__=='__main__':
    main()