import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from keras.models import load_model

# Load the data
@st.cache_data
def load_data():
    file_path = 'SENSOR DATA/SensorData.csv'
    return pd.read_csv(file_path)




def main():
    df = load_data()
    st.title("Equipment Failure Prediction")

    st.header("Prediction")
    st.write("Enter the following parameters to predict equipment failure:")

    ###################################################################################################################


    # User input fields ################################################################################################
    air_temp = st.slider("Air temperature [K]", min_value=270.0, max_value=314.0, value=295.0)
    process_temp = st.slider("Process temperature [K]", min_value=290.0, max_value=314.0, value=300.0)
    rotational_speed = st.slider("Rotational speed [rpm]", min_value=1150.0, max_value=2900.0, value=1500.0)
    torque = st.slider("Torque [Nm]", min_value=2.0, max_value=80.0, value=30.0)
    tool_wear = st.slider("Tool wear [min]", min_value=0.0, max_value=250.0, value=80.0)

    ####################################################################################################################
    user_input = pd.DataFrame({
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        })
    
    # Once the user inputs are collected, make predictions##############################################################
    if st.button("Predict"):
        predict(user_input,df)

def predict(user_input,df):
    # Perform predictions using multiple models
    Random_Forest = RF_predict(user_input,df)
    LSTM_output = LSTM_predict(user_input,df)

    if LSTM_output[0][0] > 0.5:
            st.markdown(
            """
            <style>
            .beautiful-text {
                font-size: 80px;
                color: #0072B2;
                text-align: center;
                font-weight: bold;
                margin-top: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

            # Display the beautiful message
            st.markdown('<p class="beautiful-text">💥 The machine will fail! 💥</p>', unsafe_allow_html=True)
    else:
            st.markdown(
            """
            <style>
            .beautiful-text {
                font-size: 80px;
                color: #0072B2;
                text-align: center;
                font-weight: bold;
                margin-top: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

            # Display the beautiful message
            st.markdown('<p class="beautiful-text">🌟 The machine will not fail! 🌟</p>', unsafe_allow_html=True)
    
    # Display predictions in separate tabs
    with st.expander("Random Forrest Prediction"):
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        output = Random_Forest
        for i, failure_type in enumerate(failure_types):
            st.write(f"The machine has a {output[i][0][1] * 100:.2f}% chance of {failure_type} failure.")
            # Data for the pie chart
            labels = 'Fail', 'Not Fail'
            sizes = [output[i][0][1] * 100, 100 - output[i][0][1] * 100]
            colors = ['red', 'green']
            explode = (0.1, 0)  # explode 1st slice

            # Create the pie chart
            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

            # Display the pie chart in Streamlit
            st.pyplot(fig)

##################################################################################
#########################   MODELS LOADING   #####################################
##################################################################################


def RF_predict(user_input,df):
    df1 = df.copy()
    df1 = pd.get_dummies(df1, columns=['Type'])
    df1['Air temperature [K]'] = df1['Air temperature [K]'].astype(int)
    df1['Process temperature [K]'] = df1['Process temperature [K]'].astype(int)
    df1['Torque [Nm]'] = df1['Torque [Nm]'].astype(int)

    ## Assuming 'df' is your DataFrame and it has been created from your data source

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    scaler = StandardScaler()
    X = scaler.fit_transform(df1[features])
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, df1[failure_types], test_size=0.3, random_state=42)
    
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()

    # Create a Keras model (you can customize this architecture)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(len(failure_types), activation='sigmoid'))

    # Compile the Keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Create a multi-output classifier with the Random Forest and Keras model
    multi_output_classifier = MultiOutputClassifier(estimator=rf_classifier, n_jobs=-1)

    # Create a pipeline
    pipeline = Pipeline([
        ('classifier', multi_output_classifier)
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    user_input_scaled = scaler.transform(user_input)
    prediction = pipeline.predict_proba(user_input_scaled)
    return prediction

def LSTM_predict(user_input, df):
    df2 = df.copy()
    # Drop the columns you don't need in the copied DataFrame
    columns_to_drop = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']  # Replace with actual column names
    df2.drop(columns_to_drop, axis=1, inplace=True)
    # Load the model
    lstm_model = load_model('LSTM_model.keras')
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    failure_types = ['Machine failure']
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df2[features])
    user_input_scaled = scaler.transform(user_input)
    user_input_lstm = user_input_scaled.reshape(user_input_scaled.shape[0], 1, user_input_scaled.shape[1])
    prediction = lstm_model.predict(user_input_lstm)
    return prediction





if __name__ == "__main__":
    main()
