# Heart Disease Prediction Web Application

This web application uses machine learning to predict the likelihood of heart disease based on user-provided data.

## Model Preparation

1. **Import Libraries**: Ensure necessary libraries are installed using `pip install streamlit joblib numpy requests streamlit_lottie`.

2. **Load Lottie Animations**: Provide animation URLs accessible by the `load_lottieurl` function.

3. **Load Pre-trained Model**: Load a pre-trained heart disease prediction model using `joblib.load`.

4. **Create Streamlit Web App**: Use `st.title` to set up the Streamlit web application.

5. **Create User Input Fields**: Generate input fields for user data using `st.number_input` and `st.selectbox`.

6. **Convert User Input to Numerical Values**: Convert categorical inputs to numerical values.

7. **Create Feature Array**: Build a feature array using `np.array` containing all user input values in numerical form.

8. **Create Prediction Button**: Use `st.button` to create a button that triggers the prediction.

## Steps to Use the Deployed Model

1. Run the code to start the Streamlit server: `streamlit run your_app_filename.py`.

2. Access the web application in your browser.

3. Input patient data into the provided fields.

4. Click the prediction button to obtain the heart disease prediction.

Feel free to explore the code and customize it according to your needs!
