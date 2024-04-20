import streamlit as st

def main():
    st.title("Equipment Failure Prediction")
    selected_page = st.page_link("app.py", label="Home", icon="🏠")
    selected_page = st.page_link('pages/dataVisualization.py', label="Data Visualization And Analysis", icon="📊")
    selected_page = st.page_link('pages/prediction.py', label="Prediction", icon="⚙")

    

if __name__ == "__main__":
    main()
