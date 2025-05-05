import streamlit as st  

def execute_python_file(file_path):
    st.write(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            python_code = file.read()
            exec(python_code)
    except FileNotFoundError:
        st.markdown(f"Error: The file '{file_path}' does not exist.")

