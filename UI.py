import streamlit as st
from Backend import * #1
import tempfile
import os
import pandas as pd
import PyPDF2


# Main app
def main():
    st.title("Deep into Docs")
    user_input = st.text_input("Your Message:")
    #file
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF File, and click Submit!", accept_multiple_files=True)
        if st.button("Submit!"):
            with st.spinner('Processing...'):
                pdf_bit = read_pdf(pdf_docs)
                st.success("Processing Done!")
    if user_input:
        response = get_output(user_input,pdf_bit)  #2
        st.write(response)

def read_pdf(pdf):
    for file in pdf:
        pdf_read = PyPDF2.PdfReader(file)
    return pdf_read



if __name__ == "__main__":
    main()
