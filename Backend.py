# import necessary libraries
from IPython.display import display
from IPython.display import Markdown
import textwrap
import google.generativeai as genai
import PyPDF2
import os
import urllib
import warnings
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pdf2image
import cv2
import numpy as np
import pytesseract
import re 
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore")

genai.configure(api_key="AIzaSyBAmiLQpQunz3aBgzDY9jjOwBW6jFuDp00") #my api key for google gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyBAmiLQpQunz3aBgzDY9jjOwBW6jFuDp00"
arj_API = os.environ.get('GOOGLE_API_KEY') #api key to a variable

#path to pytesseract in local machine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#function to read the pdf
def img2pdf (pdf_data):
    # pdfs = r"C:\Users\aRj\Desktop\Intern\sample3.pdf"
    # pages=pdf2image.convert_from_bytes(pdf_data)

    
    pdfs = r"C:\Users\aRj\Desktop\Intern\sample1.pdf"
    pages = pdf2image.convert_from_path(pdfs)
    return pages

#function to rotate the image to correct the skew, returning the deskewed image
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# function extracts text from an image using the Tesseract OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# converts PDF data into text
def texto(pdf_data):
    pages=img2pdf (pdf_data)
    ftext=''
    for page in pages:
        # Step 2: Preprocess the image (deskew)
        preprocessed_image = deskew(np.array(page))
        # Step 3: Extract text using OCR
        text = extract_text_from_image(preprocessed_image)
        ftext = ftext + text
        output=preprocess_text(ftext)
    return output

# this function removes extra whitespace and specific unwanted characters
def text_cleaning(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[{}\[\]/\\.*^[[]]]', '', text)
    return text.strip()

#This function takes a text, cleans it, splits it into sentences, tokenizes each sentence into words, and returns the list of tokenized sentences
def preprocess_text(text):
    cleaned_text = text_cleaning(text)
    sentences = sent_tokenize(cleaned_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_sentences
    
# this function generates a text response by querying llm model google gemini
def get_output(query,pdfd):
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=arj_API,

                                   temperature=0.3)

    prompt_template = """" Provide the extract all the text and image text inside the content 
      Context: \n {context}?\n

      Question: \n {question} \n

      Answer:

     """

    prompt = PromptTemplate(

        template=prompt_template, input_variables=["context", "question"]

    )
    
    data= texto(pdfd)
    model = genai.GenerativeModel(model_name="gemini-pro")
    responses = model.generate_content(prompt.format(context=data, question=(query)))
    print(responses.text)
    return responses.text
input= input("Enter the Query  : ")
query = input
get_output(query,"") 
# "" because convert to function having issue because not able to take variable line 29
