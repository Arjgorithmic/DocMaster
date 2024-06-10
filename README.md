# AI-Powered Document Processing Pipeline
This project demonstrates an AI-powered pipeline for understanding and processing documents using advanced NLP techniques, Large Language Models (LLMs), and Optical Character Recognition (OCR). The primary objective is to handle documents that contain both text and image content, extract essential information, and enable user interaction through a chatbot interface.

## Features

- **Document Conversion**: Converts PDF documents to images.
- **OCR Integration**: Extracts text from images using Tesseract OCR.
- **Preprocessing**: Cleans and tokenizes the extracted text for optimal LLM performance.
- **LLM-Powered Understanding**: Uses Google Gemini LLM to process and understand the document content.

## Prerequisites
- Python 3.7+
- Required Python libraries:
  - IPython
  - google-generativeai
  - PyPDF2
  - pdf2image
  - OpenCV
  - numpy
  - pytesseract
  - nltk
  - pandas
  - langchain_core
  - langchain_google_genai

## Functions

### img2pdf(pdf_data)
Converts PDF data to images.

### deskew(image)
Corrects the skew of the given image and returns the deskewed image.

### extract_text_from_image(image)
Extracts text from the given image using Tesseract OCR.

### texto(pdf_data)
Converts PDF data into text by extracting and processing each page.

### text_cleaning(text)
Removes extra whitespace and specific unwanted characters from the text.

### preprocess_text(text)
Cleans the text, splits it into sentences, tokenizes each sentence into words, and returns the list of tokenized sentences.

### get_output(query, pdfd)
Generates a text response by querying the LLM model (Google Gemini) with the given query and PDF data.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Google Generative AI](https://ai.google/)


## Contact

For any questions or inquiries, please contact [arjunpdineshofficial@gmail.com](mailto:arjunpdineshofficial@gmail.com).
