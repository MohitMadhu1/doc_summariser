# Lecture AI Assistant – AI-Powered Lecture Companion

## Overview

**Lecture AI Assistant** is an intelligent tool designed to transform lecture materials into easily digestible insights for students. Unlike traditional tools that process only a single file or limited formats, Lecture AI Assistant supports **multiple file types**, allows **batch uploads**, and enables **cross-document search** for maximum productivity.

## Key Features

- **Multi-Format Support**: Seamlessly works with various document types:
  - PDF, TXT, DOCX, EML, PNG, JPG, JPEG
- **Batch Uploads**: Upload and analyze multiple documents at once — no need to process files one by one.
- **Cross-Document Search**: Ask questions and retrieve answers from **across all uploaded files**, not just a single document.
- **Concise Summaries**: Generate brief and informative summaries of lecture notes or slides.
- **Key Takeaways**: Highlight and extract the most important concepts covered in lectures.
- **FAQ-style Answers**: Generate frequently asked questions with concise answers.
- **Practice Questions & Answers**: Automatically create practice questions and answers to aid in exam prep.

## Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **AI & NLP**: LLM APIs, Text Summarization, Question Generation, **Groq**  
- **Document Parsing**: **Unstructured** for PDF, TXT, DOCX, images, and email parsing

## Demo Video

▶️ Watch the Demo: [Lecture AI Assistant – Video Demo](https://drive.google.com/file/d/15nvdDYuf-kkKa6avhMdMp5RztvxSyrgR/view?usp=drive_link)

## Try the Live App

🚀 Explore the Deployed App: [Lecture AI Assistant – Streamlit App](https://docsummariser-mywzqmjoqbjdihesrw2gzm.streamlit.app/)

## Getting Started

> ⚠️ **Note**: Some dependencies may not be compatible with **Python 3.13**. It is recommended to use **Python 3.10 or 3.11** for best compatibility.

1. Create and activate a virtual environment:

    ```sh
    python -m venv .venv
    # On Linux/MacOS
    source .venv/bin/activate
    # On Windows
    .venv\\Scripts\\activate
    ```

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Run the application:

    ```sh
    streamlit run try.py
    ```
