# 📰 Semantic Ambiguity Analyzer

A Streamlit-based NLP tool that detects semantic ambiguity in newspaper texts using 8 different analysis functions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![spaCy](https://img.shields.io/badge/spaCy-3.7-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

Automatically detects and visualizes ambiguity in newspaper articles. Identifies words with multiple meanings, unclear references, vague expressions, and more across 8 different ambiguity types.

---

## ✨ Features

- 8 Ambiguity Detection Functions
- Visual Bar Charts
- Auto Entity Extraction
- Compare Mode
- Multiple Input Methods (.txt, .pdf, .docx)
- Analysis History
- Clean UI with Severity Indicators

---

## 🛠️ Tech Stack

Python, Streamlit, spaCy, Matplotlib, Pandas, PyPDF2, python-docx

---

## 🚀 Installation
git clone https://github.com/muhammadhaider/NLP_Project.git
cd NLP_Project
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py

text

---

## 📂 Project Structure
NLP_Project/
├── app.py
├── analyzer.py
├── papers.py
├── utils.py
└── requirements.txt

text

---

## 🧪 8 Ambiguity Types

Lexical, Referential, Scope, Attachment, Quantifier, Temporal, Contextual, Entity Overlap

---

## 📊 Severity Levels

HIGH (Red, 6+), MEDIUM (Amber, 3-5), LOW (Green, 0-2)

---

## 💡 Usage

Single Analysis or Compare Mode - select input, click analyze, view results

---

## 👤 Author

**Muhammad Haider** | CSC-23S-061

---

## 📝 License

MIT
