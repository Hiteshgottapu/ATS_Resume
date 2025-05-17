# ATS Resume Matcher

ATS Resume Matcher is a Flask web application that uses advanced Natural Language Processing (NLP) to analyze and compare resumes against job descriptions. It helps users understand how well their resume matches a given job posting, highlights matched and missing skills, and suggests related skills that may strengthen their application.

## Features

- **Upload or Paste Resume**: Supports PDF, DOCX, and TXT files, or direct text input.
- **Paste Job Description**: Analyze any job description text.
- **NLP-Powered Extraction**: Uses spaCy and fuzzy matching to extract skills, experience, education, and projects.
- **ATS Score**: Calculates a matching score based on skills, experience, and education.
- **Matched & Unmatched Skills**: Clearly shows which job requirements are met and which are missing.
- **Related Skills Suggestions**: Identifies related skills present in the resume for missing requirements.
- **User-Friendly UI**: Clean, responsive interface for easy use.

## Project Structure

```
ATS_resume/
├── app.py                  # Main Flask application
├── templates/
│   ├── index.html          # Main input form
│   └── results.html        # Results display page
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### 1. Clone the Repository

```sh
git clone <repository-url>
cd ATS_resume
```

### 2. Set Up a Virtual Environment (Recommended)

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Download spaCy Model (if not auto-downloaded)

```sh
python -m spacy download en_core_web_sm
```

### 5. Run the Application

```sh
python app.py
```

### 6. Open in Browser

Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser.

## Usage

1. **Upload your resume** (PDF, DOCX, or TXT) or paste the text.
2. **Paste the job description** you want to match against.
3. **Set a score threshold** (optional).
4. **Submit** to see your ATS score, matched/unmatched skills, related skills, and more.

## Dependencies

- Flask
- spaCy
- fuzzywuzzy
- python-Levenshtein (for fuzzywuzzy performance)
- python-docx
- PyPDF2

See `requirements.txt` for the full list.

## Contributing

Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.