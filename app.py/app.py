from flask import Flask, render_template, request
import os
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

bias_terms = {
    "gender": ["she", "he", "homemaker", "housewife", "male", "female"],
    "age": ["older", "retired", "young", "energetic", "youthful", "age"],
    "education": ["IIT", "IIM", "Stanford", "non-target"]
}

def extract_text_from_pdf(path):
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def calculate_ats_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def check_bias(text):
    flags = {}
    for category, words in bias_terms.items():
        found = [w for w in words if w in text.lower()]
        if found:
            flags[category] = found
    return flags

def extract_keywords(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return set([w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2])

def suggest_improvements(resume_text, jd_text):
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)
    missing = jd_kw - resume_kw
    return sorted(missing)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jd_text = request.form['jd']
        file = request.files['resume']

        if file and file.filename.endswith('.pdf'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            resume_text = extract_text_from_pdf(path)
            ats_score = calculate_ats_score(resume_text, jd_text)
            bias_result = check_bias(resume_text)
            suggestions = suggest_improvements(resume_text, jd_text) if ats_score < 80 else []

            return render_template("index.html",
                                   ats_score=ats_score,
                                   bias_result=bias_result,
                                   suggestions=suggestions,
                                   show_results=True)
    return render_template("index.html", show_results=False)

if __name__ == "__main__":
    app.run(debug=True)
