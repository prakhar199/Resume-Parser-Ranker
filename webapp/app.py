import os
import requests
import shutil
import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, redirect, request, flash
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlpkit.utils import read_document
from nlpkit.preprocessing import DataPreprocessor

BASE = 'http://127.0.0.1:4444/'

app = Flask(__name__)
app.secret_key = "868356"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
ALLOWED_EXTENSIONS = set(['docx'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_response(response):
    resume = ''
    for values in response.values():
        resume = resume + ' ' + ' '.join(values)

    return resume

def rank_resume(resumes):
    resumes = np.array(resumes)
    dp = DataPreprocessor()
    X = dp.fit_transform(resumes)
    df = pd.DataFrame(columns=['Resume', 'Cosine Similarity'])
    df['Resume'] = X


    tfidf = TfidfVectorizer()
    x_tfidf = tfidf.fit_transform(df['Resume'])
    df['Cosine Similarity'] = cosine_similarity(x_tfidf[0], x_tfidf[0:]).reshape(-1,1)
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def features():
    return render_template('features.html')
    
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/parser-ranker')
def parser_ranker():
    return render_template('parser_ranker.html')

@app.route('/submit', methods=['POST'])
def submit():
    descr = request.form.get('description')

    if 'files[]' not in request.files:
        return redirect(request.url)

    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    files = request.files.getlist('files[]')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    resumes = [descr]
    responses = [[None]]
    for file in files:
        data = {'Resume': read_document(app.config['UPLOAD_FOLDER'] + '/' + file)}
        response = requests.get(BASE + 'parse', data)
        resume = preprocess_response(response.json())
        responses.append([response.json()])
        resumes.append(resume)
        
    shutil.rmtree(app.config['UPLOAD_FOLDER'])
    ranked_resumes = rank_resume(resumes)
    ranked_resumes['Entities'] = np.array([responses]).reshape(-1, 1)
    ranked_resumes = ranked_resumes[['Entities', 'Cosine Similarity']]
    ranked_resumes.sort_values(by='Cosine Similarity', ascending=False, inplace=True)
    return render_template('parser_ranker.html', ranked_resumes=ranked_resumes, descr=descr)

if __name__ == "__main__":
    app.run(debug=True)