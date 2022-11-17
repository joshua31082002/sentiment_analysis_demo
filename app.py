from flask import Flask, render_template, request, redirect, url_for
import os
# from werkzeug import secure_filename
import pickle
import re
import pandas as pd
from sklearn.pipeline import Pipeline
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

word_lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

def preprocess(sentence):
    emoji_dict = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
                    ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
                    ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', r':\\': 'annoyed', 
                    ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
                    '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
                    '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
                    ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
    
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = re.sub('<[^>]*>',' tag ',sentence)
    sentence = re.sub(r'((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)', ' url ',sentence)
    sentence = re.sub('@[^\s]+>',' USER ',sentence)
    sentence = re.sub('[^a-zA-Z0-9]',' ',sentence) 
    for emoji in emoji_dict.keys():
        sentence = sentence.replace(emoji, " EMOJI "+emoji_dict[emoji])
    sentence = re.sub(r"(.)\1\1+",r"\1\1",sentence)
    
    return sentence

def lemmatizer(sentence):
    return ''.join([word_lemmatizer.lemmatize(word) for word in sentence])

def stop_words_remover(sentence):
    sentence = str(sentence)
    sentence = ''.join(sentence)
    stopwords = nlp.Defaults.stop_words
    new_sent = ''
    for word_token in sentence.split():
        if word_token not in stopwords:
            new_sent = new_sent + word_token + ' '
    return new_sent

class DataCleaner(BaseEstimator,TransformerMixin):
    def __init__(self,X=None,y=None):
        self.X = X
        self.y = y
        
    
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X_ = X.copy()
        for row in X_.iteritems():
            row = preprocess(row)
            row = stop_words_remover(row)
            row = lemmatizer(row)
        return X_

tfidf = TfidfVectorizer(
                        ngram_range=(1,2),
                        max_features=500000,
                        )

@app.route("/",methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template("index.html")
    elif request.method=='POST':
        return redirect(url_for('predictStatement'))

@app.route("/index.html",methods=['GET'])
def go_to_index():
    if request.method=='GET':
        return render_template("index.html")

@app.route("/contact-me.html",methods=['GET'])
def go_to_contact():
    if request.method=='GET':
        return render_template("contact-me.html")

@app.route("/about.html",methods=['GET'])
def go_to_about():
    if request.method=='GET':
        return render_template("about.html")

@app.route("/input",methods=['GET','POST'])
def predictStatement():    
    if request.method=='GET':
        return render_template("input.html")
    elif request.method=='POST':
        stmt,prediction = '','No input have been provided :('
        stmt = request.form['statement']
        if(stmt and stmt.strip()):
            X_test_tr =pipe.transform(pd.Series(stmt))
            predicted_label = multinomialnb.predict(X_test_tr)[0]
            prediction= 'Positive statement'
            if predicted_label==0:
                prediction = 'Negative statement'
            
            return redirect(url_for('result_single',stmt=stmt,prediction=prediction)) 
        elif 'fileButton' in request.files:
            uploaded_file = request.files['fileButton']
            if uploaded_file:
                uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename))
                df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename),header=None)
                df.columns=['statements']
                X_test_tr =pipe.transform(df['statements'])
                predicted_label = multinomialnb.predict(X_test_tr)
                temp_dict = {'statement':df['statements'].values,
                            'sentiment prediction': predicted_label
                            }
                result = pd.DataFrame(temp_dict)
                result['sentiment prediction'] = result['sentiment prediction'].apply(lambda x: 'Negative' if x==0 else 'Positive')
                result.to_csv(r'static\downloads\result_file.csv')
            return redirect(url_for('result_file'))

@app.route("/result_single/<stmt>/<prediction>",methods=['GET'])
def result_single(stmt,prediction):
    if request.method=='GET':
        return render_template("result_single.html",statement=stmt,prediction=prediction)

@app.route("/result_single",methods=['POST'])
def redirect_index():
    if request.method=='POST':
        return redirect(url_for('index'))

@app.route('/result_file',methods=['GET'])
def result_file():
    if request.method=='GET':
        return render_template("result_file.html")

@app.route("/result_file",methods=['POST'])
def redirect_index1():
    if request.method=='POST':
        return redirect(url_for('index'))

if __name__=='__main__':
    UPLOAD_FOLDER = 'uploads'
    DOWNLOAD_FOLDER = 'downloads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
    multinomialnbCV = pickle.load(open('picklefiles\multinomialnb_best_model.pickle','rb'))
    multinomialnb = multinomialnbCV.best_estimator_
    pipe = pickle.load(open('picklefiles\pipe_fitted.pickle','rb'))
    app.debug=True
    app.run()