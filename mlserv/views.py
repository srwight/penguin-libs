from flask import render_template, request
from mlserv import app

@app.route('/', methods=['GET'])
def home():
    """Renders the home page using the base index template."""
    return render_template('index.html', home='active')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', about='active')

@app.route('/bio', methods=['GET'])
def bio():
    return render_template('bio.html', bio='active')

@app.route('/predict', methods=['GET', 'POST'])
def predictor():
    return render_template('predict.html', predictor='active')