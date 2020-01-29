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

@app.route('/bio/DHenderson',methods=['GET'])
def DHenderson():
    return render_template('bio_DHenderson.html',bio='active')

@app.route('/bio/GAvetisyan',methods=['GET'])
def GAvetisyan():
    return render_template('bio_GAvetisyan.html',bio='active')

@app.route('/bio/ABuckalew',methods=['GET'])
def ABuckalew():
    return render_template('bio_ABuckalew.html',bio='active')

@app.route('/bio/MSriqui',methods=['GET'])
def MSriqui():
    return render_template('bio_MSriqui.html',bio='active')

@app.route('/bio/SWright',methods=['GET'])
def SWight():
    return render_template('bio_SWright.html',bio='active')