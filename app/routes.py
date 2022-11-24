from flask import render_template,request,jsonify
from app import app
import pickle
import numpy as np

model=pickle.load(open('modelmeans.pkl','rb')) 
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title='Home',data='hey')

@app.route('/index')
def index():
    return render_template('index.html', title='Home',data='hey')

@app.route("/prediction",methods=["POST"])
def prediction():
    height=float(request.form['height'])
    weight=float(request.form['weight'])
    arr=np.array([[height,weight]])
    pred=model.predict(arr)
    return jsonify({'bmi':str(pred)})

@app.route("/aisha")
def aisha():
    return render_template('test.html', title='aisha code')

if __name__ == "__main__":
    app.run(debug=True)