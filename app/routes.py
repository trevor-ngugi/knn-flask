from flask import render_template,request
from app import app
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb')) 
modeltwo=pickle.load(open('modelfix.pkl','rb')) 
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title='Home',data='hey')

@app.route('/index')
def index():
    return render_template('index.html', title='Home',data='hey')

@app.route('/aisha')
def trial():
    return render_template('aisha.html', title='with gender',data='hey')

@app.route("/predictiontwo",methods=["POST"])
def predictiontwo():
    height=float(request.form['height'])
    weight=float(request.form['weight'])
    gender=float(request.form['gender'])
    arr=np.array([[height,weight,gender]])
    pred=modeltwo.predict(arr)
    return render_template('aishapred.html',data=pred)

@app.route("/prediction",methods=["POST"])
def prediction():
    height=float(request.form['height'])
    weight=float(request.form['weight'])
    arr=np.array([[height,weight]])
    pred=model.predict(arr)
    return render_template('prediction.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)