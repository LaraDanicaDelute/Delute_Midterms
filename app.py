from flask import Flask, request, jsonify
from flask import render_template
import numpy as np
import pickle
#flask app
app = Flask(__name__)

model = pickle.load(open("decisiontree_model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)       
        if int(result)== 0:
            prediction ='u r likely to be put in ICU'
        if int(result)== 1:
            prediction ='u r likely not to be put in ICU'
        else:
            prediction ='not-specified'           
        return render_template("index.html", prediction_text = "decision tree model ICU == {}".format(prediction))

    if __name__ == "__main__":
        app.run(debug=True)