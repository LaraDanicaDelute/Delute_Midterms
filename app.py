from flask import Flask, request, jsonify
from flask import render_template
import numpy as np
import pickle
#flask app
app = Flask(__name__)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 11)
    loaded_model = pickle.load(open("decisiontree_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


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
            prediction ='Income less that 50K'           
        return render_template("index.html", prediction_text = "decision tree model ICU == {}".format(prediction))

    if __name__ == "__main__":
        app.run(debug=True)