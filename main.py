from flask import Flask, request, render_template, redirect, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import util
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

application = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl','rb'))
util.load_artifacts()


#home page
@application.route("/",methods=["GET", "POST"])
def index():
    return render_template("index.html")

@application.route("/home",methods=["GET", "POST"])
def home():
    return render_template("home.html")

@application.route("/form",methods=["GET", "POST"])
def form():
    return render_template("form.html")

@application.route("/analytics",methods=["GET", "POST"])
def analytics():
    return render_template("analytics.html")




@application.route("/predict", methods=["GET", "POST"])
def predict():
    Fuel_Type_Diesel = 0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Kms_Driven2 = np.log(Kms_Driven)
        Owner = int(request.form['Owner'])
        Fuel_Type_Petrol = request.form['Fuel_Type_Petrol']
        if (Fuel_Type_Petrol == 'Petrol'):
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
        Year = 2020 - Year
        Seller_Type_Individual = request.form['Seller_Type_Individual']
        if (Seller_Type_Individual == 'Individual'):
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0
        Transmission_Mannual = request.form['Transmission_Mannual']
        if (Transmission_Mannual == 'Mannual'):
            Transmission_Mannual = 1
        else:
            Transmission_Mannual = 0
        prediction = model.predict([[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                     Seller_Type_Individual, Transmission_Mannual]])
        output = round(prediction[0], 2)
        if output < 0:
            return render_template('predict.html', prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('predict.html', prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('predict.html')

@application.route("/login",methods=["GET", "POST"])
def login():
    return render_template("login.html")

@application.route("/classifywaste", methods = ["GET","POST"])
def classifywaste():
    image_data = request.files["file"]
    #save the image to upload
    basepath = os.path.dirname(__file__)
    image_path = os.path.join(basepath, "uploads", secure_filename(image_data.filename))
    image_data.save(image_path)

    predicted_value = util.classify_waste(image_path)
    os.remove(image_path)
    return jsonify(predicted_value=predicted_value)

if __name__ == '__main__':
    application.run(debug=True)