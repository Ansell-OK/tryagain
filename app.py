from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.utils import secure_filename
import os
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'

rf_filename = 'rf_model.pkl'

rf_model = pickle.load(open(rf_filename, 'rb'))

ann_filename = 'ann_model.pkl'

ann_model = pickle.load(open(ann_filename, 'rb'))


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/predict_preeclampsia', methods=['GET', 'POST'])
def predict_preeclampsia():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join( 'files', filename))

        file_path = os.path.join('files', filename)

        if filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(file_path)

        data_list = data.columns.tolist()
        feature_list = []

        for i in data_list:
            new_i = float(i)
            feature_list.append(new_i)
        
        input_data = np.array(feature_list).reshape(1, -1)
        predictions = ann_model.predict(input_data)

        if predictions[0] == 0:
            prediction_text_pre = 'Does not have Preeclampsia'
        else:
            prediction_text_pre = 'User has Preeclampsia'
        

    return render_template('predict.html', prediction_text=prediction_text_pre)


@app.route('/predict_diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join( 'files', filename))

        file_path = os.path.join('files', filename)

        if filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(file_path)

        data_list = data.columns.tolist()
        feature_list = []

        for i in data_list:
            new_i = float(i)
            feature_list.append(new_i)
        
        input_data = np.array(feature_list).reshape(1, -1)
        predictions = rf_model.predict(input_data)

        if predictions[0] == 0:
            prediction_text = 'Does not have Gestational Diabetes'
        else:
            prediction_text = 'User has Gestational Diabetes'
        

    return render_template('predict.html', prediction_text=prediction_text)


@app.route('/login', methods=['POST'])
def login_redirect():
    return redirect(url_for('predict_page'))



@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
