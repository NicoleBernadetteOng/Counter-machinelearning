import flask
from flask import Blueprint, render_template, abort, url_for
from jinja2 import TemplateNotFound
import pickle 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode

#==================================================================
connection = mysql.connector.connect(
    host='localhost', user='root', database='counterdb'
)

mycursor = connection.cursor()

connection_config_dict = {
    'user': 'root',
    'host': 'localhost',
    'database': 'counterdb'
}

# SELECT AND GET DATA FROM DATABASE
sql_select_Query = "SELECT * FROM breast_table;"
mycursor.execute(sql_select_Query)
records = mycursor.fetchall()
print("Total number of rows in breast_table is - ", mycursor.rowcount)
# for row in records:
#     print("BI-RADS = ", row[0], )
#     print("Age = ", row[1])
#     print("Shapee  = ", row[2])
#     print("Margin  = ", row[3], "\n")
#     mycursor.close()

try:
    connection = mysql.connector.connect(**connection_config_dict)
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Succesfully Connected to MySQL database. MySQL Server version on ", db_Info)
except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if (connection.is_connected()):
        connection.close()


#==================================================================

# Use pickle to load in the model which has been trained 
# Mammo
with open(f'models/trained_mammographic_model.pkl', 'rb') as f:
    mammo_model = pickle.load(f)

# Heart
with open(f'models/trained_heart_attack_model.pkl', 'rb') as f:
    heart_model = pickle.load(f)

# Lung
with open(f'models/trained_lung_cancer_model.pkl', 'rb') as f:
    lung_model = pickle.load(f)

# Prostate
with open(f'models/trained_prostate_cancer_model.pkl', 'rb') as f:
    prostate_model = pickle.load(f)

app = flask.Flask(__name__, static_folder='static', template_folder='templates')

    
@app.route('/')
def main():
    return(flask.render_template('index.html'))
    

if __name__ == '__main__':
    app.run()

# ===================================================================
@app.route('/mammo.html', methods=['GET', 'POST'])
def mammo():

    if flask.request.method == 'GET':
        return(flask.render_template('mammo.html'))


    if flask.request.method == 'POST':
        birads = flask.request.form['birads']
        age = flask.request.form['age']
        margin = flask.request.form['margin']
        density = flask.request.form['density']

        # ===== GET DATA FROM CSV ===
        # # Get data from csv
        # mammo_data = pd.read_csv("data/new_mammographic_masses.csv")
        # # Create array of only the feature data 
        # features = mammo_data[['birads', 'age', 'margin', 'density']].values
        # ===== GET DATA FROM CSV ===

        # ===== GET DATA FROM DATABASE ===
        # OR Get data from MySQL database
        mammo_data = pd.DataFrame(records, columns = ['id', 'BI-RADS' , 'Age', 'Margin', 'Density', 'Severity'])
        # Create array of only the feature data 
        features = mammo_data[['BI-RADS', 'Age', 'Margin', 'Density']].values
        # ===== GET DATA FROM DATABASE ===

        prediction = []

        # Scale the new data
        scaler = preprocessing.StandardScaler()
        features_scaled = scaler.fit_transform(features)
        input_variables = np.array([[birads, age, margin, density]], dtype=float)
        data_scaled = scaler.transform(input_variables)
        prediction = mammo_model.predict(data_scaled)

        # print the prediction result in the console
        print(prediction)

        # Change the numpy array to pandas dataframe
        input_variables = pd.DataFrame([[birads, age, margin, density]],
                                       columns=['birads', 'age', 'margin', 'density'],
                                       dtype=float)

        return flask.render_template('mammo.html',
                                     original_input={'birads':birads,
                                                     'age':age,
                                                     'margin':margin,
                                                     'density':density},
                                     result=prediction,
                                     )

if __name__ == '__mammo__':
    app.run()


# =======================================================================================
@app.route('/heart.html', methods=['GET', 'POST'])
def heart():

    if flask.request.method == 'GET':
        return(flask.render_template('heart.html'))

    if flask.request.method == 'POST':
        heart_age = flask.request.form['heart_age']
        heart_sex = flask.request.form['heart_sex']
        cp = flask.request.form['cp']
        trestbps = flask.request.form['trestbps']
        restecg = flask.request.form['restecg']
        thalach = flask.request.form['thalach']
        exang = flask.request.form['exang']
        oldpeak = flask.request.form['oldpeak']
        slope = flask.request.form['slope']
        ca = flask.request.form['ca']
        thal = flask.request.form['thal']
   
        # Get data 
        heart_data = pd.read_csv("data/new_heart_attack.csv")
        # Create array of only the feature data 
        heart_features = heart_data[['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].values

        heart_prediction = []

        # Scale the new data
        heart_scaler = preprocessing.StandardScaler()
        heart_features_scaled = heart_scaler.fit_transform(heart_features)

        heart_input_variables = np.array([[heart_age, heart_sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope, ca, thal]], dtype=float)
        heart_data_scaled = heart_scaler.transform(heart_input_variables)
        heart_prediction = heart_model.predict(heart_data_scaled)

        print (heart_data_scaled)
        # print the prediction result in the console
        print(heart_prediction)

        # Change the numpy array to pandas dataframe
        heart_input_variables = pd.DataFrame([[heart_age, heart_sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                       columns=['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
                                       dtype=float)

        return flask.render_template('heart.html',
                                        heart_original_input={'age':heart_age,
                                                        'sex':heart_sex,
                                                        'cp':cp,
                                                        'trestbps':trestbps,
                                                        'restecg':restecg,
                                                        'thalach':thalach,
                                                        'exang':exang,
                                                        'oldpeak':oldpeak,
                                                        'slope':slope,
                                                        'ca':ca,
                                                        'thal':thal},
                                                        heart_result=heart_prediction,
                                                        )


if __name__ == '__heart__':
    app.run()

# =======================================================================================
@app.route('/lung.html', methods=['GET', 'POST'])
def lung():

    if flask.request.method == 'GET':
        return(flask.render_template('lung.html'))

    if flask.request.method == 'POST':
        lung_age = flask.request.form['lung_age']
        smokes = flask.request.form['smokes']
        areaq = flask.request.form['areaq']
        alkhol = flask.request.form['alkhol']
            
        # Get data 
        lung_data = pd.read_csv("data/new_lung_cancer.csv")
        # Create array of only the feature data 
        lung_features = lung_data[['Age', 'Smokes', 'AreaQ', 'Alkhol']].values

        lung_prediction = []

        # Scale the new data
        lung_scaler = preprocessing.StandardScaler()
        lung_features_scaled = lung_scaler.fit_transform(lung_features)
        lung_input_variables = np.array([[lung_age, smokes, areaq, alkhol]], dtype=float)
        lung_data_scaled = lung_scaler.transform(lung_input_variables)
        lung_prediction = lung_model.predict(lung_data_scaled)

        # print the prediction result in the console
        print(lung_prediction)

        # Change the numpy array to pandas dataframe
        lung_input_variables = pd.DataFrame([[lung_age, smokes, areaq, alkhol]],
                                       columns=['Age', 'Smokes', 'AreaQ', 'Alkhol'],
                                       dtype=float)

        return flask.render_template('lung.html',
                                        lung_original_input={'Age':lung_age,
                                                        'Smokes':smokes,
                                                        'AreaQ':areaq,
                                                        'Alkhol':alkhol},
                                                        lung_result=lung_prediction,
                                                        )


if __name__ == '__lung__':
    app.run()

# =======================================================================================
@app.route('/prostate.html', methods=['GET', 'POST'])
def prostate():

    if flask.request.method == 'GET':
        return(flask.render_template('prostate.html'))

    if flask.request.method == 'POST':
        radius = flask.request.form['radius']
        perimeter = flask.request.form['perimeter']
        smoothness = flask.request.form['smoothness']
        compactness = flask.request.form['compactness']
        symmetry = flask.request.form['symmetry']  

        # Get data 
        prostate_data = pd.read_csv("data/new_prostate_cancer.csv")
        # Create array of only the feature data 
        prostate_features = prostate_data[['radius', 'perimeter', 'smoothness', 'compactness', 'symmetry']].values

        prostate_prediction = []

        # Scale the new data
        prostate_scaler = preprocessing.StandardScaler()
        prostate_features_scaled = prostate_scaler.fit_transform(prostate_features)
        prostate_input_variables = np.array([[radius, perimeter, smoothness, compactness, symmetry]], dtype=float)
        prostate_data_scaled = prostate_scaler.transform(prostate_input_variables)
        prostate_prediction = prostate_model.predict(prostate_data_scaled)

        # print the prediction result in the console
        print(prostate_prediction)

        # Change the numpy array to pandas dataframe
        prostate_input_variables = pd.DataFrame([[radius, perimeter, smoothness, compactness, symmetry]],
                                       columns=['radius', 'perimeter', 'smoothness', 'compactness', 'symmetry'],
                                       dtype=float)

        return flask.render_template('prostate.html',
                                        prostate_original_input={'radius':radius,
                                                        'perimeter':perimeter,
                                                        'smoothness':smoothness,
                                                        'compactness':compactness,
                                                        'symmetry':symmetry},
                                                        prostate_result=prostate_prediction,
                                                        )


if __name__ == '__prostate__':
    app.run()
