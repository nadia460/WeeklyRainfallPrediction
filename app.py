import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from asyncio.windows_utils import PipeHandle
from flask import Flask, request, render_template
from pylab import *
from sklearn import preprocessing
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import model_from_json

from tensorflow.python.keras.models import load_model

tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
build = tf.sysconfig.get_build_info()
print(build['cuda_version'])
print(build['cudnn_version'])

# configuration
DEBUG = True

#import lib
matplotlib.use('Agg')

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.static_folder = 'static'

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DATABASE = 'static/database'
app.config['DATABASE'] = DATABASE


@app.route("/")
def beranda():
    return render_template('beranda2.html', judul='beranda')

@app.route("/panduan/")
def panduan():
    return render_template('panduan2.html', judul='panduan')

@app.route("/prediksi/")
def prediksi():
    return render_template('prediksi2.html', judul='prediksi')
PipeHandle

@app.route("/prediksi/", methods=['POST'])
def loadData():
    uploaded_file1 = request.files['file']

    if uploaded_file1.filename != '':
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], uploaded_file1.filename)
        # set the file path
        uploaded_file1.save(file_path)
        # save the file
        fileName1 = uploaded_file1.filename
    
    uploaded_file2 = request.files['file2']

    if uploaded_file2.filename != '':
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], uploaded_file2.filename)
        # set the file path
        uploaded_file2.save(file_path)
        # save the file
        fileName2 = uploaded_file2.filename
    
    uploaded_file3 = request.files['file3']

    if uploaded_file3.filename != '':
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], uploaded_file3.filename)
        # set the file path
        uploaded_file3.save(file_path)
        # save the file
        fileName3 = uploaded_file3.filename
    
    return classification(fileName1,fileName2,fileName3)

def classification(fileName1,fileName2,fileName3):
    global datalatih1, datalatih2, datalatih3

    # Load dataset
    datalatih1 = pd.read_csv('static/files/'+fileName1, header=0,
                                 parse_dates=['Tanggal'], index_col='Tanggal', dayfirst=True)
    datalatih2 = pd.read_csv('static/files/'+fileName2, header=0,
                                 parse_dates=['Tanggal'], index_col='Tanggal', dayfirst=True)
    datalatih3 = pd.read_csv('static/files/'+fileName3, header=0,
                                 parse_dates=['Tanggal'], index_col='Tanggal', dayfirst=True)
    
    # Interpolasi
    hasil1 = datalatih1.replace({'SU' : [9999,8888,0.0], 'KU' : [9999,8888,0.0], 'CH':[9999,8888,0.0], 'KA':[9999,8888,0.0], 
                                    'SOI':[9999,8888,0.0]}, np.nan)
    interpolasi1 = hasil1.interpolate()
    interpolasi1 = round(interpolasi1, 3)
    hasil2 = datalatih2.replace({'SU' : [9999,8888,0.0], 'KU' : [9999,8888,0.0], 'CH':[9999,8888,0.0], 'KA':[9999,8888,0.0], 
                                    'SOI':[9999,8888,0.0]}, np.nan)
    interpolasi2 = hasil2.interpolate()
    interpolasi2 = round(interpolasi2, 3)
    hasil3 = datalatih3.replace({'SU' : [9999,8888,0.0], 'KU' : [9999,8888,0.0], 'CH':[9999,8888,0.0], 'KA':[9999,8888,0.0], 
                                    'SOI':[9999,8888,0.0]}, np.nan)
    interpolasi3 = hasil3.interpolate()
    interpolasi3 = round(interpolasi3, 3)

    # Konversi
    #df1
    ekstraksi1 = interpolasi1.resample('7D').max()
    max1 = ekstraksi1.max()
    min1 = ekstraksi1.min()
    #df2
    ekstraksi2 = interpolasi2.resample('7D').max()
    max2 = ekstraksi2.max()
    min2 = ekstraksi2.min()
    #df3
    ekstraksi3 = interpolasi3.resample('7D').max()
    max3 = ekstraksi3.max()
    min3 = ekstraksi3.min()

    # Normalisasi
    #df1
    MinMax_scaler1 = preprocessing.MinMaxScaler()
    ekstraksi1['SU'] = MinMax_scaler1.fit_transform(ekstraksi1['SU'].values.reshape(-1,1))
    ekstraksi1['KU'] = MinMax_scaler1.fit_transform(ekstraksi1['KU'].values.reshape(-1,1))
    ekstraksi1['CH'] = MinMax_scaler1.fit_transform(ekstraksi1['CH'].values.reshape(-1,1))
    ekstraksi1['KA'] = MinMax_scaler1.fit_transform(ekstraksi1['KA'].values.reshape(-1,1))
    ekstraksi1['SOI'] = MinMax_scaler1.fit_transform(ekstraksi1['SOI'].values.reshape(-1,1))
    #df2
    MinMax_scaler2 = preprocessing.MinMaxScaler()
    ekstraksi2['SU'] = MinMax_scaler2.fit_transform(ekstraksi2['SU'].values.reshape(-1,1))
    ekstraksi2['KU'] = MinMax_scaler2.fit_transform(ekstraksi2['KU'].values.reshape(-1,1))
    ekstraksi2['CH'] = MinMax_scaler2.fit_transform(ekstraksi2['CH'].values.reshape(-1,1))
    ekstraksi2['KA'] = MinMax_scaler2.fit_transform(ekstraksi2['KA'].values.reshape(-1,1))
    ekstraksi2['SOI'] = MinMax_scaler2.fit_transform(ekstraksi2['SOI'].values.reshape(-1,1))
    #df3
    MinMax_scaler3 = preprocessing.MinMaxScaler()
    ekstraksi3['SU'] = MinMax_scaler3.fit_transform(ekstraksi3['SU'].values.reshape(-1,1))
    ekstraksi3['KU'] = MinMax_scaler3.fit_transform(ekstraksi3['KU'].values.reshape(-1,1))
    ekstraksi3['CH'] = MinMax_scaler3.fit_transform(ekstraksi3['CH'].values.reshape(-1,1))
    ekstraksi3['KA'] = MinMax_scaler3.fit_transform(ekstraksi3['KA'].values.reshape(-1,1))
    ekstraksi3['SOI'] = MinMax_scaler3.fit_transform(ekstraksi3['SOI'].values.reshape(-1,1))

    # Segmentasi
    ekstraksi1 = ekstraksi1.values
    ekstraksi2 = ekstraksi2.values
    ekstraksi3 = ekstraksi3.values
    n = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    num = 0
    segmen1 = np.empty(shape=(len(ekstraksi1) - 11, 5, 12))
    segmen2 = np.empty(shape=(len(ekstraksi2) - 11, 5, 12))
    segmen3 = np.empty(shape=(len(ekstraksi3) - 11, 5, 12))
    col2 = 0
    col3 = 0
    col4 = 0
    col5 = 0
    col6 = 0
    # Segmentasi df1
    for row in range(len(ekstraksi1) - 11):
        for col in range(60):
            if (num >= 0 and num <= 11) :
                segmen1[row][0][col2] = round(ekstraksi1[row + n][0], 2)
                num = num + 1
                n = n + 1
                col2 = col2 + 1
            elif (num >= 12 and num <= 23) :
                n = 0
                col2 = 0
                segmen1[row][1][col3] = round(ekstraksi1[row + n2][1], 2)
                num = num + 1
                n2 = n2 + 1
                col3 = col3 + 1
            elif (num >= 24 and num <= 35) :
                n2 = 0
                col3 = 0
                segmen1[row][2][col4] = round(ekstraksi1[row + n3][2], 2)
                num = num + 1
                n3 = n3 + 1
                col4 = col4 + 1
            elif (num >= 36 and num <= 47) :
                n3 = 0
                col4 = 0
                segmen1[row][3][col5] = round(ekstraksi1[row + n4][3], 2)
                num = num + 1
                n4 = n4 + 1
                col5 = col5 + 1
            elif (num >= 48) :
                n4 = 0
                col5 = 0
                segmen1[row][4][col6] = round(ekstraksi1[row + n5][4], 2)
                num = num + 1
                n5 = n5 + 1
                col6 = col6 + 1
        n5 = 0
        num = 0
        col6 = 0
    # Segmentasi df2
    for row in range(len(ekstraksi2) - 11):
        for col in range(60):
            if (num >= 0 and num <= 11) :
                segmen2[row][0][col2] = round(ekstraksi2[row + n][0], 2)
                num = num + 1
                n = n + 1
                col2 = col2 + 1
            elif (num >= 12 and num <= 23) :
                n = 0
                col2 = 0
                segmen2[row][1][col3] = round(ekstraksi2[row + n2][1], 2)
                num = num + 1
                n2 = n2 + 1
                col3 = col3 + 1
            elif (num >= 24 and num <= 35) :
                n2 = 0
                col3 = 0
                segmen2[row][2][col4] = round(ekstraksi2[row + n3][2], 2)
                num = num + 1
                n3 = n3 + 1
                col4 = col4 + 1
            elif (num >= 36 and num <= 47) :
                n3 = 0
                col4 = 0
                segmen2[row][3][col5] = round(ekstraksi2[row + n4][3], 2)
                num = num + 1
                n4 = n4 + 1
                col5 = col5 + 1
            elif (num >= 48) :
                n4 = 0
                col5 = 0
                segmen2[row][4][col6] = round(ekstraksi2[row + n5][4], 2)
                num = num + 1
                n5 = n5 + 1
                col6 = col6 + 1
        n5 = 0
        num = 0
        col6 = 0
    #segmentasi df3
    for row in range(len(ekstraksi3) - 11):
        for col in range(60):
            if (num >= 0 and num <= 11) :
                segmen3[row][0][col2] = round(ekstraksi3[row + n][0], 2)
                num = num + 1
                n = n + 1
                col2 = col2 + 1
            elif (num >= 12 and num <= 23) :
                n = 0
                col2 = 0
                segmen3[row][1][col3] = round(ekstraksi3[row + n2][1], 2)
                num = num + 1
                n2 = n2 + 1
                col3 = col3 + 1
            elif (num >= 24 and num <= 35) :
                n2 = 0
                col3 = 0
                segmen3[row][2][col4] = round(ekstraksi3[row + n3][2], 2)
                num = num + 1
                n3 = n3 + 1
                col4 = col4 + 1
            elif (num >= 36 and num <= 47) :
                n3 = 0
                col4 = 0
                segmen3[row][3][col5] = round(ekstraksi3[row + n4][3], 2)
                num = num + 1
                n4 = n4 + 1
                col5 = col5 + 1
            elif (num >= 48) :
                n4 = 0
                col5 = 0
                segmen3[row][4][col6] = round(ekstraksi3[row + n5][4], 2)
                num = num + 1
                n5 = n5 + 1
                col6 = col6 + 1
        n5 = 0
        num = 0
        col6 = 0

    # Menggabungkan ke-3 segmen dataset (spasial tiga stasiun)
    stasiun1 = segmen1.reshape(-1, 5, 1, 12)
    stasiun2 = segmen2.reshape(-1, 5, 1, 12)
    stasiun3 = segmen3.reshape(-1, 5, 1, 12)

    # Dimensi yang masuk ke pembelajaran
    segmen = np.dstack((stasiun1,stasiun2,stasiun3))
    train = segmen 
    X_train = train[:, :]
    y_train = train[:, :][:,2] # Target, mengambil nilai curah hujan (yang berada pada indeks ke-2)

    #split data latih dan tes
    splitRow = round(1 * X_train.shape[0])
    train = segmen [:int(splitRow), :]

    weight_h5 = "model/new4_model.h5"
    weight_json = "model/new4_model.json"
    json_file = open(weight_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_h5)
    # training
    y_pred = loaded_model.predict(X_train)

    # Stasiun 1
    hasil1 = y_pred[-1, 0, -1]*(max1['CH']-min1['CH'])+min1['CH']

    if(hasil1 > 0.5 and hasil1 <= 20):
        prediksi1 = "Hujan Ringan"
    elif(hasil1 > 20 and hasil1 <=  50):
        prediksi1 = "Hujan Sedang"
    elif(hasil1 >  50 and hasil1 <= 100):
        prediksi1 = "Hujan Lebat"
    elif(hasil1 > 100 ):
        prediksi1 = "Hujan Sangat Lebat"
    else:
        prediksi1 = "Berawan"

    # Stasiun 2
    hasil2 = y_pred[-1, 1, -1]*(max2['CH']-min2['CH'])+min2['CH']
    if(hasil2 > 0.5 and hasil2 <= 20):
        prediksi2 = "Hujan Ringan"
    elif(hasil2 > 20 and hasil2 <=  50):
        prediksi2 = "Hujan Sedang"
    elif(hasil2 >  50 and hasil2 <= 100):
        prediksi2 = "Hujan Lebat"
    elif(hasil2 > 100 ):
        prediksi2 = "Hujan Sangat Lebat"
    else:
        prediksi2 = "Berawan"

    # Stasiun 3
    hasil3 = y_pred[-1, 2, -1]*(max3['CH']-min3['CH'])+min3['CH']
    if(hasil3 > 0.5 and hasil3 <= 20):
        prediksi3 = "Hujan Ringan"
    elif(hasil3 > 20 and hasil3 <=  50):
        prediksi3 = "Hujan Sedang"
    elif(hasil3 >  50 and hasil3 <= 100):
        prediksi3 = "Hujan Lebat"
    elif(hasil3 > 100 ):
        prediksi3 = "Hujan Sangat Lebat"
    else:
        prediksi3 = "Berawan"

    return render_template('prediksi2.html', namafile1=fileName1, namafile2=fileName2, namafile3=fileName3,
                            hasil1=prediksi1, hasil2=prediksi2, hasil3=prediksi3)

if __name__ == '__main__':
    app.run(debug=True, port=2222)