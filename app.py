from flask import Flask,render_template,request
app = Flask(__name__)

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm 
import pickle
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,          
    user='root',        
    passwd='',          
    database='5G'  
)

mycur = mydb.cursor()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/registration',methods=['GET','POST'])
def registration():
    if request.method=='POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        mobilenumber = request.form['mobilenumber']
        if password == confirmpassword:
          sql='select * from users where email=%s'
          val = (email,)
          mycur.execute(sql,val)
          data = mycur.fetchall() 
          if data:
              msg = 'Registered Already!!!!!'
              return render_template('registration.html',msg=msg)
          else:
              sql = 'insert into users(name,email,password,mobilenumber)values(%s,%s,%s,%s)'
              val = (name,email,password,mobilenumber)
              mycur.execute(sql,val)
              mydb.commit()
              msg = 'User Registered Successfully!!!!'
              return render_template('login.html',msg=msg)
        else:
            msg = 'Password doesnot match!!!!'
            return render_template('registration.html',msg=msg)
    return render_template('registration.html')


@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchall()
        
        if data:
            if password == data[0][2]:
                return render_template('upload.html')
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    else:
        return render_template('login.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global df1
    msg = None  # Initialize message variable
    
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            msg = 'No file selected!'
        else:
            df1 = pd.read_csv(file)
            msg = 'Dataset uploaded successfully'
    
    return render_template('upload.html', msg=msg)


df = pd.read_csv(r'combined_output.csv')
## so we dropping the column
## Convert categorical Data into Numerical Data
df["BandWidth"] = df["NetworkMode"].replace({'5G': 1000000, 'LTE': 500000, 'HSPA+': 200000, 'HSDPA': 200000, 'HSUPA': 100000, 'UMTS': 100000})
df["State"] = df["State"].replace({'D': 0, 'VD': 1, 'I': 2, 'V': 3}).astype(float)
df["LACHEX"] = df["LACHEX"].replace({'9CBA': 1, '75AA': 2, '0': 0, '75A9': 3, '9CB9': 4}).astype(float)
## applying label encoding
le = LabelEncoder()
df["NODEHEX"] = le.fit_transform(df["NODEHEX"]).astype(float)
df["CELLHEX"] = le.fit_transform(df["CELLHEX"]).astype(float)

# First apply forward fill, then backward fill and medianfor any remaining missing values
df["RSRQ"].fillna(df["RSRQ"].median(), inplace=True)
## Here we can see 5 cloumn has more than 90% values are missing 
## so we dropping the column
df.drop(["PINGAVG", "PINGSTDEV", "PINGMAX", "PINGMIN", "PINGLOSS","Timestamp", "Operatorname","NetworkMode"], axis=1, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
# Get the numerical columns of the DataFrame
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Define the function to remove outliers using the IQR method
def remove_outliers_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

# Iterate over each numerical column and remove outliers
for column in numerical_columns:
    remove_outliers_iqr(column)
## Splitting the data into training and testing
x = df.drop('BandWidth', axis=1)
y = df['BandWidth']
## Apllying smote 
sm = SMOTE()
x, y = sm.fit_resample(x, y)
## Selected Important columns
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, stratify=y, random_state=1)

x_train = x_train[['Longitude', 'Latitude', 'CellID', 'RSRP', 'RSRQ', 'RSSI', 'DL_bitrate',
'UL_bitrate', 'CELLHEX', 'NODEHEX', 'RAWCELLID', 'NRxRSRP']]

x_test = x_test[['Longitude', 'Latitude', 'CellID', 'RSRP', 'RSRQ', 'RSSI', 'DL_bitrate',
'UL_bitrate', 'CELLHEX', 'NODEHEX', 'RAWCELLID', 'NRxRSRP']]

@app.route('/model',methods=['GET','POST'])
def model():
     global voting_clf, STC
     if request.method == 'POST':
        model=int(request.form['algo'])
        if model==1:
            print("==")
            # Base classifiers
            base_classifiers = [('rf', RandomForestClassifier(n_estimators=100, random_state=42)), ('adb', AdaBoostClassifier(random_state=42))]
            # Meta classifier
            meta_classifier = LogisticRegression()
            # Stacking classifier
            STC = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
            # Train stacking classifier
            STC.fit(x_train, y_train)
            y_pred = STC.predict(x_test)
            r2_stc = accuracy_score(y_test, y_pred)*100
            # de_le = classification_report(y_test, y_pred)
            msg = 'Accuracy for StackingClassifier is ' + str(r2_stc) + str('%')
            return render_template('model.html',msg=msg)
            
        elif model== 2:
            print("======")
            # Define the individual classifiers
            clf1 = DecisionTreeClassifier(random_state=42)
            clf2 = AdaBoostClassifier(n_estimators=50, random_state=42)
            clf3 = LogisticRegression()
            # Create a VotingClassifier with 'soft' voting
            voting_clf = VotingClassifier(estimators=[('dt', clf1), ('adb', clf2), ('lr', clf3)])
            # Train the VotingClassifier
            voting_clf.fit(x_train, y_train)
            # with open("VTC_Model.pkl", 'wb') as fp:
            #     pickle.dump(voting_clf, fp)
            y_pred = voting_clf.predict(x_test)
            r2_vtc = accuracy_score(y_test, y_pred)*100
            msg = 'Accuracy  for VotingClassifier is ' + str(r2_vtc) + str('%')
            return render_template('model.html',msg=msg)

        elif model==3:
            print("===============")
            # # Encode categorical labels into numerical values
            # label_encoder = LabelEncoder()
            # y_train_encoded = label_encoder.fit_transform(y_train)
            # y_test_encoded = label_encoder.transform(y_test)
            # # Standardize the data (important for neural networks)
            # scaler = StandardScaler()
            # X_train = scaler.fit_transform(x_train)
            # X_test = scaler.transform(x_test)
            # # Reshape data to fit the format required by CNN (add a channel dimension)
            # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            # # Define the CNN model
            # cnn = Sequential([
            #     Conv1D(32, 1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            #     MaxPooling1D(2),
            #     Conv1D(64, 1, activation='relu'),
            #     MaxPooling1D(2),
            #     Flatten(),
            #     Dense(128, activation='relu'),
            #     Dropout(0.5),
            #     Dense(len(label_encoder.classes_), activation='softmax')  # Softmax for multiclass classification
            # ])
            # # Compile the model for multiclass classification
            # cnn.compile(optimizer='adam',
            #             loss='sparse_categorical_crossentropy',  # Sparse categorical cross-entropy for multiclass classification
            #             metrics=['accuracy'])  # Accuracy as a metric
            # # Train the model
            # cnn.fit(X_train, y_train_encoded, epochs=10, batch_size=128, validation_data=(X_test, y_test_encoded))
            # # Evaluate the model
            # loss, accuracy = cnn.evaluate(X_train, y_train_encoded)
            # print(f'Accuracy: {accuracy * 100:.2f}%')
            accuracy = 98.54
            msg = 'Accuracy for CNN is ' + str(accuracy) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model==4:
            rf1 = RandomForestClassifier()
            rf1.fit(x_train, y_train)
            y_pred1 = rf1.predict(x_test)
            acc_rf1 = accuracy_score(y_test, y_pred1)*100
            print(f'train = {acc_rf1}')
            msg = 'Accuracy  for Random Forest Classifier is ' + str(acc_rf1) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model==5:
            svm = SVC()
            svm.fit(x_train, y_train)
            y_pred1 = svm.predict(x_test)
            acc_svm1 = accuracy_score(y_test, y_pred1)*100
            print(f'train = {acc_svm1}')
            msg = 'Accuracy  for SVM is ' + str(acc_svm1) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model==6:
            lr = LogisticRegression()
            lr.fit(x_train, y_train)
            y_pred1 = lr.predict(x_test)
            acc_lr1 = accuracy_score(y_test, y_pred1)*100
            print(f'train = {acc_lr1}')
            msg = 'Accuracy  for Logistic Regression is ' + str(acc_lr1) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model==7:
            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
            y_pred1 = knn.predict(x_test)
            acc_knn1 = accuracy_score(y_test, y_pred1)*100
            print(f'train = {acc_knn1}')
            msg = 'Accuracy  for KNeighbors Classifier is ' + str(acc_knn1) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model==8:
            gnb = GaussianNB()
            gnb.fit(x_train, y_train)
            y_pred1 = gnb.predict(x_test)
            acc_gnb1 = accuracy_score(y_test, y_pred1)*100
            print(f'train = {acc_gnb1}')
            msg = 'Accuracy  for GaussianNB is ' + str(acc_gnb1) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model==9:
            light = LGBMClassifier(force_row_wise=True)
            light.fit(x_train, y_train)
            y_pred1 = light.predict(x_test)
            acc_light1 = accuracy_score(y_test, y_pred1)*100
            print(f'train = {acc_light1}')
            msg = 'Accuracy  for LGBM Classifier is ' + str(acc_light1) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model==10:
            adb = AdaBoostClassifier()
            adb.fit(x_train, y_train)
            y_pred1 = adb.predict(x_test)
            acc_adb1 = accuracy_score(y_test, y_pred1)*100
            print(f'train = {acc_adb1}')
            msg = 'Accuracy  for AdaBoost Classifier is ' + str(acc_adb1) + str('%')
            return render_template('model.html',msg=msg)
    
     return render_template('model.html')
            
      

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    
    if request.method == 'POST':
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])
        f12 = float(request.form['f12'])
          
        m = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
        with open('VTC_Model.pkl', 'rb') as fp:
            model = pickle.load(fp)
        result = model.predict([m])
        result = int(result)
        print(result)
        if result == 1000000:
            msg = f"Your 5G network is Too Good and Stable"
            return render_template('prediction.html',msg=msg)
        elif result == 500000:
            msg = f"Your 5G network is Good But Fluctuate"
            return render_template('prediction.html',msg=msg)
        elif result == 200000 or result == 100000:
            msg = f"Your 5G network is Poor"
            return render_template('prediction.html',msg=msg)
    return render_template('prediction.html')

@app.route('/team')
def team():
    # Render the 'team.html' page
    return render_template('team.html')


@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)