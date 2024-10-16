import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)

##import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/reg.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route('/',methods=['GET','POST'])
def predict():
     if request.method=='POST':
          Age=float(request.form.get('Age'))
          sex=float(request.form.get('sex'))
          cp=float(request.form.get('cp'))
          tresttps=float(request.form.get('trestbps'))
          chol=float(request.form.get('chol'))
          fbs=float(request.form.get('fbs'))
          restecg=float(request.form.get('restecg'))
          thalach=float(request.form.get('thalach'))
          exang=float(request.form.get('exang'))
          oldpeak=float(request.form.get('oldpeak'))
          slope=float(request.form.get('slope'))
          ca=float(request.form.get('ca'))
          thal=float(request.form.get('thal'))


          new_data_scaled=standard_scaler.transform([[Age,sex,cp,tresttps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
          result=ridge_model.predict(new_data_scaled)

          return render_template('index.html',results=result[0])
     else:
          return render_template('index.html')


if __name__=="__main__":
     app.run()