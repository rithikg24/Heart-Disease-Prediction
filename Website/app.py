from flask import Flask, redirect,render_template,request,redirect
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
app = Flask(__name__)


model=pickle.load(open('Website/lr_model_final.pkl','rb'))

dataset=pd.read_csv("Website/heart_2020_cleaned.csv")
features_dataset=pd.get_dummies(dataset.drop('HeartDisease',axis=1))
scaler=StandardScaler()
scaled_features_dataset=scaler.fit_transform(features_dataset.values)

#@app.route("/")
#def start():
    #return render_template('index.html')

@app.route('/',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        case=[]
        bmi=request.form['bmi']
        case.append(float(bmi))

        physical_health=request.form['physical_health']
        case.append(float(physical_health))

        mental_health=request.form['mental_health']
        case.append(float(mental_health))

        sleeptime=request.form['sleeptime']
        case.append(float(sleeptime))

        smoking_list=[0,0]
        smoking=request.form['smoking']
        if smoking =='No':
            smoking_list[0]=1
        elif smoking =='Yes':
            smoking_list[1]=1
        
        for item in smoking_list:
            case.append(item)

        alcohol_drinking_list=[0,0]
        alcohol_drinking=request.form['alcohol_drinking']
        if alcohol_drinking =='No':
            alcohol_drinking_list[0]=1
        elif alcohol_drinking =='Yes':
            alcohol_drinking_list[1]=1
        
        for item in alcohol_drinking_list:
            case.append(item)

        stroke_list=[0,0]
        stroke=request.form['stroke']
        if stroke =='No':
            stroke_list[0]=1
        elif stroke =='Yes':
            stroke_list[1]=1
        
        for item in stroke_list:
            case.append(item)

        diffwalking_list=[0,0]
        diffwalking=request.form['diffwalking']
        if diffwalking =='No':
            diffwalking_list[0]=1
        elif diffwalking =='Yes':
            diffwalking_list[1]=1
        
        for item in diffwalking_list:
            case.append(item)

        sex_list=[0,0]
        sex=request.form['sex']
        if sex =='Female':
            sex_list[0]=1
        elif sex =='Male':
            sex_list[1]=1
        
        for item in sex_list:
            case.append(item)

        agecat_list=[0,0,0,0,0,0,0,0,0,0,0,0,0]
        agecat=request.form['age_category']
        agecat_list[int(agecat)]=1

        for item in agecat_list:
            case.append(item)

        race_list=[0,0,0,0,0,0]
        race=request.form['race']
        if race=='White':
            race_list[5]=1
        elif race=='Black':
            race_list[2]=1
        elif race=='Hispanic':
            race_list[3]=1
        elif race=='Other':
            race_list[4]=1
        elif race=='Asian':
            race_list[1]=1
        elif race=='American_Indian':
            race_list[0]=1
        
        for item in race_list:
            case.append(item)

        diabetic_list=[0,0,0,0]
        diabetic=request.form['diabetic']
        if diabetic=='yes':
            diabetic_list[2]=1
        elif diabetic=='no':
            diabetic_list[0]=1
        elif diabetic=='no_boderline':
            diabetic_list[1]=1
        elif diabetic=='yes_pregnancy':
            diabetic_list[3]=1
        
        for item in diabetic_list:
            case.append(item)
        
        physical_activity_list=[0,0]
        physical_activity=request.form['physical_activity']
        if physical_activity =='No':
            physical_activity_list[0]=1
        elif physical_activity =='Yes':
            physical_activity_list[1]=1
        
        for item in physical_activity_list:
            case.append(item)
        
        gen_health_list=[0,0,0,0,0]
        gen_health=request.form['gen_health']
        if gen_health=='excellent':
            gen_health_list[0]=1
        elif gen_health=='very_good':
            gen_health_list[4]=1
        elif gen_health=='good':
            gen_health_list[2]=1
        elif gen_health=='fair':
            gen_health_list[1]=1
        elif gen_health=='poor':
            gen_health_list[3]=1
        
        for item in gen_health_list:
            case.append(item)

        asthma_list=[0,0]
        asthma=request.form['asthma']
        if asthma =='No':
            asthma_list[0]=1
        elif asthma =='Yes':
            asthma_list[1]=1
        
        for item in asthma_list:
            case.append(item)
        
        kidney_disease_list=[0,0]
        kidney_disease=request.form['kidney_disease']
        if kidney_disease =='No':
            kidney_disease_list[0]=1
        elif kidney_disease =='Yes':
            kidney_disease_list[1]=1
        
        for item in kidney_disease_list:
            case.append(item)
        
        skin_cancer_list=[0,0]
        skin_cancer=request.form['skin_cancer']
        if skin_cancer =='No':
            skin_cancer_list[0]=1
        elif skin_cancer =='Yes':
            skin_cancer_list[1]=1
        
        for item in skin_cancer_list:
            case.append(item)
        
        scaled_case=scaler.transform([case])
        case_pred=model.predict(scaled_case)
        prediction=case_pred[0]
        return render_template("index.html",prediction=prediction)
    else:
        return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True)