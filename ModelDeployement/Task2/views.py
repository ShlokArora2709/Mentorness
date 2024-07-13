from django.shortcuts import render
import pandas as pd
import joblib
import datetime

pipeline=joblib.load("Models\pipeline.pkl")
model=joblib.load("Models\salary_prediction_model.pkl")
scaler=joblib.load("Models\salary_scaler.pkl")
def home(request):
    return render(request, 'home.html')

def form_view(request):
    context = {}
    if request.method == 'POST':
        age = int(request.POST['age'])
        joining_date = request.POST['joining_date']
        past_experience = int(request.POST['past_experience'])
        designation = request.POST['designation']
        
        jd=datetime.datetime.strptime(joining_date, '%Y-%m-%d')
        today=datetime.datetime.now()
        if jd>today:
            curr_exp=0
            days=0
        else:
            curr_exp = today.year - jd.year
            if today.month < jd.month or (today.month == jd.month and today.day < jd.day):
                curr_exp -= 1
            days = (today - jd).days

        total_exp=curr_exp+past_experience
        model_input=pd.DataFrame({'AGE':[age],'DAYS WITH COMPANY':[days],'TOTAL EXPERIENCE':[total_exp],'Current Experience (Years)':[curr_exp],'PAST EXP':[past_experience],'DESIGNATION':[designation]})
        model_input=pipeline.transform(model_input)
        res=model.predict(model_input)
        res=scaler.inverse_transform(res)
        context={'Salary':round(res[0][0],2)}

    return render(request, 'input.html', context)
