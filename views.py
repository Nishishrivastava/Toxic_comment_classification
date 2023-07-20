# from django.template.loader import render_to_string
import time
from django.shortcuts import render
from django.http import HttpResponse
from django.middleware import csrf
import pickle
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt

# Create your views here.
def main_func(request):
    response = render(request, "the_project/index.html")
    return HttpResponse(response)

# Loading the TF-IDF vocabulary specific to the category
with open("the_project/toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open("the_project/severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open("the_project/obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open("the_project/insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open("the_project/threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open("the_project/identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open("the_project/toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open("the_project/severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open("the_project/obscene_model.pkl", "rb") as f:
    obs_model = pickle.load(f)

with open("the_project/insult_model.pkl", "rb") as f:
    ins_model = pickle.load(f)

with open("the_project/threat_model.pkl", "rb") as f:
    thr_model = pickle.load(f)

with open("the_project/identity_hate_model.pkl", "rb") as f:
    ide_model = pickle.load(f)

def predict(request):
    if request.method == 'POST':
        user_input = request.POST.get('text')
        data = [user_input]

        vect = tox.transform(data)
        pred_tox = tox_model.predict_proba(vect)[:, 1]

        vect = sev.transform(data)
        pred_sev = sev_model.predict_proba(vect)[:, 1]

        vect = obs.transform(data)
        pred_obs = obs_model.predict_proba(vect)[:, 1]

        vect = thr.transform(data)
        pred_thr = thr_model.predict_proba(vect)[:, 1]

        vect = ins.transform(data)
        pred_ins = ins_model.predict_proba(vect)[:, 1]

        vect = ide.transform(data)
        pred_ide = ide_model.predict_proba(vect)[:, 1]

        out_tox = round(pred_tox[0], 2)
        out_sev = round(pred_sev[0], 2)
        out_obs = round(pred_obs[0], 2)
        out_ins = round(pred_ins[0], 2)
        out_thr = round(pred_thr[0], 2)
        out_ide = round(pred_ide[0], 2)

        plt.figure(figsize=(8,6))
        y = [out_tox, out_sev, out_obs, out_ins, out_thr,out_ide]
        x = ["TOXIC", "SEVERE TOXIC", "OBSCENE", "INSULT", "THREAT", "HATRED"]
        plt.bar(x, y)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Graph')
        timestamp = str(int(time.time()))  # Get current timestamp as a string
        graph_filename = f'graph_{timestamp}.png'
        plt.savefig(f'the_project/static/the_project/{graph_filename}')

        # ##########################################
        if out_tox > 0.5 or out_ide > 0.5 or out_ins > 0.5 or out_obs > 0.5 or out_sev > 0.5 or out_thr > 0.5:
            statement = "STOP MAKING TOXIC COMMENTS!!"
        else:
            statement = "Your comment is not toxic."

        return render(request, 'the_project/index.html', {
            'data': user_input,
            'pred_tox': '{:.2f}'.format(out_tox*100),
            'pred_sev': '{:.2f}'.format(out_sev*100),
            'pred_obs': '{:.2f}'.format(out_obs*100),
            'pred_ins': '{:.2f}'.format(out_ins*100),
            'pred_thr': '{:.2f}'.format(out_thr*100),
            'pred_ide': '{:.2f}'.format(out_ide*100),
            'graph_filename': graph_filename, 
            'st' : statement,
            'sym' : "%",
            'csrf_token': csrf.get_token(request),
        })