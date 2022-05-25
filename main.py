from unittest import result
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import json
import requests
import os
import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.secret_key = 'encryption_key'
app.config['UPLOAD_FOLDER'] = str(os.getcwd())+'/static/uploads'

ALLOWED_EXTENSIONS = set(["txt"])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/ai', methods=['get', 'post'])
def ai():
    return render_template('chatScreen.html')

@app.route('/machineLearning/<msg>', methods=['get', 'post'])
def machine_learning(msg):
    print(msg)
    with open("rasa/data.json","r") as f:
        sta_data = json.load(f)
        sta_data["text"].append(msg)
        print(sta_data)
    
    with open("rasa/data.json","w") as f1:
        sta_data = json.dump(sta_data,f1)
    
    data1 = json.dumps({"sender": "Rasa","message": msg})
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    res1 = requests.post('http://localhost:5005/webhooks/rest/webhook', data= data1, headers = headers)
    res1 = res1.json()
    print(res1)
    try:
        if len(res1)>=2:
            s=res1[0]['text']+"\n\n\n"+res1[1]['text']
            res1=s
        if len(res1)<2:
            res1=res1[0]['text']
    except:
        res1="please can you repeat the last response"
    return jsonify(data=res1)

result = ""

@app.route('/social', methods=['get', 'post'])
def social():
    global result
    if request.method=="POST":
        print(request.files)
        file1 = request.files['file_id1']
        file2 = request.files['file_id2']

        if file1.filename == '' and file2.filename == '':
            return redirect(request.url)

        
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename) :
                print("in condition")
                filename1 = secure_filename(file1.filename)
                file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
                filename2 = secure_filename(file2.filename)
                file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

                with open(os.path.join(app.config['UPLOAD_FOLDER'], filename1),"r") as f:
                    facebook = f.read()
                    # facebook_scores = sid.polarity_scores(facebook)
                    # facebook_scores.pop("compound")
                    # facebook_result,facebook_scores = sorted(facebook_scores.items(),key=lambda x : x[1],reverse=True)
                with open(os.path.join(app.config['UPLOAD_FOLDER'], filename2),"r") as f1:
                    twitter = f1.read()
                social_scores = sid.polarity_scores(twitter+facebook)
                social_scores.pop("compound")
                social_result,social_score = sorted(social_scores.items(),key=lambda x : x[1],reverse=True)[0]
                with open("rasa/data.json","r") as f0:
                    user_data = json.load(f0)
                    zchi = user_data["text"][-2]
                    # zchi.pop("compound")
                    # zchi_result,zchi_score = sorted(zchi.items(),key=lambda x : x[1],reverse=True)[0]

                with open("rasa/result.json","r") as f2:
                    chatbot_data = json.load(f2)
                    print(chatbot_data)
                    if "yes" in zchi.lower() or "not" in zchi.lower():
                        result = """</br>
    <strong>"THERE IS HOPE, EVEN WHEN YOUR BRAIN TELLS YOU THERE ISN’T"</strong>
</br>
After the analysis we conclude that you possibly could be suffering with Schizophrenia. 
</br>
Here is some tips for you : <a href="https://www.nimh.nih.gov/health/topics/schizophreni">click here</a>!
</br>
It is highly suggested to visit a psychiatrist at earliest to avoid the worst.
</br>
Don’t hesitate to reach out for help: <a href="https://www.thelivelovelaughfoundation.org/helpline">click here</a>!
</br>
check the Doctor profile here: <a href="/profile">click here</a>."""
                    elif chatbot_data["result"] == "sadness":
                        if social_result == "neg":
                            result = """</br>
<strong>‘Help is available’</strong>
</br>
After the interaction we conclude that you possibly could be depressed. Here are some tips for you: <a href="https://www.nimh.nih.gov/health/publications/depression#part_6149">click here</a>!.
</br>
It is highly suggested to visit a psychiatrist at earliest to avoid the worst.
</br>
Don’t hesitate to reach out for help: <a href="https://www.thelivelovelaughfoundation.org/helpline" >click here</a>!
</br>
check the Doctor profile here: <a href="/profile">click here</a>.
"""
                        else:
                            result = """</br>
<strong>I can understand that you feel stressed....</strong>
</br>
so to get refreshed let's go for a walk and eat something nice......
</br>
if that is not possible here is something visit this to make your mood better '<a href="https://www.thelivelovelaughfoundation.org/helpline">click here</a>!'
</br>
check the Doctor profile here: <a href="/profile">click here</a>."""

                    elif chatbot_data["result"] == "anger": 
                        if social_result == "neg":
                            result = """</br>
<strong>“Change what you can, manage what you can’t.”</strong>
</br>
After the analysis we conclude that you possibly could be suffering with Anger Management Issues . Here is some tips for you : <a href="https://www.nimh.nih.gov/health/publications/disruptive-mood-dysregulation-disorder">click here</a>!
</br>
It is highly suggested to visit a psychiatrist at earliest to avoid the worst.
</br>
Don’t hesitate to reach out for help: <a href="https://www.thelivelovelaughfoundation.org/helpline"> click here</a>!
</br>
check the Doctor profile here: <a href="/profile">click here</a>.
"""
                        else:
                            result = """
</br>
<strong>One of the best ways to reduce stress is to accept the things that you cannot control</strong>
</br>
After the analysis we conclude that you  could possibly be stressed.
</br>
Here is some tips for you to keep up <a href="https://www.nimh.nih.gov/sites/default/files/documents/health/publications/so-stressed-out-fact-sheet/20-mh-8125-imsostressedout.pdf">click here</a>!
</br>
It is highly suggested to visit a psychiatrist at earliest to avoid the worst.
</br>
Don’t hesitate to reach out for help: <a href="https://www.thelivelovelaughfoundation.org/helpline">click here</a>!
</br>
check the Doctor profile here: <a href="/profile">click here</a>.
"""
                    elif chatbot_data["result"] == "fear":
                        if social_result == "neg":
                            result = """</br>
<strong>‘It’s OK to not feel OK’</strong>
</br>
After the analysis we conclude that you possibly could be suffering with Panic disorders . Here is some tips for you : <a href="https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/feelings-and-symptoms/anxiety-fear-panic/"> click here</a>!
</br>
It is highly suggested to visit a psychiatrist at earliest to avoid the worst.
</br>
Don’t hesitate to reach out for help:<a href="https://www.thelivelovelaughfoundation.org/helpline">click here</a>! 
</br>
check the Doctor profile here: <a href="/profile">click here</a>.
"""
                        else:
                            result = """</br>
<strong>Anxiety does not empty tomorrow of its sorrows, but only empties today of its strength</strong> 
</br>
After the analysis we conclude that you  could possibly be having Anxiety issues.
</br>
Here is some tips for you to keep up <a href="https://www.nimh.nih.gov/news/media/2021/great-helpful-practices-to-manage-stress-and-anxiety">click here</a>!
</br>
It is highly suggested to visit a psychiatrist at earliest to avoid the worst.
</br>
Don’t hesitate to reach out for help: <a href="https://www.thelivelovelaughfoundation.org/helpline">click here</a>!
</br>
check the Doctor profile here: <a href="/profile">click here</a>.

"""
                    elif chatbot_data["result"] == "joy" or chatbot_data["result"] == "love" or chatbot_data["result"] == "surprise":
                        if social_result == "neg":
                            result = """
</br>
<strong>!!Sending some good vibes and happy thoughts your way!!</strong>
</br>
After the analysis we conclude that you  are mentally healthy.
</br>
Here is some tips for you to keep up : <a href="https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health"> click here </a>!
</br>
Here’s something that might help you: <a href="https://www.thelivelovelaughfoundation.org/helpline">click here </a>!</body>
</br>
"""
                        else:
                            result = """S</br>
<strong>self-care is how you take your power back</strong>
</br>
After the analysis we conclude that you  are mentally healthy.
</br>
Here is some tips for you to keep up : <a href=https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health> click here </a>!
</br>
Here’s something that might help people with mental health issues: <a href=https://www.thelivelovelaughfoundation.org/helpline>click here </a>!
</br>
check the Doctor profile here: <a href="/profile">click here</a>.
"""
                    print(result)
                    with open("fin_result.txt","w") as f5:
                        f5.write(result)
                    # return render_template("data_page.html", result = result)
                    return redirect(url_for("display_image",result=result))


    else:
        return render_template('social.html')


@app.route('/data_page/')
def display_image():
    with open("fin_result.txt","r") as f5:
        result = f5.read()
    with open("rasa/data.json","w") as f1:
        json.dump({"text":[]},f1)
	#print('display_image filename: ' + filename)
    return render_template("data_page.html", result = result)

@app.route('/profile')
def display_profile():
    return render_template("profile.html")


if __name__ == '__main__':
    app.run(port=8008,debug=True,threaded=True)

