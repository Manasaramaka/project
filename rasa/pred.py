import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import re
# import nltk
# nltk.download("punkt")
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model


import json


# initializing the required processes for backend machine learning
print("initializing the required processes for backend machine learning.......")
data = pd.read_csv("train.txt",delimiter=";",header=None)
data.columns = ["documents","label"]

le = LabelEncoder()
le.fit(data.label)
print("it may take a while to initialize...")
lemm = WordNetLemmatizer()
def cleaning_lemm(sent):
    sent = sent.lower()
    sent = re.sub(r"([^a-z ])","",sent)   
    word_list = word_tokenize(sent)  
    lem_words = list(map(lemm.lemmatize,word_list))   
    cln_word = list(filter(lambda x: x not in stopwords.words("english"),lem_words))   
    return " ".join(cln_word)
    
cleaned_sent = []
for i in data.documents:
    temp = cleaning_lemm(i)
    cleaned_sent.append(temp)
cleaned_sent = np.array(cleaned_sent)
data["lemmatized_docs"] = cleaned_sent
data["length_le"] = np.array(map(len,data["lemmatized_docs"]))


NB_WORDS = 13444
MAX_LEN = 229

tk = Tokenizer(num_words=NB_WORDS,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True)
tk.fit_on_texts(data.lemmatized_docs)

lstm = load_model("LSTM_model.hdf5")

count = CountVectorizer()
count.fit(data["lemmatized_docs"])
vocabulary = count.get_feature_names()
print("model ready......!")
def reply(emotion):
    if emotion=="sadness":
        return """i can understand that you feel sad and depressed......
so to get refreshed let's go for a walk and eat something nice......
if that is not possible here is something visit this to make your mood better 
'<a href="https://i.pinimg.com/originals/b6/3c/b2/b63cb2f46125953046854667f4a461e4.jpg">click here</a>'
"""
    if emotion=="anger":
        return """i can understand that you are angry.....
it is not good to suppress it so let's do some exercise and release all the anger....
but if you want a quick relief
<a href="https://www.reddit.com/r/funny/">click here</a>"""
    if emotion=="fear":
        return """i can understand that you are feeling fearfull....
now sit in a comfortable place and imagine your happy place....
after that if you have cooled down....try to contact anyone who can help"
this may help a little
<a href="https://www.reddit.com/r/Courage/">click here</a>"""

    if emotion=="joy" or emotion=="surprise":
        return """what...you are already in a good mood..
  this is just a little souvenir
<a href="https://grammartop.com/wp-content/uploads/2020/11/joyful-4052744a43a2cbd414f3be155732b42eda376535.png">click here</a>"""    

    if emotion=="love" :
        return """i can sense the love radiating from you. we will keep that up"""



def predict():
    with open("data.json","r") as f:
        com_data = json.load(f)["text"]
        if "yes" in str(com_data[-2]).lower() and "not" in str(com_data[-2]).lower(): 
            return """i can understand that......
"""
        else:
            text_inp = str(json.load(f)["text"][2:4])
            cl_text = cleaning_lemm(text_inp)
            cl_text = " ".join(list(filter(lambda x: x in vocabulary,word_tokenize(cl_text))))
            seq_text = tk.texts_to_sequences([cl_text])
            processed_input = pad_sequences(seq_text, maxlen=MAX_LEN)
            output = lstm.predict(processed_input)
            emotion = le.inverse_transform([np.argmax(output)])[0]
            with open("result.json","w") as f1:
                json.dump({"result":emotion},f1)
            return reply(emotion)
            

if __name__ == "__main__":
    print(predict())