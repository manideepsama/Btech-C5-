from flask import Flask,render_template,request
from sklearn.linear_model import SGDClassifier
import pickle
import re
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
def preprocess_tweet(text):
    text = re.sub('[\W]+', ' ', text.lower())
    return text
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('[\W]+', ' ', text.lower())
    tokenized = [w for w in tokenizer_porter(text) if w not in stop]
    return tokenized
from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, 
                         preprocessor=None,tokenizer=tokenizer)
app = Flask(__name__)
#@app.route('/2')
#def call():
 #   return render_template('second.html')
@app.route('/predict',methods=['GET', 'POST'])
def predict():
	#if request.method =='POST':
	text=request.form['data']
	text=preprocess_tweet(text)
		#text="i am happy today"
	clf=pickle.load(open('temp.pkl','rb'))
	#label = {0:'negative', 1:'positive'}
	text=[text]
	X = vect.transform(text)
	text=clf.predict(X)
	#threshold = 0.4
	
	prob=clf.predict_proba(X)[:,0]*100

	if(text==[0]):
		text="positive statement no need to worry "+np.array_str(prob/100)+" probability to belong to 0 class"
		
	else:
		text="dangerous situation try to help him "+np.array_str(clf.predict_proba(X)[:,1])+" probability to belong to 1 class"
	return render_template('model.html',prediction=text)
               #print('Prediction: %s\nProbability: %.2f%%'%(label[clf.predict(X[0]],np.max(clf.predict_proba(X))*100))
@app.route('/data',methods=['GET', 'POST'])
def weather_dashboard():
    filename = '/home/manideep/Downloads/Temp final proj/data.csv'
    num=request.form['num']
    num1=request.form['num1']
    data = pd.read_csv(filename, header=int(num1),nrows=int(num))
    myData = list(data.values)
    return render_template('data.html', myData=myData)
@app.route('/')
def hello_world():
    return render_template('model.html')
@app.route('/projecthtml',methods=['GET', 'POST'])
def h1():
	return render_template('project.html')
@app.route('/modelhtml',methods=['GET', 'POST'])
def h3():
	return render_template('model.html')
@app.route('/abouthtml',methods=['GET', 'POST'])
def h2():
	return render_template('about.html')
if __name__=='__main__':
	app.run(host='192.168.43.199')

