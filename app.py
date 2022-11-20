#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


app = Flask(__name__)
@app.route('/')
def home():
      return render_template('index.html')


# In[ ]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output =prediction[0]

    return render_template('index.html', prediction_text='The Flower is {}'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




