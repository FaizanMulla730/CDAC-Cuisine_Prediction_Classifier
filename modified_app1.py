# -*- coding: utf-8 -*-
"""

"""

# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import pickle
import pandas as pd

import streamlit as st 

from PIL import Image
import pandas as pd
import xgboost as xgb
 
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV



                  
st.title("CLASSIFICATION")
pickle_in=open('C:/Users/Faizan/xgboostnew.sav','rb')
pickle_in2=open('C:/Users/Faizan/countvect.pkl','rb')


model=pickle.load(pickle_in)
cv=pickle.load(pickle_in2)



var=[st.text_input("Enter sentence","")]
doc_term_matrix=cv.transform(var)



if st.button("Predict"):
    result=model.predict(doc_term_matrix)
    st.write(str(result[0]))
    



    
