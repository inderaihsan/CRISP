from distutils.command.upload import upload
import streamlit as st 
import pandas as pd
import numpy as np 
import seaborn as sns
from io import StringIO 
import matplotlib.pyplot as plt
import plotly.figure_factory as ff  
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report 
import gdown



#global var 

downloaded = False 
uploaded = False
sns.set_style("whitegrid")
uploaded = False 
dataframe = pd.DataFrame() 
def fileupload() : 
    uploaded_file = st.file_uploader("Choose a file") 
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        global dataframe
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)  
        global uploaded 
        uploaded = True


def convert(uploaded) :
    if (uploaded) : 
        global dataframe
        columns = dataframe.columns 
        #selected_columns = st.multiselect('Please pick a columns', columns)  
        X = st.selectbox('Select an X variable', dataframe.select_dtypes(include='number').columns)
        y = st.selectbox('Select a y variable', dataframe.select_dtypes(include='number').columns) 
        hue = st.selectbox('Select a hue', dataframe.select_dtypes(exclude='number').columns)  
        plot_kind = st.radio( "what kind of plot do you want to use?",
        ('Scatter', 'Violin'))

        if (plot_kind == 'Scatter'):
            fig, ax = plt.subplots()
            plt.title(plot_kind + ' of '+ X + ' & '+ y)
            ax = sns.scatterplot(x = dataframe[X], y = dataframe[y], data = dataframe, hue = hue)  
            st.plotly_chart(fig)

        elif (plot_kind == 'Violin') : 
            fig, ax = plt.subplots() 
            plt.title(plot_kind + ' of '+ X + ' & '+ y) 
            ax = sns.violinplot(x = dataframe[X], y = dataframe[y], data = dataframe, hue= hue)
            st.pyplot(fig)


def eda (dataframe) : 
    global uploaded
    if (uploaded) :   
        st.subheader('You can find a brief overview of your data below by looking at the profile page of your data.') 
        options = st.multiselect('please select the column you want to analyze from your data : ',np.sort(dataframe.columns))
        #st.write('selected columns : ', (x for x in options)) 
        but = st.button('Analyze')  
        if (but and len(options)!=0): 
            pr = dataframe[options].profile_report()
            st_profile_report(pr) 



def get_id(url) :  
    res = url.split('/')
    for el in res : 
      if len(el)> 10 and el !='drive.google.com' and el!='https:' and el!='docs.google.com' and el!= 'spreadsheets': 
        #print(el) 
        return el 


    

################################################algo#######################################################

st.title('Welcome!') 
st.write('this web app is a beta version of a completely GUI based exploratory data analysis.') 
st.write('this app will guide you step by step through data analysis based on  [CRISP](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) method. so the first step is to upload your file. ') 
st.write('you can upload your file below using the form')
fileupload()  

eda(dataframe) 








 
