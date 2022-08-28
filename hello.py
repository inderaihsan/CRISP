from distutils.command.upload import upload
from multiprocessing import dummy
from turtle import onclick 
import streamlit.components.v1 as components
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
from dataprep.eda.missing import plot_missing 
from dataprep.eda import create_report



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
        if 'dataset' not in st.session_state:
                st.session_state['dataset'] = dataframe  
        return dataframe

def use_dummy_data() : 
    global uploaded 
    global dataframe
    data = sns.load_dataset('penguins')  
    dataframe = data 
    uploaded = True 
    if 'dataset' not in st.session_state:
            st.session_state['dataset'] = dataframe  
    st.write(dataframe) 
    return dataframe
    


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

 
def showmissing(dataframe) :   
    if(uploaded) : 
        missing_val = []
        list_of_col = dataframe.columns.unique() 
        for items in list_of_col : 
            missing_val.append(dataframe[items].isna().sum()) 
        
        missing_tab = pd.DataFrame({'kolom' : list_of_col, 'number of missing' : missing_val}) 
        st.bar_chart(data = missing_tab, x = 'kolom', y = 'number of missing') 

    else : 
        st.write("you haven't uploaded a file, please upload a CSV file or use seaborn dummy data") 

def fill_missing(dataframe) :  
    sel_col = st.radio('please select a column to be filled', dataframe.columns) 
    st.write(sel_col) 

    #categorical missing value
    if(type(st.session_state['dataset'][sel_col][0])==str) : 
        sel_val = st.radio('select a value to replace the null : ', st.session_state['dataset'][st.session_state['dataset'][sel_col].notnull()][sel_col].unique()) 
        if(st.button('fill null values')) : 
            st.session_state['dataset'][sel_col] = st.session_state['dataset'][sel_col].apply(lambda x:sel_val if pd.isnull(x) else x) 
            #showmissing(st.session_state['dataset'])  
            st.session_state['dataset'] = st.session_state['dataset']  

    #numerical missing value
    if(type(st.session_state['dataset'][sel_col][0])!=str) : 
        sel_method = st.radio('numerical method : ', ['Mean', 'Minimum', 'Maximum'], horizontal = True)  
        sel_group = st.radio('select a group column' , st.session_state.dataset.select_dtypes(exclude='number').columns, horizontal=True) 
        operator = {'Mean' : st.session_state.dataset.groupby(by=[sel_group]).mean().reset_index() , 
                    'Minimum' : st.session_state.dataset.groupby(by=[sel_group]).min().reset_index(), 
                    'Maximum' : st.session_state.dataset.groupby(by=[sel_group]).max().reset_index(), 
                   } 
        temp = operator[sel_method] 
        gather = dict(zip(temp[sel_group], temp[sel_col]))
        st.write(gather)
        if(st.button('fill null values')) :  
            index = 0
            for x,y in zip(st.session_state.dataset[sel_group], st.session_state.dataset[sel_col]) : 
                if (pd.isnull(y)) : 
                    st.session_state.dataset[sel_col][index] = gather[x] 
                    print('miss miss miss') 
                    index=index+1 
                index=index+1
            st.session_state.dataset = st.session_state.dataset 
            
            
    
################################################algo#######################################################



st.title('Welcome!') 
st.write('this web app is a beta version of a completely GUI based exploratory data analysis.') 
st.write('this app will guide you step by step through data analysis based on  [CRISP](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) method. so the first step is to upload your file. ') 
st.write('you can upload your file below using the form')

fileupload()   

st.write('you can also use a dummy data from seaborn [penguins](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv) dataset ')

dummyyy = st.checkbox('i want to use seaborn dataset') 

if(dummyyy) : 
    use_dummy_data() 
    df = st.session_state['dataset']
eda(dataframe)    

st.header('Missing value') 

if(uploaded) :  
    tab_1, tab_2 = st.tabs(["Missing value", "Fill Missing Value"]) 
    with tab_1 : 
        df = st.session_state['dataset']
        showmissing(df) 
    with tab_2 : 
        st.session_state['dataset']
        fill_missing(df) 
        showmissing(df)





 
