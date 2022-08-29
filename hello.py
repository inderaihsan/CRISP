from distutils import errors
from distutils.command.upload import upload
from multiprocessing import dummy
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
st.session_state.selected = False
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

def pick_columns(df) : 
    selected_columns = st.multiselect('please select the column you want to analyze from your data : ',np.sort(df.columns), key='select_column')
    df = df[selected_columns]  
    if (st.button('confirm')) :   
        global selected  
        selected = True
        st.session_state.dataset = df 

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

def fill_missing(data) :   
    st.write(data)
    sel_col = st.radio('please select a column to be filled', data.columns) 
    st.write(sel_col) 

    #categorical missing value
    if(type(data[sel_col][0])==str) : 
        sel_val = st.radio('select a value to replace the null : ', data[data[sel_col].notna()][sel_col].unique()) 
        if(st.button('fill null values')) : 
            data[sel_col] = data[sel_col].apply(lambda x:sel_val if pd.isnull(x) else x) 
            #showmissing(st.session_state['dataset'])  
            st.session_state['dataset'] = data  

    #numerical missing value
    if(type(data[sel_col][0])!=str) : 
        sel_method = st.radio('numerical method : ', ['Mean', 'Minimum', 'Maximum'], horizontal = True)  
        sel_group = st.radio('select a group column' , data.select_dtypes(exclude='number').columns, horizontal=True)  
        if(len(sel_group)>0) : 
            operator = {'Mean' : data.groupby(by=[sel_group]).mean().reset_index() , 
                        'Minimum' : data.groupby(by=[sel_group]).min().reset_index(), 
                        'Maximum' : data.groupby(by=[sel_group]).max().reset_index(), 
                    }   
            temp = operator[sel_method] 
            gather = dict(zip(temp[sel_group], temp[sel_col]))
            st.write(gather)
            index = 0 
            if(st.button('fill null values')) :  
                for x,y in zip(data[sel_group], data[sel_col]) : 
                    if (pd.isnull(y)) : 
                        value_to_replace = operator[sel_method] 
                        #st.write(value_to_replace)
                        #value_to_replace = value_to_replace[[value_to_replace][sel_group]==x]
                        value_to_replace = value_to_replace[value_to_replace[sel_group]==x][sel_col].values[0]
                        df[sel_col][index] = value_to_replace  
                    index = index+1

    
        elif(len(sel_group)==0) :  
             operator = {'Mean' : np.mean(df[sel_col]) , 
                        'Minimum' : np.min(df[sel_col]), 
                        'Maximum' : np.min(df[sel_col]), 
                        } 
             df[sel_col] = df[sel_col].apply(lambda x: operator[sel_method] if pd.isnull(x) else x)   

        st.session_state.dataset = data #save value as a state

   
            #showmissing(df) 

def removeduplicate(df) :   
    st.write('below are the duplicated value of your data, you can simply remove the duplicate from your data by clicking the button below the table')
    st.table(df[df.duplicated()]) 
    if(st.button('Remove Duplicate')) : 
        df.drop_duplicates(inplace = True, keep='first') 
    st.session_state.dataset = df
            
    
################################################algo#######################################################



st.title('Welcome!') 
st.write('this web app is a beta version of a completely GUI based exploratory data analysis.') 
st.write('this app will guide you step by step through data analysis based on  [CRISP](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) method. so the first step is to upload your file. ') 
st.write('you can upload your file below using the form')

fileupload()   

st.write('you can also use a dummy data from seaborn [penguins](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv) dataset ')

dummyyy = st.checkbox('i want to use seaborn dataset')  
selected = False

if(dummyyy) : 
    use_dummy_data() 
    df = st.session_state['dataset'] 
  
if(uploaded) :  
    df = st.session_state['dataset']  
    st.header('Select a column for further analysis')
    pick_columns(df)  



if(uploaded) :   
    try : 
        st.header('Missing value') 
        tab_1, tab_2 = st.tabs(["Missing value", "Fill Missing Value"]) 
        with tab_1 : 
            df = st.session_state['dataset']
            showmissing(df) 
        with tab_2 : 
            df = st.session_state['dataset']
            fill_missing(df) 

    except ValueError:
        st.error('Please select a data first')
##removeduplicate 

if(uploaded) :    
    try : 
        st.header('Duplicate Value') 
        df = st.session_state.dataset  
        removeduplicate(df)  
    except ValueError : 
        st.write("you haven't upload a file, please select a file")
#exploratory data analysis
if 'dataset' in st.session_state : 
    eda(st.session_state.dataset)  





 
