from ctypes.wintypes import POINT
from distutils import errors
from distutils.command.upload import upload
from functools import cache
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
from streamlit_folium import folium_static
import folium 




#global var 

downloaded = False 
uploaded = False
selected = False
sns.set_style("whitegrid")
uploaded = False 
dataframe = pd.DataFrame() 
st.session_state.selected = False
def fileupload() :  
    uploaded_file = st.file_uploader("Choose a file") 
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        global dataframe
        dataframe = pd.read_csv(uploaded_file, on_bad_lines='skip')
        st.write(dataframe)  
        global uploaded 
        uploaded = True 
        return dataframe
@st.experimental_singleton
def use_dummy_data() : 
    global uploaded 
    global dataframe
    data = sns.load_dataset('penguins')  
    dataframe = data 
    uploaded = True  
    st.write(dataframe) 
    return dataframe
    

@st.experimental_singleton
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
    selected_columns = st.multiselect('please select the column you want to analyze from your data : ',np.sort(df.columns), key='select column')
    @st.experimental_singleton
    def return_val(df , val) :     
        boole = True
        st.session_state.dataset = df[val]  
    if (st.button('confirm')) : 
        return_val(df, selected_columns)

@st.experimental_singleton(suppress_st_warning=True)
def eda (df) : 
    global uploaded
    st.subheader('You can find a brief overview of your data below by looking at the profile page of your data.')     
    pr = df.profile_report()
    st_profile_report(pr) 

@st.experimental_singleton
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

@st.experimental_singleton(suppress_st_warning=True)
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
        if(pd.notnull(sel_group)) : 
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

    
        elif(pd.isnull(sel_group)) :   
            if(st.button('fill null values')) : 
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
          
def spatial_transform(data) : 
    import geopandas as gpd 
    from shapely import geometry
    st.header('Spatial Transformation') 
    st.subheader('is there any longitude or latitude in your data?')
    longitude = st.radio('Longitude',data.columns) 
    latitude = st.radio('Latitude',data.columns) 
    if(st.button('convert')) :
        geom = [] 
        for x,y in zip(data[longitude], data[latitude]) : 
            geom.append(geometry.Point(x,y)) 
        geodataframe = gpd.GeoDataFrame(data, geometry=geom)  
        st.session_state['Geodataset'] = geodataframe 
        world =  gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")) 
        m = world.explore(name = 'polygon of world map')  
        geodataframe.explore(m = m, name = 'map from your data')
        folium.TileLayer('Stamen Terrain', control=True).add_to(m)  # use folium to add alternative tiles
        folium.LayerControl().add_to(m)  # use folium to add layer  
        st.subheader('Voila , your data has been successfully converted into Geodataframe file.') 
        st.write('for further information about Geodataframe, you can read the document provided by geopandas by clicking [here](https://geopandas.org/en/stable/docs/reference/geodataframe.html)')
        st.write('and voala, this is a map generated based on the longitude and latitude in your data.')
        st.markdown(folium_static(m), unsafe_allow_html=True)
 

def check_outlier(data) : 
    st.header('Checking outlier') 
    st.subheader('wait, what is an outlier?') 
    st.write('outlier is a value stored in your data that differs significantly from other observations') 
    st.write('if you want to have a better explanation about outlier, checkout the cool explanation [here](https://medium.com/analytics-vidhya/its-all-about-outliers-cbe172aa1309)') 
    st.write("let's visualize the outlier in your data, please select the numerical value from your data")  
    outlier_variab = st.selectbox('select the numerical value from your data',data.select_dtypes(include='number').columns) 
    method = st.selectbox('please choose the method for detecting outlier',['iqr' , 'std'])   
    plotkind = st.selectbox('what kind of plot you want to use for detecting outlier',['hist' , 'boxplot']) 
    fetched_data = data[outlier_variab]
    min_slider = (int(fetched_data.min().round())) 
    max_slider = (int(fetched_data.max().round()))  

    filter_value = st.slider('you can also make a filter in your data : ',min_slider,max_slider) 
    operand = st.selectbox('filter : ', ['>', '<', 'no filter'])
    
    #function inside function 
    
    
    def visualize_data(data,x,method, plot, filter_value = 0, operand = 'no filter') :   
        data = data.dropna(subset =[x]) 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if data.empty : 
            return 'No Values generated, please select another values or operations for filtering'
        if (filter_value!=0) & (operand!= 'no filter') : 
            if operand=='<' : 
                data = data[data[x] < filter_value] 
            elif operand =='>' :  
                data = data[data[x] > filter_value] 
        #if x not in df.columns : 
        #  return 'columns are not in dataframe'
        mean = np.mean(data[x]) 
        median = np.median(data[x]) 
        min = np.min(data[x]) 
        max = np.max(data[x])   
        std = np.std(data[x]) 
        Q1 = np.quantile(data[x], 0.25) 
        Q3 = np.quantile(data[x],0.75) 
        IQR = Q3-Q1 
        lowerbound = Q1-(1.5*IQR) 
        upperbound = Q3+(1.5*IQR)
        plt.figure(figsize=(15,11))   
        if plot == 'hist' :  
            fig, ax = plt.subplots() 
            ax = sns.histplot(data=data, x=x, kde=True)    
            #g.set(ylim=(0, data[x].max())) 
            ax.set(xlim=(0, data[x].max()))
            if (method=='iqr') :
                plt.axvline(x = np.mean(data[x]), color= 'r', ls=':', label='mean = '+str(np.mean(data[x])), lw=3)     
                plt.axvline(x = np.median(data[x]), color= 'y', ls='--', label='median = '+str(np.median(data[x])), lw=1.8)  
                plt.axvline(x = Q1, color= 'g', ls='--', label='Q1 = '+str(Q1), lw=1.8) 
                plt.axvline(x = Q3, color= 'b', ls='--', label='Q3 = '+str(Q3), lw=1.8) 
                plt.axvline(x = lowerbound, color= 'g', ls='--', label='lowerbound = '+str(lowerbound), lw=1.8) 
                plt.axvline(x = upperbound, color= 'b', ls='--', label='upperbound = '+str(upperbound), lw=1.8)   
            else : 
                plt.axvline(x = mean , color= 'r', ls='--', label='mean = '+str(mean), lw=3)  
                plt.axvline(x = mean + std, color= 'g', ls='--', label='1 std = '+str(std), lw=1.8) 
                plt.axvline(x = mean + (2*std), color= 'y', ls='--', label='2 std = '+str(mean + (2*std)), lw=1.8) 
                plt.axvline(x = mean + (3*std), color= 'b', ls='--', label='3 std = '+str(mean + (3*std)), lw=1.8)   
                plt.axvline(x = mean - std, color= 'g', ls='--', label='-1 std = '+str(std), lw=1.8) 
                plt.axvline(x = mean - (2*std), color= 'y', ls='--', label='-2 std = '+str(mean + (2*std)), lw=1.8) 
                plt.axvline(x = mean - (3*std), color= 'b', ls='--', label='-3 std = '+str(mean + (3*std)), lw=1.8)  

            #plt.axvline(x = statistics.mode(data[columns]), color= 'g', ls='--', label='mode = '+str(statistics.mode(data[columns])), lw=2.5)      
            plt.legend(loc='upper right')  
            plt.title(label = 'Outlier and Boundaries of ' + x + ' with ' + str(method)) 
            plt.show() 
        else :  
            fig, ax = plt.subplots() 
            ax = sns.boxplot(data=data, x=x)    
            ax.set(xlim=(0, data[x].max()))  
            if (method=='iqr') :
                plt.axvline(x = np.mean(data[x]), color= 'r', ls=':', label='mean = '+str(np.mean(data[x])), lw=3)     
                plt.axvline(x = np.median(data[x]), color= 'y', ls='--', label='median = '+str(np.median(data[x])), lw=1.8)  
                plt.axvline(x = Q1, color= 'g', ls='--', label='Q1 = '+str(Q1), lw=1.8) 
                plt.axvline(x = Q3, color= 'b', ls='--', label='Q3 = '+str(Q3), lw=1.8) 
                plt.axvline(x = lowerbound, color= 'g', ls='--', label='lowerbound = '+str(lowerbound), lw=1.8) 
                plt.axvline(x = upperbound, color= 'b', ls='--', label='upperbound = '+str(upperbound), lw=1.8)   
            else : 
                plt.axvline(x = mean , color= 'r', ls='--', label='mean = '+str(mean), lw=3)  
                plt.axvline(x = mean + std, color= 'g', ls='--', label='1 std = '+str(std), lw=1.8) 
                plt.axvline(x = mean + (2*std), color= 'y', ls='--', label='2 std = '+str(mean + (2*std)), lw=1.8) 
                plt.axvline(x = mean + (3*std), color= 'b', ls='--', label='3 std = '+str(mean + (3*std)), lw=1.8)   
                plt.axvline(x = mean - std, color= 'g', ls='--', label='-1 std = '+str(std), lw=1.8) 
                plt.axvline(x = mean - (2*std), color= 'y', ls='--', label='-2 std = '+str(mean + (2*std)), lw=1.8) 
                plt.axvline(x = mean - (3*std), color= 'b', ls='--', label='-3 std = '+str(mean + (3*std)), lw=1.8)  

            #plt.axvline(x = statistics.mode(data[columns]), color= 'g', ls='--', label='mode = '+str(statistics.mode(data[columns])), lw=2.5)      
            plt.legend(loc='upper right')  
            plt.title(label = 'Outlier and Boundaries of ' + x + ' with ' + str(method)) 
            plt.show()    
            return fig
    #end function  
    def show() : 
        st.pyplot(visualize_data(data, outlier_variab, method, plotkind, filter_value=filter_value, operand=operand))

    
    show() 


@st.experimental_singleton
def bivar_visual(data , X, y , kind) : 
  plt.figure(figsize=(10,10))
  if kind == 'scatter' : 
    sns.scatterplot(data[X], data[y]) 
  elif kind == 'line' : 
    sns.lineplot(data[X], data[y]) 
  if kind == 'swarm' : 
    sns.swarmplot(data[X], data[y]) 
  elif kind == 'regplot' : 
    sns.regplot(data[X], data[y]) 
  elif kind =='violin' : 
    sns.violinplot(data[X], data[y]) 
  elif kind =='bar' : 
    sns.barplot(data[X], data[y], palette = 'coolwarm')    

    
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

  
if(uploaded) :  
    df = dataframe  
    st.header('Select a column for further analysis')
    pick_columns(df)  



if('dataset' in st.session_state) :   
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
if('dataset' in st.session_state) :    
    st.header('Duplicate Value') 
    df = st.session_state.dataset  
    removeduplicate(df)  


#exploratory data analysis
st.header('Click the button below to analyze your data')
if st.button('Analyze my Data') : 
    eda(st.session_state.dataset)  

if ('dataset' in st.session_state) : 
    dataframe = st.session_state.dataset.copy()
    spatial_transform(dataframe)  
    st.header('Here comes the outlier')     
    outlier = st.session_state.dataset.copy()
    check_outlier(outlier)     











 
