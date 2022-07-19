import joblib
import pandas as pd 
import numpy as np
import streamlit as st
from model_functions import do_predict

st.image('dataset-cover.jpg')
df_clarity = pd.read_csv('clarity.csv')
df_cut = pd.read_csv('cut.csv')

df_color = pd.read_csv('color.csv')

df_models = pd.read_csv('models.csv')

selected_carat = st.number_input('Quantas quilates tem o diamante?',min_value=0.2,max_value=7.0, step=0.01)
selected_clarity = st.selectbox('Qual a claridade do diamante?', df_clarity['clarity'].values)
selected_cut = st.selectbox('Qual o corte do diamante?', df_cut['cut'].values)
selected_color = st.selectbox('Qual a cor do diamante?', df_color['color'].values)

df_new_diamond = pd.DataFrame({'carat_log':np.log(selected_carat),'clarity_num':df_clarity.loc[df_clarity['clarity']==selected_clarity,'clarity_num'].values[0],'cut_num':df_cut.loc[df_cut['cut']==selected_cut,'cut_num'].values[0],'color_num':df_color.loc[df_color['color']==selected_color,'color_num'].values[0]},index=[0])

df_models['loaded_model'] = df_models['model_name'].apply(lambda x: joblib.load(x))

if st.button('Previsão de preço', key=None, help=None, on_click=None, args=None, kwargs=None):
    st.write('$', str(round(do_predict(df_new_diamond.loc[0], df_models = df_models)[0],2)))
    


