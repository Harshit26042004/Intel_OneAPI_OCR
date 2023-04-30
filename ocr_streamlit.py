# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:27:58 2023

@author: ASUS
"""
import tensorflow
import streamlit as st
import keras_ocr
import matplotlib.pyplot as plt



pipeline = keras_ocr.pipeline.Pipeline()

file = st.file_uploader("Upload the File:",accept_multiple_files=True)

st.image(file)

prediction=pipeline.recognize(file)
prediction

fig,axs = plt.subplots(nrows=len(file),figsize=(10,20))
for ax,images,predictions in zip(axs,file,prediction):
    keras_ocr.tools.drawAnnotations(image=images, predictions=predictions,ax=ax)
    
    