import streamlit as st
import cv2
# from PIL import Videos
import numpy as np
import os
from streamlit.proto.Video_pb2 import Video

@st.cache()
def read_classes():
    
    with open('utils/streamUtils.py') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return classes

@st.cache
def load_video(vid):
    video = Videos.open(vid)
    return video


def main():
    """Object detection"""
    st.title("Dolphin-Eye")
    st.text("A Virtual Eye for the Visually Impaired")
    activities = ["Detection"]
    choice = st.sidebar.selectbox("Select Activity",activities) 

    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')



if __name__ == '__main__':
    main()        



