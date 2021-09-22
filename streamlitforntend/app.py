import streamlit as st
import cv2
from PIL import Videos,VideoEnhance
import numpy as np
import os

from streamlit.proto.Video_pb2 import Video

# Some utils to make things faster 
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
    st.text("A Virtual eye of the Visually Impaired")
    activities = ["Detection"]
    choice = st.sidebar.selectbox("Select Activity",activities) 

    if choice == 'Detection':
        st.subheader("Object Detection")

        video_file = st.file_uploader("Upload Video")
        if video_file is not None:
            our_video = Video.open(video_file)
            st.text("Original Video")
            st.video(our_video)

        enhance_type = st.sidebar.radio("Enhance Type",["Original","Working"])  
        if enhance_type == 'Working':
            #new_video = pointing to the working file
        
            #st.video(new_video)
        #if __name__ == '__main__':
                 main()        



