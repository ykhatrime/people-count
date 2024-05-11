# Python In-built packages
from pathlib import Path

# External packages
import streamlit as st

# Local Modules
import settings
import helper


# Setting page layout
st.set_page_config(
    page_title="People Counter",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Video Tracking And Counting")

# Sidebar
st.sidebar.header("ML Model People Counter")





# Load Pre-trained ML Model
try:
    model = helper.load_model(Path(settings.DETECTION_MODEL))
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {Path(settings.DETECTION_MODEL)}")
    st.error(ex)

st.sidebar.header("Source Video")



helper.run_tracking_video(model)



