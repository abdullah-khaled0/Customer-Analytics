import streamlit as st
import warnings

from customer_analytics_segmentation import *
from purchase_descriptive_analysis import *

warnings.filterwarnings("ignore")


st.title('Customer Analytics📊')


selected_btn = st.radio(
    " ",
    ["Customer Segmentation", "Purchase - Descriptive Analysis"])

st.divider()
if selected_btn == "Customer Segmentation":
    segmentation_inputs()
elif selected_btn == "Purchase - Descriptive Analysis":
    csv_file = "Streamlit-Apps/customer-analytics/purchase_data.csv"
    purch_descriptive_analysis(csv_file)
