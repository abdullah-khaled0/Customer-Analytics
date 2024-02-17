import streamlit as st
import pandas as pd
import pickle


@st.cache_data
def load_scaler_pca_kmeans_pca():
    scaler = pickle.load(open('Streamlit-Apps/customer-analytics/scaler.pickle', 'rb'))
    pca = pickle.load(open('Streamlit-Apps/customer-analytics/pca.pickle', 'rb'))
    kmeans_pca = pickle.load(open('Streamlit-Apps/customer-analytics/kmeans_pca.pickle', 'rb'))

    return scaler, pca, kmeans_pca


def segmentation_inputs():

    sex = st.radio("Sex", ["Male", "Female"])
    marital_status = st.radio("Marital Status", ["Single", "Non-single"])
    age = st.slider("Age", 18, 76)
    education = st.selectbox("Education", ["Other / Unknown", "High School", "University", "Graduate School"])
    income = st.slider("Income (in USD)", 35832, 309364)
    occupation = st.selectbox("Occupation", ["Unemployed / Unskilled", "Skilled Employee / Official", "Management / Self-employed / Highly Qualified Employee / Officer"])
    settlement_size = st.selectbox("Settlement Size", ["Small City", "Mid-sized City", "Big City"])

    st.write("Selected Values:")
    st.write("Sex:", sex)
    st.write("Marital Status:", marital_status)
    st.write("Age:", age)
    st.write("Education:", education)
    st.write("Income:", income)
    st.write("Occupation:", occupation)
    st.write("Settlement Size:", settlement_size)

    scaler, pca, kmeans_pca = load_scaler_pca_kmeans_pca()

    sex = 0 if sex == "Male" else 1
    marital_status = 0 if sex == "Single" else 1

    if education == "Other / Unknown":
        education = 0
    elif education == "High School":
        education = 1
    elif education == "University":
        education = 2
    elif education == "Graduate School":
        education = 3

    if occupation == "Unemployed / Unskilled":
        occupation = 0
    elif occupation == "Skilled Employee / Official":
        occupation = 1
    elif occupation == "Management / Self-employed / Highly Qualified Employee / Officer":
        occupation = 2

    if settlement_size == "Small City":
        settlement_size = 0
    elif settlement_size == "Mid-sized City":
        settlement_size = 1
    elif settlement_size == "Big City":
        settlement_size = 2

    data = {
        "Sex": sex,
        "Marital status": marital_status,
        "Age": age,
        "Education": education,
        "Income": income,
        "Occupation": occupation,
        "Settlement size": settlement_size
    }

    df = pd.DataFrame([data])
    df_segm_std = scaler.transform(df)
    df_segm_pca = pca.transform(df_segm_std)
    segm_kmeans_pca = kmeans_pca.predict(df_segm_pca)

    result = ''
    if segm_kmeans_pca[0] == 0:
        result = 'standard'
    elif segm_kmeans_pca[0] == 1:
        result = 'career focused'
    elif segm_kmeans_pca[0] == 2:
        result = 'fewer opportunities'
    elif segm_kmeans_pca[0] == 3:
        result = 'well-off'

    st.divider()
    st.header('The segment you belong to is🔍:')
    st.subheader("👉 " + result)
    

