import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle




@st.cache_data
def load_data(csv_file):
    df_purchase = pd.read_csv(csv_file)
    scaler = pickle.load(open('Streamlit-Apps/customer-analytics/scaler.pickle', 'rb'))
    pca = pickle.load(open('Streamlit-Apps/customer-analytics/pca.pickle', 'rb'))
    kmeans_pca = pickle.load(open('Streamlit-Apps/customer-analytics/kmeans_pca.pickle', 'rb'))

    features = df_purchase[['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']]
    df_purchase_segm_std = scaler.transform(features)
    df_purchase_segm_pca = pca.transform(df_purchase_segm_std)
    purchase_segm_kmeans_pca = kmeans_pca.predict(df_purchase_segm_pca)
    df_purchase_predictors = df_purchase.copy()
    df_purchase_predictors['Segment'] = purchase_segm_kmeans_pca
    return df_purchase_predictors
    

def purch_descriptive_analysis(csv):
    df = load_data(csv)
    st.write(df.head(100))
    sidebar(df)


def sidebar(df):
    choose_explore_option(df)
    descriptive_analysis(df)

def choose_explore_option(df):
    st.sidebar.header("Exploration & Descriptive Analysis")

    explore_ops = st.sidebar.multiselect('Explore Data',
        ['shape', 'cols', 'describe', 'dtypes', 'nan_vals'],
        ['shape', "cols", "describe", "dtypes", "nan_vals"])

    if "shape" in explore_ops:
        st.subheader("Shape")
        st.write(df.shape)

    if "cols" in explore_ops:
        st.subheader("Columns")
        st.write(df.columns)

    if "describe" in explore_ops:
        st.subheader("Describe")
        st.write(df.describe())

    if "dtypes" in explore_ops:
        st.subheader("dtypes")
        st.write(df.dtypes)

    if "nan_vals" in explore_ops:
        st.subheader("NaN Values")
        st.write(df.isnull().sum())

def descriptive_analysis(df):

    ops = st.sidebar.multiselect(
        'Descriptive Analysis by Segments',
        ['Segment Proportions', 'Describe Purchase Data', 'Average Number of Store Visits by Segment', 'Average Number of Purchases by Segment', 'Average Brand Choice by Segment', 'Revenue Brands'])
    
    df_purchase_descr = describe_purchase_data(df)

    segments_mean = df_purchase_descr.groupby(['Segment']).mean()
    segments_std = df_purchase_descr.groupby(['Segment']).std()

    segm_prop = df_purchase_descr[['N_Purchases', 'Segment']].groupby(['Segment']).count() / df_purchase_descr.shape[0]
    segm_prop = segm_prop.rename(columns = {'N_Purchases': 'Segment Proportions'})
    
    if "Segment Proportions" in ops:
        fig, ax = plt.subplots(figsize=(10, 8))

        st.subheader("Segment Proportions")
        plt.pie(segm_prop['Segment Proportions'],
                labels = ['Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'],
                autopct = '%1.1f%%', 
                colors = ('b', 'g', 'r', 'orange'))
        ax.set_title("Segment Proportions")
        plt.show()
        st.pyplot(fig)
    
    if "Describe Purchase Behaivours by Segments" in ops:
        st.subheader("Describe Purchase Behaivours by Segments")
        st.write(df_purchase_descr.head(100))

    if "Average Number of Store Visits by Segment" in ops:
        fig, ax = plt.subplots(figsize=(10, 8))

        st.subheader("Average Number of Store Visits by Segment")
        plt.bar(x = (0, 1, 2, 3),
                tick_label = ('Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'), 
                height = segments_mean['N_Visits'],
                yerr = segments_std['N_Visits'],
                color = ('b', 'g', 'r', 'orange'))
        plt.xlabel('Segment')
        plt.ylabel('Number of Store Visits')
        plt.title('Average Number of Store Visits by Segment')
        plt.show()
        st.pyplot(fig)

    if "Average Number of Purchases by Segment" in ops:
        fig, ax = plt.subplots(figsize=(10, 8))

        st.subheader("Average Number of Purchases by Segment")
        plt.bar(x = (0, 1, 2, 3),
                tick_label = ('Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'), 
                height = segments_mean['N_Purchases'],
                yerr = segments_std['N_Purchases'],
                color = ('b', 'g', 'r', 'orange'))
        plt.xlabel('Segment')
        plt.ylabel('Purchase Incidences')
        plt.title('Number of Purchases by Segment')
        plt.show()
        st.pyplot(fig)


    if "Average Brand Choice by Segment" in ops:
        fig, ax = plt.subplots(figsize=(10, 8))

        st.subheader("Average Brand Choice by Segment")
        df_purchase_incidence = df[df['Incidence'] == 1]
        brand_dummies = pd.get_dummies(df_purchase_incidence['Brand'], prefix = 'Brand', prefix_sep = '_')
        brand_dummies['Segment'], brand_dummies['ID'] = df_purchase_incidence['Segment'], df_purchase_incidence['ID']
        temp = brand_dummies.groupby(['ID'], as_index = True).mean()
        mean_brand_choice = temp.groupby(['Segment'], as_index = True).mean()
        sns.heatmap(mean_brand_choice,
                    vmin = 0, 
                    vmax = 1,
                    cmap = 'PuBu',
                    annot = True)
        plt.yticks([0, 1, 2, 3], ['Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'], rotation = 45, fontsize = 9)
        plt.title('Average Brand Choice by Segment')
        plt.show()
        st.pyplot(fig)

    if "Revenue Brands" in ops:
        temp = df[df['Brand'] == 1]
        temp.loc[:, 'Revenue Brand 1'] = temp['Price_1'] * temp['Quantity']
        segments_brand_revenue = pd.DataFrame()
        segments_brand_revenue[['Segment', 'Revenue Brand 1']] = temp[['Segment', 'Revenue Brand 1']].groupby(['Segment'], as_index = False).sum()

        temp = df[df['Brand'] == 2]
        temp.loc[:, 'Revenue Brand 2'] = temp['Price_2'] * temp['Quantity']
        segments_brand_revenue[['Segment', 'Revenue Brand 2']] = temp[['Segment', 'Revenue Brand 2']].groupby(['Segment'], as_index = False).sum()

        temp = df[df['Brand'] == 3]
        temp.loc[:,'Revenue Brand 3'] = temp['Price_3']*temp['Quantity']
        segments_brand_revenue[['Segment','Revenue Brand 3']] = temp[['Revenue Brand 3','Segment']].groupby(['Segment'], as_index = False).sum()

        temp = df[df['Brand'] == 4]
        temp.loc[:,'Revenue Brand 4'] = temp['Price_4']*temp['Quantity']
        segments_brand_revenue[['Segment','Revenue Brand 4']] = temp[['Revenue Brand 4','Segment']].groupby(['Segment'], as_index = False).sum()

        temp = df[df['Brand'] == 5]
        temp.loc[:,'Revenue Brand 5'] = temp['Price_5']*temp['Quantity']
        segments_brand_revenue[['Segment','Revenue Brand 5']] = temp[['Revenue Brand 5','Segment']].groupby(['Segment'], as_index = False).sum()

        segments_brand_revenue['Total Revenue'] = (segments_brand_revenue['Revenue Brand 1'] +
                                           segments_brand_revenue['Revenue Brand 2'] +
                                           segments_brand_revenue['Revenue Brand 3'] +
                                           segments_brand_revenue['Revenue Brand 4'] +
                                           segments_brand_revenue['Revenue Brand 5'] )
        
        segments_brand_revenue['Segment Proportions'] = segm_prop['Segment Proportions']
        segments_brand_revenue['Segment'] = segments_brand_revenue['Segment'].map({0:'Standard',
                                                                                1:'Career-Focused',
                                                                                2:'Fewer-Opportunities',
                                                                                3:'Well-Off'})
        segments_brand_revenue = segments_brand_revenue.set_index(['Segment'])

        st.subheader("Revenue Brands")
        st.write(segments_brand_revenue)



def describe_purchase_data(df):
    temp1 = df[['ID', 'Incidence']].groupby(['ID'], as_index = False).count()
    temp1 = temp1.set_index('ID')
    temp1 = temp1.rename(columns = {'Incidence': 'N_Visits'})

    temp2 = df[['ID', 'Incidence']].groupby(['ID'], as_index = False).sum()
    temp2 = temp2.set_index('ID')
    temp2 = temp2.rename(columns = {'Incidence': 'N_Purchases'})
    temp3 = temp1.join(temp2)

    temp3['Average_N_Purchases'] = temp3['N_Purchases'] / temp3['N_Visits']

    temp4 = df[['ID', 'Segment']].groupby(['ID'], as_index = False).mean()
    temp4 = temp4.set_index('ID')
    df_purchase_descr = temp3.join(temp4)

    return df_purchase_descr
