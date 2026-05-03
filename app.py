import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import joblib


st.set_page_config(layout='wide')


df2 = pd.read_csv(r'.\files\df1.csv')

df3 = pd.read_csv(r'.\files\df3.csv')


encoded_state = joblib.load("./files/encoded_state.pkl")

encoded_city = joblib.load("./files/encoded_city.pkl")

encoded_availability_status = joblib.load("./files/encoded_availability_status.pkl")

encoded_security = joblib.load("./files/encoded_security.pkl")


# --------------------------------------------------------------------------------------------------------------
                                               

import mlflow.sklearn
import mlflow.pyfunc

mlflow.set_tracking_uri("http://127.0.0.1:5000/")


                                                     # regression model

r_model_name = "XGBoost Regression Production"
r_model_version = 1
r_model_uri = f"models:/{r_model_name}@champion"
r_model = mlflow.sklearn.load_model(r_model_uri)

                                                    # classification model

c_model_name = "XGBoost Classification Production"
c_model_version = 1
c_model_uri = f"models:/{c_model_name}@champion"
c_model = mlflow.sklearn.load_model(c_model_uri)
# --------------------------------------------------------------------------------------------------------------

                                                # sidebar

with st.sidebar:
    select = option_menu("Main Menu", ['Home', 'Data Exploration', 'Predict'])



# --------------------------------------------------------------------------------------------------------------

                                                # Home

if select == "Home":
    
    st.header("Real Estate Investment Advisor: Predicting Property Profitability & Future Value")
    
    images = Image.open('./files/images_2.png')
    st.image(images)

    st.header("About")
    st.write("")
    st.write("""This application helps real estate investors make smarter property decisions using data-driven insights powered by machine learning. It analyzes various property features to classify whether a property is a good investment and predicts the estimated property price after 5 years, helping users understand the long-term potential of their investment before making a purchase.""")
    st.write("")
    st.write("""The system is built as an interactive web application using Streamlit, allowing users to easily input property details and receive instant predictions and recommendations. To ensure reliable model development and experiment tracking, the project integrates MLflow, enabling systematic monitoring of model performance and improvements.""")


# --------------------------------------------------------------------------------------------------------------

                                                # Data Exloration

if select == 'Data Exploration':
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Price & Size Analysis", "📍 Location-based Analysis", "📊 Feature Relationship & Correlation", "🏢 Investment / Amenities / Ownership Analysis"])


    analysis_1 = [
                    '1. What is the distribution of property prices?',
                    '2. What is the distribution of property sizes?', 
                    '3. How does the price per sq ft vary by property type?', 
                    '4. Is there a relationship between property size and price?', 
                    '5.Are there any outliers in price per sq ft or property size?'
                ]
    
    analysis_2 = [
                    '1. What is the average price per sq ft by state?', 
                    '2. What is the average property price by city?',
                    '3. What is the median age of properties by locality?',
                    '4. How is BHK distributed across cities?',
                    '5.What are the price trends for the top 5 most expensive localities?'      
                ]
    
    analysis_3 = [
                    '1. How are numeric features correlated with each other?',
                    '2. How do nearby schools relate to price per sq ft?',
                    '3. How do nearby hospitals relate to price per sq ft?',
                    '4. How does price vary by furnished status?',
                    '5. How does price per sq ft vary by property facing direction?'
                ]
    
    analysis_4 = [
                    '1.How many properties belong to each owner type?',
                    '2. How many properties are available under each availability status?',
                    '3. Does parking space affect property price?',
                    '4. How do amenities affect price per sq ft?',
                    '5. How does public transport accessibility relate to price per sq ft or investment potential?' 
                ]

# ----------------------------------------------------------------------------    

    with tab1:
        st.subheader('💰 Price & Size Analysis')

        question = st.selectbox('Choose a question', analysis_1)


        if question == analysis_1[0]:

            st.write('Distribution (Skewness) of price: ',df2['Price'].skew())

            fig = px.histogram(df2['Price'], nbins = 30)
            fig.update_layout(
                xaxis_title = "Price",
                yaxis_title = "Count", 
                title = 'Price Distribution'
            )

            st.plotly_chart(fig, use_container_width=True)

        if question == analysis_1[1]:

            st.write('Distribution (Skewness) of property size: ',df2['Size_in_SqFt'].skew())


            fig = px.histogram(df2['Size_in_SqFt'], nbins = 30)
            fig.update_layout(
                xaxis_title = "Size_in_SqFt",
                yaxis_title = "Count", 
                title = 'Property size Distribution'
            )

            st.plotly_chart(fig, use_container_width=True)        

        if question == analysis_1[2]:

            fig = px.box(x = df2['Property_Type'], y = df2['Price_per_SqFt'])

            st.plotly_chart(fig, use_container_width=True)    

        if question == analysis_1[3]:

            st.write('Correlation between property size and price: ',df2['Price'].corr(df2['Size_in_SqFt']))

            # fig, ax = plt.subplots()
            # # fig.set_size_inches(2, 2)
            # sns.lineplot(data = df2, x = 'Size_in_SqFt', y = 'Price')
            # ax.set_xlabel("Size in SqFt")
            # ax.set_ylabel("Price")
            # ax.set_title("Price vs Size")

            # st.pyplot(fig)

        if question == analysis_1[4]:

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Price per SqFt")

                from scipy import stats
                z_score_price_per_sqft = stats.zscore(df2['Price_per_SqFt'])
                z_score_price_per_sqft = np.abs(z_score_price_per_sqft)
                st.write('Number of outlier by z-score: ',len(z_score_price_per_sqft[z_score_price_per_sqft > 3]))
                
                fig1 = px.box(df2['Price_per_SqFt'])
                # fig.update_layout(title = 'Price per SqFt')
                st.plotly_chart(fig1, use_container_width=True) 
            with c2:
                st.subheader('Size in SqFt')
                from scipy import stats
                z_score_size_in_sqft = stats.zscore(df2['Size_in_SqFt'])
                z_score_size_in_sqft = np.abs(z_score_size_in_sqft)
                st.write('Number of outlier by z-score: ',len(z_score_size_in_sqft[z_score_size_in_sqft > 3]))

                fig2 = px.box(df2['Size_in_SqFt'])
                # fig.update_layout(title='Size in SqFt')
                st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------        

    with tab2:
        st.subheader("📍 Location-based Analysis")

        question = st.selectbox('Choose a question', analysis_2)
        
        if question == analysis_2[0]:
            q2_1 =df2.groupby(['State'])['Price_per_SqFt'].mean()

            fig = px.line(q2_1)
            st.plotly_chart(fig, use_container_width=True)
                
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(q2_1)
        
        if question == analysis_2[1]:
            q2_2 = df2.groupby(['City'])['Price'].mean()

            fig = px.line(q2_2)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(q2_2)

        if question == analysis_2[2]:
            q2_3 = df2.groupby('Locality')['Age_of_Property'].median()

            st.write('No.of unique locality:', len(df2['Locality'].unique()))

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(q2_3)  

            fig = px.scatter(q2_3)
            st.plotly_chart(fig, use_container_width=True)

        if question == analysis_2[3]:
            q2_4 = df2.groupby(['City', "BHK"])['BHK'].sum().reset_index(name='Total')
            
            st.write('Total no. of Cities: ', len(df2['City'].unique()))

            import altair as alt

            chart = alt.Chart(q2_4).mark_bar().encode(
                x="City:N",
                y="Total:Q",
                color="BHK:N"
            )

            st.altair_chart(chart, use_container_width=True)  

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(q2_4)  
      
        if question == analysis_2[4]:
            top_locality = df2.groupby(['Locality'])['Price'].mean().sort_values(ascending=False).head(5)
            top_5_name = top_locality.index.tolist()
            df_top_5 = df2[df2['Locality'].isin(top_5_name)][['Locality', 'Price', 'Year_Built']]

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(pd.DataFrame(top_5_name, columns=['Top 5 locality'], index=[1, 2, 3, 4, 5]))
            with col2:    
                i = st.selectbox('Select the localities', top_5_name)

            # for i in df_top_5['Locality'].unique():
            fig, ax = plt.subplots(figsize=(4,2))
            sns.lineplot(data = df_top_5, x = 'Year_Built', y = 'Price', hue=df_top_5[df_top_5['Locality'] == i]['Locality'])
            ax.set_title(i, fontsize=10)          # title size
            ax.set_xlabel("Year Built", fontsize=8)   # x label size
            ax.set_ylabel("Price", fontsize=8)        # y label size

            ax.tick_params(axis='x', labelsize=7)     # x axis values size
            ax.tick_params(axis='y', labelsize=7)     # y axis values size

            ax.legend(fontsize=7)                      
            st.pyplot(fig)

# ----------------------------------------------------------------------------        

    with tab3:
        st.subheader("📊 Feature Relationship & Correlation")

        question = st.selectbox('Choose a question', analysis_3)

        if question == analysis_3[0]:
            fig, ax = plt.subplots(figsize=(5,3))
            sns.heatmap(df2[['BHK', 'Size_in_SqFt', 'Price','Price_per_SqFt','Floor_No','Total_Floors','Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals']].corr(), cmap="YlGnBu")
            ax.tick_params(axis='x', labelsize=7)   # x-axis labels
            ax.tick_params(axis='y', labelsize=7)   # y-axis labels            
            st.pyplot(fig)

            st.dataframe(df2[['BHK', 'Size_in_SqFt', 'Price','Price_per_SqFt','Floor_No','Total_Floors','Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals']].corr())

        if question == analysis_3[1]:
            st.write('Correaltion of Nearby School and Price per SqFt', df2['Nearby_Schools'].corr(df2['Price_per_SqFt']))

            fig, ax = plt.subplots(figsize=(4,2))
            sns.lineplot(
                data=df2,
                x='Nearby_Schools',
                y='Price_per_SqFt',
                ax=ax
            )
            ax.set_xlabel("Nearby Schools", fontsize=4)
            ax.set_ylabel("Price per SqFt", fontsize=4)
            ax.tick_params(labelsize=3)
            st.pyplot(fig)            

        if question == analysis_3[2]:
            st.write('Correaltion ofPrice per SqFt and Nearby Hospitals', df2['Nearby_Hospitals'].corr(df2['Price_per_SqFt']))

            fig, ax = plt.subplots(figsize=(4,2))
            sns.lineplot(
                data=df2,
                x='Price_per_SqFt',
                y='Nearby_Hospitals',
                ax=ax
            )
            ax.set_xlabel("Price per SqFt", fontsize=4)
            ax.set_ylabel("Nearby Hospitals", fontsize=4)
            ax.tick_params(labelsize=3)
            st.pyplot(fig)        

        if question == analysis_3[3]:
            st.write(df2.groupby('Furnished_Status')['Price'].mean())

            fig = px.box(df2, x='Price', y='Furnished_Status')
            st.plotly_chart(fig, use_container_width=True)
            
        if question == analysis_3[4]:
            st.write(df2.groupby(['Facing'])['Price_per_SqFt'].mean())

            fig = px.box(df2, x='Price_per_SqFt', y='Facing')
            st.plotly_chart(fig, use_container_width=True)            
            

# ----------------------------------------------------------------------------        

    with tab4:
        st.subheader("🏢 Investment / Amenities / Ownership Analysis")

        question = st.selectbox('Choose a question', analysis_4)        

        if question == analysis_4[0]:
            st.write(df2.groupby(['Property_Type', 'Owner_Type'])['Owner_Type'].count().reset_index(name='Count'))  

        if question == analysis_4[1]:
            st.dataframe(df2.groupby(['Property_Type', 'Availability_Status'])['Availability_Status'].count().reset_index(name='Count'))

        if question == analysis_4[2]:
            st.dataframe(df2.groupby('Parking_Space')['Price'].mean().reset_index())

            fig = px.bar(df2, x='Parking_Space', y='Price')
            st.plotly_chart(fig, use_container_width=True)  

        if question == analysis_4[3]:

            col1, col2 = st.columns(2)
            with col1:
                df2['Amenities'] = df2["Amenities"].apply(lambda x: len(str(x).split(",")))
                st.dataframe(df2.groupby('Amenities')['Price_per_SqFt'].mean())

            fig = px.box(df2, x = 'Amenities', y = 'Price_per_SqFt')
            st.plotly_chart(fig, use_container_width=True)              

        if question == analysis_4[4]:
            col1, col2 = st.columns(2)
            with col1:            
                st.dataframe(df2.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].mean())

            fig = px.box(df2, x = 'Public_Transport_Accessibility', y = 'Price_per_SqFt')
            st.plotly_chart(fig, use_container_width=True)  

# ----------------------------------------------------------------------------        

                                                # Prediction

if select == 'Predict':
    tab1, tab2 = st.tabs(["Property Price Prediction after 5years", "Predict Property is a Good Investment"])

    col1, col2 = st.columns(2)

    with col1:
        # st.write(sorted(df2['State'].unique()))
        states = sorted(df2['State'].unique())
        state = st.selectbox('State', states)

        state_vector = encoded_state.transform([state])
        # st.write(state_vector)

    with col2:
        # st.write(sorted(df2['City'].unique()))
        cities = sorted(df2['City'].unique())
        city = st.selectbox('City', cities)

        city_vector = encoded_city.transform([city])
        # st.write(city_vector)

    col3, col4 = st.columns(2)

    with col3:
        properties = sorted(df2['Property_Type'].unique())
        property_type = st.selectbox('Property Type', properties)

        property_type_vector = [0] *len(properties)
        idx = properties.index(property_type)
        property_type_vector[idx] = 1
        # st.write(property_type_vector)

        # st.write(min(df2['BHK']), max(df2['BKH']))        
        bhk = st.slider('Number of Bedroom', 1, 5)

        # st.write(min(df2['Size_in_SqFt']), max(df2['Size_in_SqFt']))
        size_in_sqft = st.number_input('Size in SqFt', min(df2['Size_in_SqFt']), max(df2['Size_in_SqFt']))

    with col4:
        # st.write(min(df2['Price_per_SqFt']), max(df2['Price_per_SqFt']))
        price_per_sqft = st.number_input('Price in SqFt', min(df2['Price_per_SqFt']), max(df2['Price_per_SqFt']))

    col5, col6 = st.columns(2)

    with col5:
        # st.write(min(df2['Price']), max(df2['Price']))
        price = st.number_input('Price', min(df2['Price']), max(df2['Price']))

        # st.write(min(df2['Year_Built']), max(df2['Year_Built']))
        # year_built = st.number_input('Year Built', min(df2['Year_Built']), max(df2['Year_Built']))

        interior = sorted(df2['Furnished_Status'].unique())
        furnished_status = st.selectbox('Furnished Status', interior)

        furnished_status_vector = [0] * len(interior)
        idx = interior.index(furnished_status)
        furnished_status_vector[idx] = 1
        # st.write(furnished_vector)

        # st.write(min(df2['Floor_No']), max(df2['Floor_No']))
        # floor_no = st.number_input('Floor Number', min(df2['Floor_No']), max(df2['Floor_No'])) 
        
        # st.write(min(df2['Total_Floors']), max(df2['Total_Floors']))
        # total_floors = st.number_input('Total Floors', min(df2['Total_Floors']), max(df2['Total_Floors']))   

        # st.write(min(df2['Age_of_Property']), max(df2['Age_of_Property']))
        age_of_property = st.number_input('Age of Property', min(df2['Age_of_Property']), max(df2['Age_of_Property']))
        
        # st.write(min(df2['Nearby_Schools']), max(df2['Nearby_Schools']))
        nearby_schools = st.slider('Nearby Schools Rating', min(df2['Nearby_Schools']), max(df2['Nearby_Schools']))           

    with col6:     
        # st.write(min(df2['Nearby_Hospitals']), max(df2['Nearby_Hospitals']))
        nearby_hospitals = st.slider('Nearby Hospitals Rating', min(df2['Nearby_Hospitals']), max(df2['Nearby_Hospitals']))          

    with col5:
        # st.write(df2['Public_Transport_Accessibility'].unique())   
        public_transport_accessibility = st.selectbox('Public Transport Accessibility', ['Low', 'Medium', 'High'])
        if public_transport_accessibility == 'Low':
            public_transport_accessibility_vector = 0
        elif public_transport_accessibility == "Medium":
            public_transport_accessibility_vector = 1
        else:
            public_transport_accessibility_vector = 2 
        # st.write(public_transport_accessibility_vector)

        # st.write(df2['Parking_Space'].unique())
        parking_space = int(st.checkbox('Parking Space'))

        # st.write(df2['Security'].unique())
        security = int(st.checkbox('Security Availability'))

        extra = df2['Amenities'][df2['Amenities'].str.len().idxmax()]
        extra = list(sorted(extra.split(',')))
        amenities = st.multiselect('Amenities', extra)

        amenities_vector = [0] * len(extra)
        for i in amenities:
            idx = extra.index(i)
            amenities_vector[idx] = 1
        # st.write(amenities_vector)
        
        # st.write(df2['Facing'].unique())
        direction = sorted(df2['Facing'].unique())
        facing = st.selectbox('Facing', direction)

        facing_vector = [0] *len(direction)
        idx = direction.index(facing)
        facing_vector[idx] = 1
        # st.write(facing_vector)
        
        # st.write(df2['Owner_Type'].unique())
        owners = sorted(df2['Owner_Type'].unique())
        owner_type = st.selectbox('Owner Type', owners)

        owner_type_vector = [0] *len(owners)
        idx = owners.index(owner_type)
        owner_type_vector[idx] = 1

        # st.write(df2['Availability_Status'].unique())
        availability_status = st.selectbox('Availability Status', sorted(df2['Availability_Status'].unique()))
        if availability_status == 'Ready_to_Move':
            availability_status = 0
        else:
            availability_status = 1

    
    input = [bhk, size_in_sqft, price, price_per_sqft, age_of_property, nearby_schools, nearby_hospitals, public_transport_accessibility_vector, parking_space, security, availability_status] 
    for i in [state_vector, city_vector, property_type_vector, furnished_status_vector, amenities_vector, facing_vector, owner_type_vector]:
        input.extend(i)

    # st.write(len(input))
    # st.write(len(df3.columns))
    # col3, col4 = st.columns(2)
    # with col3:
    #     st.write(pd.DataFrame(input))
    # with col4:
    #     st.write(df3.iloc[0])
 

    with tab1:
        if st.button("Predict Price"):
            r_prediction = r_model.predict([input])
            st.success(f"Predicted Price: {int(r_prediction[0])}")

    with tab2:
        if st.button("Investment Type"):
            c_prediction = c_model.predict([input])
            # st.write(c_prediction)
            
            if c_prediction == 1:
                st.success(f"Good Investment")
            else:
                st.warning(f"Bad Investment")
