import streamlit as st
import recommender_system as rc


def recommendation_page():
    st.title('Skincare Recommendation System')
    st.write('Welcome!')
    st.write('Take a free skin type test on the next page to get to know your skin type.')
    st.write('Use the filters below to find skincare products suited for your skin type.')
    
    

    col1, col2, col3 = st.columns(3)

    with col1:
        skin_type = st.selectbox('Select your skin type:', ('Combination', 'Dry', 'Normal', 'Oily', 'Sensitive'))

    unique_labels = rc.df['Label'].unique().tolist()
    unique_labels.insert(0, 'All')

    with col2:
        label_filter = st.selectbox('Filter by label (optional):', unique_labels)

    with col1:
        rank_filter = st.slider(
            'Select rank range:', 
            min_value=int(rc.df['Rank'].min()), 
            max_value=int(rc.df['Rank'].max()), 
            value=(int(rc.df['Rank'].min()), int(rc.df['Rank'].max()))
        )

    unique_brands = rc.df['Brand'].unique().tolist()
    unique_brands.insert(0, 'All')

    with col2:
        brand_filter = st.selectbox('Filter by brand (optional):', unique_brands)

    with col3:
        price_range = st.slider(
            'Select price range:', 
            min_value=float(rc.df['Price'].min()), 
            max_value=float(rc.df['Price'].max()), 
            value=(float(rc.df['Price'].min()), float(rc.df['Price'].max()))
        )

    st.write("Or enter ingredients to get product recommendations (optional):") 
    ingredient_input = st.text_area("Ingredients (comma-separated)", "")

    if st.button('Find similar products'):
        top_recommended_products = rc.recommend_cosmetics(
            skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input
        )
        
        st.subheader('Recommended Products')
        st.write(top_recommended_products[['Label', 'Brand', 'Name', 'Ingredients', 'Rank']])

def skin_type_test_page():
    st.title('Skin Type Test')
    st.write('Answer the questions below to determine your skin type.')

    # Questions to assess skin characteristics
    q1 = st.radio("How does your skin feel in the morning?", ["Oily", "Dry", "Normal", "Sensitive"])
    q2 = st.radio("How often do you experience acne or breakouts?", ["Often", "Rarely", "Sometimes", "Never"])
    q3 = st.radio("How does your skin feel after washing?", ["Tight and dry", "Greasy", "Normal", "Itchy"])
    q4 = st.radio("How would you describe the size of your pores?", ["Large and visible", "Small and not visible", "Medium size", "Changes with weather"])
    q5 = st.radio("Does your skin become red or irritated easily?", ["Yes, very easily", "No, hardly ever", "Sometimes", "Only with certain products"])
    q6 = st.radio("How does your skin feel by midday?", ["Shiny or oily", "Dry or flaky", "Normal", "Sensitive or irritated"])
    q7 = st.radio("How does your skin respond to seasonal changes?", ["Gets oilier in hot weather, dry in cold weather", "Feels dry year-round", "Feels balanced all year", "Becomes irritated easily with temperature changes"])

    # Logic to determine skin type based on responses
    if st.button("Submit"):
        oily_count = sum([q1 == "Oily", q2 == "Often", q3 == "Greasy", q4 == "Large and visible", q6 == "Shiny or oily"])
        dry_count = sum([q1 == "Dry", q3 == "Tight and dry", q4 == "Small and not visible", q6 == "Dry or flaky", q7 == "Feels dry year-round"])
        normal_count = sum([q1 == "Normal", q2 == "Sometimes", q3 == "Normal", q6 == "Normal", q7 == "Feels balanced all year"])
        sensitive_count = sum([q1 == "Sensitive", q5 == "Yes, very easily", q3 == "Itchy", q6 == "Sensitive or irritated", q7 == "Becomes irritated easily with temperature changes"])

        # Determine the most likely skin type based on answers
        if oily_count >= 3:
            skin_type = "Oily"
        elif dry_count >= 3:
            skin_type = "Dry"
        elif sensitive_count >= 3:
            skin_type = "Sensitive"
        elif normal_count >= 3:
            skin_type = "Normal"
        else:
            skin_type = "Combination"

        st.write(f"Based on your answers, your skin type is: **{skin_type}**")




# Main function to select page
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Recommendation System", "Skin Type Test"])

    if page == "Recommendation System":
        recommendation_page()
    elif page == "Skin Type Test":
        skin_type_test_page()

if __name__ == "__main__":
    main()
