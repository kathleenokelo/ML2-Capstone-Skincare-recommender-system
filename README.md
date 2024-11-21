# **Skincare Products Recommendation System**

## **Overview**
The Skincare Products Recommendation System is a Streamlit-based web application designed to help users find skincare products tailored to their specific skin type, preferences, and budget. Leveraging machine learning and content-based filtering, the system analyzes product features, such as ingredients, brands, and price ranges, to recommend the most suitable products. The app also provides an interactive skin type test for users to determine their skin type.

## **Dataset**
The dataset was provided on Kaggle from the link below:
'https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets/data'

It consists of 1,472 entries and 11 columns. Here’s an overview of the columns:

Label: Category of the product (e.g., "Moisturizer").

Brand: Brand name.

Name: Product name.

Price: Price of the product.

Rank: Product rating.

Ingredients: List of ingredients (text field).

Combination, Dry, Normal, Oily, Sensitive: Binary columns indicating suitability for different skin types.

## **Features**
1. **Personalized Recommendations**:
   - Users can filter products by skin type, brand, label, rank range, and price range.
   - Optional ingredient-based recommendations are available for users who wish to search for products containing specific ingredients.

2. **Interactive Skin Type Test**:
   - A multi-question test helps users determine their skin type (e.g., Dry, Oily, Combination, Normal, Sensitive) based on their responses to questions about their skin's behavior in various conditions.

3. **Product Filters**:
   - Filter options for brand, label, price range, and rank range ensure recommendations meet user preferences.
   - Ingredient search allows users to input key ingredients for targeted product suggestions.

4. **User-Friendly Interface**:
   - Built with Streamlit, the app is intuitive and easy to use, with dropdowns, sliders, and text input fields for customization.

---

## **Key Technologies Used**
- **Streamlit**: Framework for creating the interactive web interface.
- **Pandas**: Data manipulation and analysis of the product dataset.
- **Scikit-learn**:
  - **TF-IDF Vectorization**: To analyze and compare product ingredients.
  - **Cosine Similarity**: For ingredient-based content filtering.
- **Python**: Core programming language for system implementation.

---

## **How It Works**
1. **Input Options**:
   - Users input their skin type, desired product label, brand preferences, rank range, and price range.
   - Users can also enter specific ingredients for targeted recommendations.

2. **Recommendation Algorithm**:
   - The system filters the dataset based on user preferences.
   - If ingredient input is provided, the system calculates cosine similarity between user-provided ingredients and product ingredient descriptions using TF-IDF vectorization.

3. **Output**:
   - A list of recommended products is displayed, showing details such as the product name, label, brand, ingredients, and rank.

4. **Skin Type Test**:
   - Users answer questions about their skin's oiliness, sensitivity, and reactions to environmental factors.
   - The test determines the user's skin type and recommends products accordingly.

---

## **How to Run the Application**
1. Clone the repository:
   ```bash
   git clone <'https://github.com/kathleenokelo/ML2-Capstone-Skincare-recommender-system'>
   ```
2. Navigate to the project directory:
   ```bash
   cd ML2-Capstone-Skincare-recommender-system

   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run deploy.py
   ```
5. Access the app in your web browser at `http://localhost:8501`.

---




## **Future Improvements**
1. **Deployment**:
   - Resolve current deployment issues to make the app available on platforms like Streamlit Sharing or Heroku.
2. **Enhanced Recommendation System**:
   - Incorporate collaborative filtering or deep learning for more robust recommendations.
3. **Expanded Dataset**:
   - Include user reviews, product ratings, and additional product attributes for better insights.
4. **Multi-Language Support**:
   - Add support for non-English-speaking users to increase accessibility.

---

## **Conclusion**
This Skincare Products Recommendation System provides a valuable tool for individuals seeking personalized skincare solutions. The combination of ingredient-based recommendations and interactive skin type testing ensures a tailored experience for every user.


