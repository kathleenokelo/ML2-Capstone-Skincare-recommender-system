#!/usr/bin/env python
# coding: utf-8

# # Skincare Recommender System

# ## 1. Imports

# In[60]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[61]:


df = pd.read_csv(r"C:\Users\ADMIN\Downloads\ML2\cosmetics.csv")


# In[62]:


df.head(10)


# In[63]:


df.sample(5)


# In[64]:


df.info()


# In[65]:


df.describe()


# In[66]:


df.columns


# In[67]:


df.isnull().sum()


# In[68]:


df[df.duplicated()]


# In[69]:


df.shape


# #### Total Brand Names

# In[70]:


df['Brand'].unique()


# In[71]:


df['Brand'].value_counts()


# In[72]:


total_brands_count = len(df["Brand"].unique())
total_brands_count


# In[73]:


df['Label'].unique()


# In[74]:


total_Labels_count = len(df["Label"].unique())
total_Labels_count


# In[75]:


max_price_brand = df.loc[df["Price"].idxmax(), "Brand"]
print("Brand with the highest price:", max_price_brand)


# In[76]:


max_price_Label = df.loc[df["Price"].idxmax(), "Label"]
print("Label with the highest price:", max_price_Label)


# In[77]:


print("Brand with the highest rating:", df.loc[df["Rank"].idxmax(), "Brand"])


# In[78]:


all_skin_types_count = df[(df["Combination"] == 1) & 
                          (df["Dry"] == 1) & 
                          (df["Normal"] == 1) & 
                          (df["Oily"] == 1) & 
                          (df["Sensitive"] == 1)]["Brand"].nunique()

print("Number of brands suitable for all skin types:", all_skin_types_count)


# In[79]:


top_5_highest_rank_brands = df.sort_values(by="Rank", ascending=False).head(5)["Brand"]
print("Top 5 highest-ranked brands:", top_5_highest_rank_brands.tolist())


# ## 2. EDA

# In[80]:


# Plot the distribution of prices
plt.figure(figsize=(10, 5))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# Prices are skewed, with a peak at the lower end, suggesting many products are relatively affordable

# In[81]:


# Plot the distribution of ranks
plt.figure(figsize=(10, 5))
sns.histplot(df['Rank'], bins=30, kde=True, color="orange")
plt.title('Rank Distribution')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.show()


# The ratings are fairly concentrated around the middle to high range, indicating generally well-reviewed products.

# In[82]:


# Count plot for product labels (categories)
plt.figure(figsize=(12, 6))
sns.countplot(y = df['Label'],hue=df['Label'], order=df['Label'].value_counts().index, palette="viridis",legend=False)
plt.title('Product Label Distribution')
plt.xlabel('Count')
plt.ylabel('Product Label')
plt.show()


#  Certain product types dominate the dataset, with "Moisturizer" and "Cleanser" being the most frequent.

# ## Data Preprocessing

# #### Tokenization

# In[83]:


# import library for tokenization
from nltk.tokenize import word_tokenize


# In[84]:


#Preparing the 'Ingredients' column for feature extraction


df['features'] = df['Ingredients'].apply(word_tokenize)


# In[85]:


df['tokenized_ingredients'] = df['Ingredients'].apply(lambda x: x.lower().split(", "))


# In[86]:


df.head()


# In[87]:


# Create a dictionary with unique ingredients

# Flatten all tokenized ingredients into a single list
all_ingredients = [ingredient for sublist in df['tokenized_ingredients'] for ingredient in sublist]

# Create a unique list of ingredients and assign an index to each
unique_ingredients = list(set(all_ingredients))
ingredient_idx = {ingredient: index for index, ingredient in enumerate(unique_ingredients)}


# In[88]:


ingredient_idx


# #### Creating a binary bag-of-words

# In[89]:


import numpy as np

# Initialize a matrix of zeros with dimensions [num_products, num_unique_ingredients]
num_products = len(df)
num_ingredients = len(unique_ingredients)
bag_of_words_matrix = np.zeros((num_products, num_ingredients), dtype=int)

# Fill the matrix based on the presence of each ingredient in each product
for i, ingredients in enumerate(df['tokenized_ingredients']):
    for ingredient in ingredients:
        if ingredient in ingredient_idx:
            bag_of_words_matrix[i, ingredient_idx[ingredient]] = 1


# In[90]:


df.info()


# In[91]:


df = df.reset_index()


# In[92]:


df.info()


# In[93]:


# Convert to DataFrame with ingredient names as column headers

bow_df = pd.DataFrame(bag_of_words_matrix, columns=unique_ingredients, index=df['index'])


# In[94]:


bow_df


# #### Word Embedding

# In[95]:


# Join tokens into a single string for each row in the 'features' column
df['features'] = df['features'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)


# In[96]:


# Initialize the TF-IDF Vectorizer and transform the features

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])


# In[97]:


tfidf_matrix


# Using TD-IDF + SVD method

# In[98]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Join ingredients lists into strings for TF-IDF vectorization
df['ingredient_string'] = df['tokenized_ingredients'].apply(lambda x: ' '.join(x))

# Apply TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['ingredient_string'])

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=100)  # n_components is the embedding size
ingredient_embeddings = svd.fit_transform(tfidf_matrix.T)

# Create a dictionary for the embeddings
ingredient_embedding = {ingredient: ingredient_embeddings[i] for i, ingredient in enumerate(tfidf.get_feature_names_out())}


# In[99]:


ingredient_embedding


# #### Cosine Similarity

# In[100]:


#Calculate cosine similarity for all products in the dataset

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[101]:


cosine_sim


# Generate product vectors by averaging Ingredient embeddings. This will give each product that represents the combined ingredients

# In[102]:


# Function to get the average vector for each product
def get_product_vector(ingredients, ingredient_embedding):
    # Get embeddings for each ingredient in the product, if it exists in the dictionary
    ingredient_vectors = [ingredient_embedding[ingredient] for ingredient in ingredients if ingredient in ingredient_embedding]
    # Average the ingredient vectors to get the product vector
    if ingredient_vectors:
        return np.mean(ingredient_vectors, axis=0)
    else:
        return np.zeros(len(next(iter(ingredient_embedding.values()))))  # return zero vector if no ingredients found

# Apply the function to each product's ingredients
df['product_vector'] = df['tokenized_ingredients'].apply(lambda x: get_product_vector(x, ingredient_embedding))



# Use cosine similarity to generate a similarity matrix of all product vectors

# In[103]:


# Stack product vectors to create a matrix
product_vectors = np.stack(df['product_vector'].values)

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(product_vectors)

# Convert to DataFrame for better readability (optional)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=df.index, columns=df.index)


# In[104]:


cosine_sim_df


# A value of 1 indicates high similarity, while a value close to 0 indicates low similarity

# In[105]:


# Finding the top similar products for a given product

def get_top_similar_products(product_id, cosine_sim_df, top_n=5):
    # Exclude self-similarity and get top N similar products
    similar_products = cosine_sim_df[product_id].sort_values(ascending=False).iloc[1:top_n+1]
    return similar_products

# Get top 5 similar products for product at index 0
top_similar_products = get_top_similar_products(0, cosine_sim_df, top_n=5)
print(top_similar_products)


# #### Using reverse mapping for easy lookup

# In[106]:


# Create a reverse map of indices to product names
index_to_product = pd.Series(df['Name'].values, index=df.index).to_dict()




# In[107]:


# Example of how to use this map
product_index = 0  # Suppose you want the name of the product at index 0
product_name = index_to_product[product_index]
print(f"Product at index {product_index}: {product_name}")


# In[108]:


# Create a reverse map of indices and product names for easy lookup
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()


# In[109]:


indices


# #### Creating a function to recommend products based on filters and optional ingredients

# In[110]:


# Create a function to recommend products based on filters and optional ingredients
def recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input=None, num_recommendations=10):
    recommended_products = df[df[skin_type] == 1]
    
    if label_filter != 'All':
        recommended_products = recommended_products[recommended_products['Label'] == label_filter]
    
    recommended_products = recommended_products[
        (recommended_products['Rank'] >= rank_filter[0]) & 
        (recommended_products['Rank'] <= rank_filter[1])
    ]
    
    if brand_filter != 'All':
        recommended_products = recommended_products[recommended_products['Brand'] == brand_filter]
    
    recommended_products = recommended_products[
        (recommended_products['Price'] >= price_range[0]) & 
        (recommended_products['Price'] <= price_range[1])
    ]

    if ingredient_input:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['Ingredients'])
        input_vec = vectorizer.transform([ingredient_input])
        cosine_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
        recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
        ingredient_recommendations = df.iloc[recommended_indices]
        recommended_products = recommended_products[recommended_products.index.isin(ingredient_recommendations.index)]
    
    return recommended_products.sort_values(by=['Rank']).head(num_recommendations)


# This function:
# 
# -Filters products based on skin type, label, rank, brand, and price range.
# 
# -Optionally filters products based on ingredient similarity if ingredients are provided.
# 
# -Returns the top recommended products sorted by rank.
