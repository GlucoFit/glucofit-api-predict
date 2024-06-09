from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('./model/Deep_content_based_model.h5')

# Load the encoders and preprocessors
ohe_menu = OneHotEncoder(handle_unknown='ignore')
ohe_user = OneHotEncoder(handle_unknown='ignore')

# Dummy fitting encoders with the same structure as training data
# You should fit these encoders with your full training data or load fitted encoders
menu_df = pd.read_csv("./csv/menu.csv")
user_df = pd.read_csv("./csv/user.csv")

menu_cat_features = menu_df[['diet_labels', 'recipe_name']]
user_cat_features = user_df[['diet_labels', 'preferred_food']]

ohe_menu.fit(menu_cat_features)
ohe_user.fit(user_cat_features)

@app.route('/predictMRS', methods=['POST'])
def predict():
    data = request.json
    
    user_input = data['user_features']
    
    user_cat_features = pd.DataFrame([user_input])
    
    user_features_encoded = ohe_user.transform(user_cat_features).toarray()
    
    # Ensure the user features are repeated for each menu item
    user_features_repeated = np.tile(user_features_encoded, (len(menu_df), 1))
    
    menu_features_encoded = ohe_menu.transform(menu_cat_features).toarray()
    
    # Make predictions for each unique menu item with the same user features
    predictions = model.predict([menu_features_encoded, user_features_repeated])
    
    # Combine the predictions with the menu items
    menu_df['compatibility_score'] = predictions
    
    top_n = 10
    top_recommendations = menu_df.sort_values(by='compatibility_score', ascending=False).head(top_n)
    
    # Extract necessary fields for the response
    result = top_recommendations[['recipe_name', 'diet_labels', 'compatibility_score']].to_dict(orient='records')
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
