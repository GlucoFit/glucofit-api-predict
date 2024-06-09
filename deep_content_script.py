# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
import os
import matplotlib.pyplot as plt
os.listdir('/kaggle/input/')

# %%
menu_df = pd.read_csv("./csv/menu.csv")
menu_df

# %%
user_df = pd.read_csv("./csv/user.csv")
user_df

# %%
user_df = pd.concat([user_df]*int(np.ceil(len(menu_df)/len(user_df))), ignore_index=True)
user_df = user_df.iloc[:len(menu_df)]  # Trim to match the length of menu_df

# %% [markdown]
# menu_features = menu_df[['calories', 'sugar_content(g)','diet_labels','recipe_name']]
# user_features = user_df[['calorie_intake (kcal/day)', 'sugar_intake (g/day)','diet_labels','preferred_food']]

# %%
menu_cat_features = menu_df[['diet_labels', 'recipe_name']]
user_cat_features = user_df[['diet_labels', 'preferred_food']]

# %%


# %%


# %%
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
menu_cat_encoded = ohe.fit_transform(menu_cat_features)
user_cat_encoded = ohe.fit_transform(user_cat_features)

# %%
menu_features_encoded = np.hstack([menu_cat_encoded]).astype(np.float32)
user_features_encoded = np.hstack([user_cat_encoded]).astype(np.float32)

# %%
X_menu_train, X_menu_test, X_user_train, X_user_test = train_test_split(
    menu_features_encoded, user_features_encoded, test_size=0.2, random_state=32)

# %%
y = np.random.randint(0, 2, len(X_menu_train) + len(X_menu_test)).astype(np.float32)
y_train = y[:len(X_menu_train)]
y_test = y[len(X_menu_train):]

# %%
input_menu = Input(shape=(menu_features_encoded.shape[1],), name='menu_input')
input_user = Input(shape=(user_features_encoded.shape[1],), name='user_input')

# %%
# Menu branch
menu_layer = Dense(128, activation='relu')(input_menu)
menu_layer = Dropout(0.5)(menu_layer)
menu_layer = BatchNormalization()(menu_layer)
menu_layer = Dense(256, activation='relu')(input_menu)
menu_layer = Dropout(0.5)(menu_layer)
menu_layer = BatchNormalization()(menu_layer)

# User branch
user_layer = Dense(128, activation='relu')(input_user)
user_layer = Dropout(0.5)(user_layer)
user_layer = BatchNormalization()(user_layer)
user_layer = Dense(256, activation='relu')(input_user)
user_layer = Dropout(0.5)(user_layer)
user_layer = BatchNormalization()(user_layer)

# Concatenate the outputs of the two branches
concatenated = Concatenate()([menu_layer, user_layer])

# Add more layers
x = Dense(256, activation='relu')(concatenated)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
output = Dense(1, activation='sigmoid')(x)  # Assuming binary classification
rms = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

# Build and compile the model
model = Model(inputs=[input_menu, input_user], outputs=output)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit([X_menu_train, X_user_train], y_train, epochs=100, batch_size=32, validation_split=0.2)

# %%
loss, accuracy = model.evaluate([X_menu_test, X_user_test], y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# %%
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Example new user input
new_user_input = {
    'diet_labels': 'Low-Sugar',
    'preferred_food': 'ayam geprek'
}

# %%
new_user_cat_features = pd.DataFrame([{
    'diet_labels': new_user_input['diet_labels'],
    'preferred_food': new_user_input['preferred_food']
}])

new_user_cat_encoded = ohe.transform(new_user_cat_features)

new_user_features_encoded = np.hstack([new_user_cat_encoded]).astype(np.float32)

new_user_features_encoded = np.tile(new_user_features_encoded, (len(menu_features_encoded), 1))

# %%
predictions = model.predict([menu_features_encoded, new_user_features_encoded])

# %%
menu_df['compatibility_score'] = predictions

# %%
top_n = 10
top_recommendations = menu_df.sort_values(by='compatibility_score', ascending=False).head(top_n)

print("Top Recommendations:")
print(top_recommendations[['recipe_name', 'diet_labels', 'compatibility_score']])

# %%
# Example of making predictions
predictions = model.predict([X_menu_test, X_user_test])
print(predictions)

# %%
# Save the model if needed
model.save('Deep_content_based_model.h5')

# %%
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# %%



