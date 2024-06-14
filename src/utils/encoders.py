from sklearn.preprocessing import OneHotEncoder

def create_one_hot_encoder():
    return OneHotEncoder(handle_unknown='ignore')
