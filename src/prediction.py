import pickle
import numpy as np

class PetrolPricePredictor:

    def __init__(self):

        with open("artifacts/model.pkl","rb") as f:
            self.model = pickle.load(f)

    def predict(self,features):

        features = np.array(features).reshape(1,-1)

        prediction = self.model.predict(features)

        return prediction[0]