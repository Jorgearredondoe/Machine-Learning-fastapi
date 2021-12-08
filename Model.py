import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path


class Model:
    def __init__(self, model_path: str = None):
        self.model = None
        self._model_path = model_path
        self.load()


    # Entrenamiento del modelo
    def train(self, x:np.ndarray, y:np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self._model = RandomForestRegressor()
        self._model.fit(X_train, y_train)

        return self

    # Predicción
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    # Guardar modelo
    def save(self):
        if self._model is not None:
            joblib.dump(self._model, self._model_path)
        else:
            raise TypeError("The model is no trained yet, use .train() before saving")

    # Cargar modelo
    def load(self):
        try:
            self._model = joblib.load(self._model_path)
            print('Funcionó: ',self._model)
        except:
            self._model = None
        return self 


model_path = Path(__file__).parent / "model.joblib"
diabetes = load_diabetes(return_X_y=True, as_frame=True)
diabetes = pd.concat([diabetes[0], diabetes[1]], axis=1)
diabetes.drop(columns=['age','sex','s1','s2','s3','s6'],inplace=True)
n_features = diabetes.iloc[:, :-1] 
n_features = np.array(n_features).shape[1]
model = Model(model_path)


def get_model():
    return model


if __name__ == '__main__':
    x = diabetes.iloc[:, :-1] 
    y = diabetes.iloc[:, -1:]['target'] 
    model.train(x,y)
    model.save()