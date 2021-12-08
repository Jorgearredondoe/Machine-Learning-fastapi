from fastapi import FastAPI
from fastapi import Depends
import numpy as np
import uvicorn
from pydantic import BaseModel
from pydantic import validator
from typing import List
from Model import Model
from Model import get_model
from Model import n_features


class PredictRequest(BaseModel):
    data: List[List[float]]
    

    @validator("data")
    def check_dimensionality(cls, v):
        for point in v:
            if len(point) != n_features:
                raise ValueError(f"Each data point must contain {n_features} features")

        return v
        
class PredictResponse(BaseModel):
    data: List[float]


app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest, model: Model = Depends(get_model)):
    X = np.array(input.data)
    print('Este es el modelo,',model)

    y_pred = model.predict(X)
    result = PredictResponse(data=y_pred.tolist())

    return result



if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)