from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np


api=FastAPI()

diabetes = load_diabetes()
model = LinearRegression()
model.fit(diabetes.data, diabetes.target)


class data(BaseModel):
    age: float = Field(..., description="Normalized age")
    sex: float = Field(..., description="Normalized sex")
    bmi: float = Field(..., description="Body mass index")
    bp: float = Field(..., description="Average blood pressure")
    s1: float = Field(..., description="TC (total serum cholesterol)")
    s2: float = Field(..., description="LDL (low-density lipoproteins)")
    s3: float = Field(..., description="HDL (high-density lipoproteins)")
    s4: float = Field(..., description="TCH (thyroid stimulating hormone)")
    s5: float = Field(..., description="LTG (lamotrigine)")
    s6: float = Field(..., description="Glucose level")
    
class return_method(BaseModel):
    disease_progression: float
        

@api.get("/")
def welcome():
    return {"message": "Welcome to the Diabetes Prediction API"}

@api.post("/predict",response_model=return_method)
def predict(data :data ):
    data=[[data.age, data.sex,data.bmi, data.bp, data.s1, data.s2, data.s3, data.s4, data.s5, data.s6]]
    
    prediction = model.predict(data)[0]
    return {"disease_progression": round(prediction, 2)}
    