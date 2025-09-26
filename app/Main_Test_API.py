from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from Main_Recommender_System import load_model_artefacts, get_recommendations_pipeline

app = FastAPI(
    title="Symptom Recommender API",
    description="An API that recommends symptoms based on user input.",
    version="1.0.0"
)

MODEL_ARTEFACTS = load_model_artefacts()

class RecommendRequest(BaseModel):
    search_term: str
    age: Optional[int] = None
    gender: Optional[str] = None
    top_n_next_symptoms: Optional[int] = 5
    
@app.post("/recommend")
def recommend_symptoms_api(request: RecommendRequest):
    recommendations = get_recommendations_pipeline(
        request.search_term,
        MODEL_ARTEFACTS,
        age=request.age,
        gender=request.gender,
        top_n=request.top_n_next_symptoms
    )
    return recommendations
