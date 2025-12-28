"""
ML Service for Activity-Field Prediction API
FastAPI endpoint for model inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import sys
from pathlib import Path

# Add ml directory to path (parent directory now)
ml_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ml_dir))

from inference import ActivityFieldPredictor, load_config
from hybrid_recommender import HybridWorkerRecommender


# Request/Response models
class PredictionRequest(BaseModel):
    activity: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "activity": "Design 4-wheel rover chassis"
            }
        }


class PredictionResponse(BaseModel):
    activity: str
    field: str
    reason: str
    confidence: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "activity": "Design 4-wheel rover chassis",
                "field": "robotics",
                "reason": "autonomous mechanisms",
                "confidence": 0.95
            }
        }


class WorkerQuery(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "High voltage electrician with NFPA certification",
                "top_k": 5,
                "filters": {
                    "min_experience": 10,
                    "min_safety": 4.0,
                    "only_available": True
                }
            }
        }


class WorkerRecommendation(BaseModel):
    id: int
    name: str
    role: str
    experience: int
    safety_rating: float
    availability: str
    skills: str
    similarity_score: float
    suitability_score: float
    reasoning: str


# Initialize FastAPI app
app = FastAPI(
    title="Activity-Field Prediction API",
    description="API for predicting engineering fields from activity descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instances (loaded once)
predictor: Optional[ActivityFieldPredictor] = None
worker_recommender: Optional[HybridWorkerRecommender] = None


@app.on_event("startup")
async def load_model():
    """Load models on startup"""
    global predictor, worker_recommender
    
    # Change to ml directory for file access
    import os
    original_dir = os.getcwd()
    os.chdir(ml_dir)
    
    # Load activity-field predictor
    try:
        print("🚀 Loading Activity-Field ML model...")
        config = load_config("config.yaml")
        model_path = config['model']['model_save_path']
        predictor = ActivityFieldPredictor(model_path, config)
        print("✅ Activity-Field model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading activity-field model: {e}")
        predictor = None
    
    # Load worker recommender
    try:
        print("🚀 Loading Worker Recommender...")
        worker_recommender = HybridWorkerRecommender(use_slm=False)
        print("✅ Worker Recommender loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading worker recommender: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("⚠️  Worker recommendations will fail. Run: python prepare_worker_data.py && python train_decision_tree.py")
        worker_recommender = None
    
    # Change back to original directory
    os.chdir(original_dir)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Activity-Field Prediction API",
        "model_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "model_ready": predictor is not None,
        "message": "✅ Ready for predictions" if predictor else "⚠️ Model not loaded. Please train the model first."
    }


@app.post("/predict-field", response_model=PredictionResponse)
async def predict_field(request: PredictionRequest):
    """
    Predict the best field for a given activity
    
    Args:
        request: PredictionRequest containing the activity description
        
    Returns:
        PredictionResponse with field, reason, and confidence
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using: cd ml && python train_model.py"
        )
    
    try:
        # Get prediction
        result = predictor.predict(request.activity)
        
        # Return formatted response
        return PredictionResponse(
            activity=result['activity'],
            field=result['field'],
            reason=result['reason'],
            confidence=0.85  # Placeholder - implement actual confidence scoring if needed
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict")
async def batch_predict(activities: list[str]):
    """Predict fields for multiple activities"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        results = predictor.batch_predict(activities)
        return {
            "predictions": [
                {
                    "activity": r['activity'],
                    "field": r['field'],
                    "reason": r['reason']
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/recommend-workers")
async def recommend_workers(request: WorkerQuery):
    """
    Recommend workers based on job requirements
    
    Args:
        request: WorkerQuery with query string, top_k, and optional filters
        
    Returns:
        List of worker recommendations with scores and reasoning
    """
    if worker_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Worker recommender not loaded. Please run data preparation and train decision tree."
        )
    
    try:
        # Get recommendations
        recommendations = worker_recommender.recommend(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        return {
            "query": request.query,
            "total_results": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Worker recommendation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Activity-Field Prediction API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
