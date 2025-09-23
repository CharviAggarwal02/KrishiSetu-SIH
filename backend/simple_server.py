from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import uuid
from datetime import datetime, timezone

app = FastAPI(title="HarvestGuru API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo purposes
fake_users_db = {}

# Pydantic models
class UserRegister(BaseModel):
    email: str
    password: str
    name: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    phone: Optional[str] = None
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

@app.get("/")
async def root():
    return {"message": "HarvestGuru API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Authentication endpoints
@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    if user_data.email in fake_users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_id = str(uuid.uuid4())
    user = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "phone": user_data.phone,
        "password": user_data.password,  # In real app, hash this
        "created_at": datetime.now(timezone.utc)
    }
    
    fake_users_db[user_data.email] = user
    
    # Generate a simple token (in real app, use JWT)
    token = f"demo_token_{user_id}"
    
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    user = fake_users_db.get(user_credentials.email)
    if not user or user["password"] != user_credentials.password:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Generate a simple token (in real app, use JWT)
    token = f"demo_token_{user['id']}"
    
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
async def get_me():
    # For demo purposes, return a mock user
    # In real app, extract user from JWT token
    return {
        "id": "demo_user_id",
        "email": "demo@gmail.com",
        "name": "Demo User",
        "phone": "1234567890",
        "created_at": datetime.now(timezone.utc)
    }

# Mock endpoints for the frontend
@app.get("/api/states")
async def get_states():
    return {
        "states": [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
            "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
            "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
            "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
        ]
    }

@app.get("/api/districts/{state}")
async def get_districts(state: str):
    # Mock districts for demo
    districts = {
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
        "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"]
    }
    return {"districts": districts.get(state, ["District 1", "District 2", "District 3"])}

@app.get("/api/crops")
async def get_crops():
    return {
        "crops": [
            "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", "Groundnut",
            "Rapeseed", "Sunflower", "Potato", "Onion", "Tomato", "Chili", "Turmeric"
        ]
    }

@app.get("/api/soil-types")
async def get_soil_types():
    return {
        "soilTypes": [
            {"name": "Alluvial", "description": "Rich in nutrients, good for most crops"},
            {"name": "Black", "description": "High clay content, good water retention"},
            {"name": "Red", "description": "Well-drained, suitable for many crops"},
            {"name": "Laterite", "description": "Low fertility, needs improvement"},
            {"name": "Mountain", "description": "Variable composition, location dependent"}
        ]
    }

@app.post("/api/predict-yield")
async def predict_yield(prediction_request: dict):
    # Mock prediction response
    return {
        "id": str(uuid.uuid4()),
        "predicted_yield": 18.5,
        "yield_unit": "quintals per hectare",
        "district_average": 16.2,
        "comparison_percentage": 14.2,
        "recommendations": [
            "Follow standard farming practices",
            "Check weather updates regularly",
            "Maintain proper irrigation schedule",
            "Monitor soil health regularly"
        ],
        "confidence_score": 85.5,
        "created_at": datetime.now(timezone.utc)
    }

@app.get("/api/my-predictions")
async def get_my_predictions():
    # Mock predictions for demo
    return {
        "predictions": [
            {
                "id": str(uuid.uuid4()),
                "predicted_yield": 18.5,
                "yield_unit": "quintals per hectare",
                "comparison_percentage": 14.2,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_data": {
                    "crop_info": {"crop_name": "Rice"}
                }
            }
        ]
    }

@app.get("/api/weather/{lat}/{lon}")
async def get_weather(lat: float, lon: float):
    # Mock weather data
    return {
        "temperature": 28.5,
        "humidity": 65,
        "description": "partly cloudy",
        "rainfall": 2.5
    }

@app.post("/api/chat")
async def chat_with_assistant(chat_request: dict):
    # Mock chat response
    return {
        "response": "Thank you for your question! This is a demo response. In a real application, this would be powered by AI to provide farming advice.",
        "language": chat_request.get("language", "en"),
        "recommendations": [
            "Consult with local agricultural experts",
            "Check weather forecasts regularly",
            "Follow sustainable farming practices"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("simple_server:app", host="127.0.0.1", port=8000, reload=True)
