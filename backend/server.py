from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import uuid
from datetime import datetime, timezone, timedelta
import numpy as np
from routes.auth import router as auth_router 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from passlib.context import CryptContext
import jwt
import httpx
import logging
import uvicorn


app=FastAPI()
app.include_router(auth_router, prefix="/api/auth")

# ------------------- Load env -------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# ------------------- MongoDB -------------------
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# ------------------- App & Router -------------------
app = FastAPI(title="HarvestGuru API")
router = APIRouter()  # No prefix

# ------------------- Security -------------------
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get("JWT_SECRET", "harvestguru-jwt-secret-key-2025")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

WEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "7e78b79404aaf3dc70824def3c556b49")

# ------------------- Models -------------------
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    phone: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class FarmDetails(BaseModel):
    state: str
    district: str
    village: str
    pincode: str
    farm_size: float
    farm_size_unit: str

class CropInfo(BaseModel):
    crop_name: str
    variety: str
    sowing_date: str
    season: str

class SoilInputs(BaseModel):
    soil_type: str
    fertilizer_used: str
    ph_level: Optional[float] = None
    organic_carbon: Optional[float] = None

class IrrigationInfo(BaseModel):
    irrigation_source: str
    irrigation_frequency: str
    water_availability: str

class CropPredictionRequest(BaseModel):
    user_id: str
    farm_details: FarmDetails
    crop_info: CropInfo
    soil_inputs: SoilInputs
    irrigation_info: IrrigationInfo

class CropPredictionResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    predicted_yield: float
    yield_unit: str
    district_average: float
    comparison_percentage: float
    recommendations: List[str]
    confidence_score: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessage(BaseModel):
    message: str
    language: str = "en"

class ChatResponse(BaseModel):
    response: str
    language: str
    recommendations: Optional[List[str]] = None

# ------------------- Auth functions -------------------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if not user_email:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = await db.users.find_one({"email": user_email})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return User(**user)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# ------------------- Sample ML model -------------------
def get_sample_agricultural_data():
    np.random.seed(42)
    n = 1000
    data = {
        'farm_size': np.random.uniform(0.5, 10, n),
        'rainfall': np.random.uniform(400, 1200, n),
        'temperature': np.random.uniform(20, 35, n),
        'humidity': np.random.uniform(40, 90, n),
        'soil_ph': np.random.uniform(5.5, 8.5, n),
        'fertilizer_amount': np.random.uniform(50, 200, n),
        'irrigation_frequency': np.random.randint(1,4,n)
    }
    yield_base = (data['farm_size']*0.5 + data['rainfall']*0.01 + (35-abs(data['temperature']-27))*0.3 +
                  data['humidity']*0.05 + (7-abs(data['soil_ph']-7))*2 + data['fertilizer_amount']*0.02 +
                  data['irrigation_frequency']*2)
    data['yield'] = np.maximum(yield_base + np.random.normal(0,2,n), 5)
    return pd.DataFrame(data)

df = get_sample_agricultural_data()
X = df.drop('yield', axis=1)
y = df['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------- Router & Routes -------------------
@router.get("/")
async def root():
    return {"message": "HarvestGuru API is running"}

@router.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    if await db.users.find_one({"email": user_data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = get_password_hash(user_data.password)
    user = User(email=user_data.email, name=user_data.name, phone=user_data.phone)
    user_dict = user.dict()
    user_dict["password_hash"] = hashed
    await db.users.insert_one(user_dict)
    token = create_access_token({"sub": user.email}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@router.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    user = await db.users.find_one({"email": user_credentials.email})
    if not user or not verify_password(user_credentials.password, user.get("password_hash","")):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": user["email"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ------------------- Crop Prediction -------------------
@router.post("/predict-yield", response_model=CropPredictionResponse)
async def predict_yield(prediction_request: CropPredictionRequest, current_user: User = Depends(get_current_user)):
    # Mock weather
    weather_data = {"temperature":28.5, "humidity":65, "rainfall":200}
    features = np.array([[prediction_request.farm_details.farm_size,
                          weather_data["rainfall"]*10,
                          weather_data["temperature"],
                          weather_data["humidity"],
                          prediction_request.soil_inputs.ph_level or 7.0,
                          100,  # default fertilizer
                          {"Rarely":1,"Sometimes":2,"Regularly":3}.get(prediction_request.irrigation_info.irrigation_frequency,2)
                         ]])
    predicted_yield = model.predict(features)[0]
    confidence = model.score(X_test, y_test)
    district_avg = np.random.uniform(12,18)
    comparison = ((predicted_yield - district_avg)/district_avg)*100
    recommendations = ["Follow standard farming practices","Check weather updates regularly"]
    response = CropPredictionResponse(
        predicted_yield=round(predicted_yield,2),
        yield_unit="quintals per hectare",
        district_average=round(district_avg,2),
        comparison_percentage=round(comparison,1),
        recommendations=recommendations,
        confidence_score=round(confidence*100,1)
    )
    prediction_dict = response.dict()
    prediction_dict["user_id"] = current_user.id
    prediction_dict["input_data"] = prediction_request.dict()
    await db.predictions.insert_one(prediction_dict)
    return response

# ------------------- Include router -------------------
app.include_router(router)

# ------------------- CORS -------------------
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run("server:app", host=host, port=port, reload=True, log_level="info")