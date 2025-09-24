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

# ------------------- Load env -------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# ------------------- MongoDB -------------------
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# ------------------- App & Router -------------------
app = FastAPI(title="HarvestGuru API")
router = APIRouter(prefix="/api")  # Add /api prefix to all routes

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

# ------------------- Additional Routes -------------------
@router.get("/states")
async def get_states():
    return {"states": ["Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Goa", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Lakshadweep", "Andaman and Nicobar Islands"]}

@router.get("/districts/{state}")
async def get_districts(state: str):
    # Sample districts for a few states
    districts_map = {
        "Andhra Pradesh": ["Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna", "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam", "Vizianagaram", "West Godavari", "YSR Kadapa"],
        "Assam": ["Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar", "Charaideo", "Chirang", "Darrang", "Dhemaji", "Dhubri", "Dibrugarh", "Dima Hasao", "Goalpara", "Golaghat", "Hailakandi", "Hojai", "Jorhat", "Kamrup", "Kamrup Metropolitan", "Karbi Anglong", "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon", "Nagaon", "Nalbari", "Sivasagar", "Sonitpur", "South Salmara-Mankachar", "Tinsukia", "Udalguri", "West Karbi Anglong"],
        "Bihar": ["Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur", "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj", "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj", "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa", "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan", "Supaul", "Vaishali", "West Champaran"],
        "Gujarat": ["Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar", "Botad", "Chhota Udaipur", "Dahod", "Dang", "Devbhoomi Dwarka", "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kheda", "Kutch", "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar", "Tapi", "Vadodara", "Valsad"],
        "Haryana": ["Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"],
        "Karnataka": ["Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar", "Chamarajanagar", "Chikballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", "Davangere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"],
        "Kerala": ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"],
        "Maharashtra": ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"],
        "Punjab": ["Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka", "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Mansa", "Moga", "Muktsar", "Pathankot", "Patiala", "Rupnagar", "Sahibzada Ajit Singh Nagar", "Sangrur", "Shahid Bhagat Singh Nagar", "Tarn Taran"],
        "Rajasthan": ["Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bharatpur", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Dholpur", "Dungarpur", "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur", "Karauli", "Kota", "Nagaur", "Pali", "Pratapgarh", "Rajsamand", "Sawai Madhopur", "Sikar", "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur"],
        "Tamil Nadu": ["Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai", "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"],
        "Uttar Pradesh": ["Agra", "Aligarh", "Allahabad", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Ayodhya", "Azamgarh", "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki", "Bareilly", "Basti", "Bhadohi", "Bijnor", "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah", "Etawah", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddha Nagar", "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", "Hapur", "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj", "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kheri", "Kushinagar", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri", "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pilibhit", "Pratapgarh", "Prayagraj", "Raebareli", "Rampur", "Saharanpur", "Sambhal", "Sant Kabir Nagar", "Shahjahanpur", "Shamli", "Shravasti", "Siddharthnagar", "Sitapur", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi"],
        "West Bengal": ["Alipurduar", "Bankura", "Birbhum", "Cooch Behar", "Dakshin Dinajpur", "Darjeeling", "Hooghly", "Howrah", "Jalpaiguri", "Jhargram", "Kalimpong", "Kolkata", "Malda", "Murshidabad", "Nadia", "North 24 Parganas", "Paschim Bardhaman", "Paschim Medinipur", "Purba Bardhaman", "Purba Medinipur", "Purulia", "South 24 Parganas", "Uttar Dinajpur"]
    }
    return {"districts": districts_map.get(state, ["District 1", "District 2", "District 3"])}

@router.get("/crops")
async def get_crops():
    return {"crops": ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", "Groundnut", "Rapeseed", "Sunflower", "Sesame", "Jute", "Tea", "Coffee", "Rubber", "Coconut", "Banana", "Mango", "Grapes", "Potato", "Onion", "Tomato", "Chilli", "Turmeric", "Ginger", "Black Pepper", "Cardamom", "Cinnamon", "Cloves", "Nutmeg", "Vanilla"]}

@router.get("/soil-types")
async def get_soil_types():
    return {"soilTypes": [
        {"name": "Alluvial", "description": "Rich in nutrients, good for most crops"},
        {"name": "Black", "description": "High clay content, good water retention"},
        {"name": "Red", "description": "Well-drained, suitable for many crops"},
        {"name": "Laterite", "description": "Low fertility, needs organic matter"},
        {"name": "Mountain", "description": "Variable composition, good for specific crops"},
        {"name": "Desert", "description": "Sandy, low organic matter"},
        {"name": "Peaty", "description": "High organic content, good for vegetables"},
        {"name": "Clay", "description": "Heavy soil, good water retention"},
        {"name": "Sandy", "description": "Well-drained, needs frequent irrigation"},
        {"name": "Loamy", "description": "Ideal soil type, balanced properties"}
    ]}

@router.get("/my-predictions")
async def get_my_predictions(current_user: User = Depends(get_current_user)):
    predictions = await db.predictions.find({"user_id": current_user.id}).to_list(100)
    return {"predictions": predictions}

@router.get("/weather/{lat}/{lon}")
async def get_weather(lat: float, lon: float):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "rainfall": data.get("rain", {}).get("1h", 0)
                }
    except Exception as e:
        logging.error(f"Weather API error: {e}")
    
    # Return mock data if API fails
    return {
        "temperature": 28.5,
        "humidity": 65,
        "description": "partly cloudy",
        "rainfall": 2.5
    }

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(chat_message: ChatMessage, current_user: User = Depends(get_current_user)):
    # Mock AI response
    responses = {
        "en": "Based on your farming conditions, I recommend regular irrigation and proper fertilization. Monitor soil moisture levels and adjust watering accordingly.",
        "hi": "आपकी खेती की स्थिति के आधार पर, मैं नियमित सिंचाई और उचित उर्वरक की सिफारिश करता हूं। मिट्टी की नमी के स्तर की निगरानी करें और पानी को तदनुसार समायोजित करें।",
        "bn": "আপনার চাষের অবস্থার ভিত্তিতে, আমি নিয়মিত সেচ এবং সঠিক সার প্রয়োগের পরামর্শ দিই। মাটির আর্দ্রতার মাত্রা পর্যবেক্ষণ করুন এবং জল দেওয়া তদনুসারে সামঞ্জস্য করুন।",
        "or": "ଆପଣଙ୍କ କୃଷି ଅବସ୍ଥା ଅନୁଯାୟୀ, ମୁଁ ନିୟମିତ ଜଳସେଚନ ଏବଂ ଉପଯୁକ୍ତ ସାର ବ୍ୟବହାରର ସୁପାରିଶ କରୁଛି। ମାଟିର ଆର୍ଦ୍ରତା ସ୍ତର ନିରୀକ୍ଷଣ କରନ୍ତୁ ଏବଂ ଜଳ ଦେବା ତଦନୁସାରେ ସମନ୍ୱୟ କରନ୍ତୁ।"
    }
    
    recommendations = [
        "Monitor soil moisture regularly",
        "Apply organic fertilizers",
        "Check for pest infestations",
        "Maintain proper irrigation schedule"
    ]
    
    return ChatResponse(
        response=responses.get(chat_message.language, responses["en"]),
        language=chat_message.language,
        recommendations=recommendations
    )

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