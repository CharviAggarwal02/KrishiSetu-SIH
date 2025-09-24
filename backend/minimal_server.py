from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Minimal API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRegister(BaseModel):
    email: str
    password: str
    name: str
    phone: str = None

@app.get("/")
async def root():
    return {"message": "Minimal API is working"}

@app.get("/api/")
async def api_root():
    return {"message": "API is working"}

@app.post("/api/")
async def api_post():
    return {"message": "POST to API is working"}

@app.post("/api/auth/register")
async def register(user_data: UserRegister):
    return {"access_token": "test_token", "token_type": "bearer"}

if __name__ == "__main__":
    uvicorn.run("minimal_server:app", host="0.0.0.0", port=8000, reload=True)
