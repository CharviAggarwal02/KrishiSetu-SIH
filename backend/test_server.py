from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test API")

@app.get("/")
async def root():
    return {"message": "Test API is working", "status": "success"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/")
async def api_root():
    return {"message": "API is working"}

if __name__ == "__main__":
    uvicorn.run("test_server:app", host="0.0.0.0", port=8000, reload=True)
