from fastapi import FastAPI
from src.api import query
from fastapi.responses import RedirectResponse
from src.utils import data_store

# Load data at startup
data_store.load_data()

# Create FastAPI app
app = FastAPI(
    title="ML API",
    description="API for ML Model Inference",
    version="1.0.0",
)

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Include the API router
app.include_router(query.router)