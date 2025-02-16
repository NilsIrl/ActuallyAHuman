from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os
import logging
import googlemaps
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS - more permissive configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=False,  # Must be False if allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))
# Pydantic models for request/response validation
class LocationQuery(BaseModel):
    query: str

class OrderRequest(BaseModel):
  
    order: str

class Location(BaseModel):
    longitude: float
    latitude: float

class OrderItem(BaseModel):
    name: str
    quantity: int

class OrderResponse(BaseModel):
    start: Location
    end: Location
    order: list[OrderItem]

@app.post("/api/search-location")
async def search_location(query: LocationQuery):
    """Search for a location and return its coordinates"""
    try:
        result = gmaps.places(query.query)
        if result['status'] == 'OK' and result['results']:
            location = result['results'][0]['geometry']['location']
            return {
                'success': True,
                'location': {
                    'latitude': location['lat'],
                    'longitude': location['lng']
                }
            }
        return {'success': False, 'error': 'Location not found'}
    except Exception as e:
        logger.error(f"Search location error: {str(e)}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 