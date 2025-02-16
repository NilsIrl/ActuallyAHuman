import sqlite3
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
from microservices_utils import get_latest_gps_coordinates, get_latest_imu_data
import random  # Simulating GPS data (replace with real data)

class Order(BaseModel):
    order: str
    
class Waypoints(BaseModel):
    waypoints: list[tuple[float, float]]

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost",
    "ws://localhost:5173",
    "ws://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GlobalState:
    def __init__(self):
        self.current_waypoints: List[tuple[float, float]] = []

# Create a global instance
global_state = GlobalState()

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("New WebSocket connection attempting to connect...")
    try:
        await websocket.accept()
        print("WebSocket connection established successfully")
        
        while True:
            try:
                coordinates = get_latest_gps_coordinates()
                
                if coordinates:
                    latitude, longitude = coordinates
                    gps_data = {"latitude": latitude, "longitude": longitude}
                    print(f"Sending GPS data: {gps_data}")
                    await websocket.send_json(gps_data)
                else:
                    # If no GPS data found, check waypoints
                    print("Error: No GPS data or waypoints available")

                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"Error in WebSocket loop: {e}")
                break
                
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
    finally:
        print("WebSocket connection closed")
        try:
            await websocket.close()
        except:
            pass


@app.post("/add_order")
async def add_order(order: Order):
    print(f"Received order: {order.order}")
    return {"message": "Order added successfully"}

async def process_waypoints(waypoints: List[tuple[float, float]]):
    # Your long-running code here
    print("Starting waypoint processing...")
    initial_position = get_latest_gps_coordinates()
    print(f"Initial position: {initial_position}")
    print(waypoints)
    await asyncio.sleep(600)  # Simulating 10-minute process
    print("Waypoint processing complete")

@app.post("/send_waypoints")
async def send_waypoints(waypoints: Waypoints, background_tasks: BackgroundTasks):
    print(f"Received waypoints: {waypoints}")
    global_state.current_waypoints = waypoints.waypoints
    
    # Schedule the long-running task to run in the background
    background_tasks.add_task(process_waypoints, waypoints.waypoints)
    
    return {"status": 200, "message": "Waypoints received successfully. Processing started."}
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
