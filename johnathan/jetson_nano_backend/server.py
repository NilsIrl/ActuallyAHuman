import sqlite3
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
from microservices_utils import get_latest_gps_coordinates, get_latest_imu_data, move_robot_forward_time, rotate_robot
import random  # Simulating GPS data (replace with real data)
import math
from roboclaw import Roboclaw

rc = Roboclaw("/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0", 115200)
rc.Open()
address = 0x80
version = rc.ReadVersion(address)
if not version[0]:
    print("GETVERSION Failed")
else:
    print(repr(version[1]))

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
    for waypoint in waypoints:
        latitude, longitude = get_latest_gps_coordinates()
        # Calculate heading from current position to waypoint
        lat2, lon2 = waypoint  # Target waypoint
        
        lat1_rad = math.radians(latitude)
        lon1_rad = math.radians(longitude)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        delta_lon = lon2_rad - lon1_rad
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        initial_bearing = math.atan2(x, y)
        bearing_degrees = math.degrees(initial_bearing)
        heading = (bearing_degrees + 360) % 360

        print(f"Heading to waypoint: {heading:.2f} degrees")
        current_imu = get_latest_imu_data() % 360
        print(f"Current heading: {current_imu:.2f} degrees")

        # Compute the minimal rotation needed (-180° to 180° range)
        rotation_degrees = ((heading - current_imu + 180) % 360) - 180
        print(f"Initiating rotation from {current_imu:.2f}° to {heading:.2f}° (rotation: {rotation_degrees:.2f}°)")
        
        # Rotate the robot using the rotate_robot function
        if rotate_robot(rc, rotation_degrees):
            print("Rotation successful.")
        else:
            print("Rotation not completed within timeout.")

        R = 6371000  # Earth's radius in meters
        # Calculate differences in radians
        dlat = math.radians(lat2 - latitude)
        dlon = math.radians(lon2 - longitude)
        # Compute the Haversine distance
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(latitude)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_meters = R * c
        print(f"Calculated distance to waypoint: {distance_meters:.2f} meters")
        # Move the robot forward by the calculated distance
        move_robot_forward_time(rc, distance_meters)
    

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
