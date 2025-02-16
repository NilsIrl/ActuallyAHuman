import React, { useState, useEffect, useRef } from "react";
import {
  GoogleMap,
  DirectionsRenderer,
  Marker,
  LoadScript,
} from "@react-google-maps/api";
import { getLocation } from "./processing/open_ai_processing";

const containerStyle = {
  width: "100%",
  height: "100%",
  minHeight: "400px",
};

const center = {
  lat: 37.7749,
  lng: -122.4194,
};

interface LatLng {
  lat: number;
  lng: number;
}

const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY!;

const MapView: React.FC = () => {
  const [origin, setOrigin] = useState<LatLng | null>(null);
  const [destination, setDestination] = useState<LatLng | null>(null);
  const [directions, setDirections] =
    useState<google.maps.DirectionsResult | null>(null);
  const [currentLocation, setCurrentLocation] = useState<LatLng | null>(null);
  const [gpsData, setGpsData] = useState<LatLng[]>([]);
  const mapRef = useRef<google.maps.Map | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [locationError, setLocationError] = useState<string | null>(null);

  useEffect(() => {
    let reconnectAttempt = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 1000; // 1 second

    const connectWebSocket = () => {
      console.log("Attempting to connect to WebSocket...");
      wsRef.current = new WebSocket("ws://localhost:8000/ws");

      wsRef.current.onopen = () => {
        console.log("WebSocket connection established");
        reconnectAttempt = 0; // Reset reconnect attempts on successful connection
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // Skip ping messages
        if (data.type === "ping") {
          return;
        }

        // Only process messages with GPS coordinates
        if (data.latitude !== undefined && data.longitude !== undefined) {
          setGpsData((prevData) => [
            ...prevData,
            { lat: data.latitude, lng: data.longitude },
          ]);
          setCurrentLocation({ lat: data.latitude, lng: data.longitude });
          if (mapRef.current && currentLocation) {
            mapRef.current.panTo(currentLocation);
          }
        }
      };

      wsRef.current.onclose = (event) => {
        console.log("WebSocket closed:", event);
        if (reconnectAttempt < maxReconnectAttempts) {
          console.log(
            `Attempting to reconnect... (${
              reconnectAttempt + 1
            }/${maxReconnectAttempts})`
          );
          setTimeout(connectWebSocket, reconnectDelay);
          reconnectAttempt++;
        }
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    };

    connectWebSocket();

    // Cleanup function
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []); // Empty dependency array since we want this to run once

  useEffect(() => {
    const getInitialLocation = async () => {
      try {
        setLocationError(null);
        const position = await getLocation();
        console.log("Found position:", position);
        if (!position) {
          throw new Error("Location not found");
        }
        const userLocation = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        setCurrentLocation(userLocation);

        if (mapRef.current) {
          mapRef.current.panTo(userLocation);
          mapRef.current.setZoom(15);
        }
      } catch (error) {
        console.error("Error getting initial location:", error);
        if (error instanceof Error) {
          setLocationError(error.message);
        } else {
          setLocationError(
            "Unable to get your location. Please check your browser settings."
          );
        }
      }
    };

    getInitialLocation();
  }, []);

  useEffect(() => {
    if (origin && destination) {
      const service = new window.google.maps.DirectionsService();
      service.route(
        {
          origin,
          destination,
          travelMode: window.google.maps.TravelMode.DRIVING,
        },
        (result, status) => {
          if (status === "OK") {
            setDirections(result);
          } else {
            console.error(`Error fetching directions: ${status}`);
          }
        }
      );
    }
  }, [origin, destination]);

  const onLoad = (map: google.maps.Map) => {
    mapRef.current = map;
    // If we already have the current location, pan to it
    if (currentLocation) {
      map.panTo(currentLocation);
      map.setZoom(15);
    }
  };

  const onUnmount = () => {
    mapRef.current = null;
  };

  return (
    <div className="w-full h-full relative">
      <LoadScript googleMapsApiKey={API_KEY}>
        <GoogleMap
          mapContainerStyle={containerStyle}
          center={currentLocation || center}
          zoom={14}
          onLoad={onLoad}
          onUnmount={onUnmount}
        >
          {directions && <DirectionsRenderer directions={directions} />}

          {gpsData.map((point, index) => (
            <Marker
              key={index}
              position={point}
              icon={{
                path: window.google.maps.SymbolPath.CIRCLE,
                scale: 4,
                fillColor: "blue",
                fillOpacity: 0.8,
                strokeColor: "blue",
                strokeOpacity: 0.8,
              }}
            />
          ))}

          {currentLocation && (
            <Marker
              position={currentLocation}
              icon={{
                path: google.maps.SymbolPath.CIRCLE,
                scale: 8,
                fillColor: "#4285F4",
                fillOpacity: 1,
                strokeColor: "#ffffff",
                strokeWeight: 2,
              }}
              title="Your Location"
            />
          )}
        </GoogleMap>
      </LoadScript>

      {locationError && (
        <div className="absolute top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p className="font-bold">Location Error</p>
          <p>{locationError}</p>
        </div>
      )}

      {/* <div className="absolute top-4 left-4 bg-white p-4 rounded shadow">
        {!currentLocation && !locationError && (
          <button
            onClick={() => getCurrentLocation()}
            className="mt-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Get My Location
          </button>
        )}
      </div> */}
    </div>
  );
};

export default MapView;
