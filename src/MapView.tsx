import React, { useState, useEffect, useRef, useCallback } from "react";
import { GoogleMap, DirectionsRenderer, Marker } from "@react-google-maps/api";
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

  const [currentPoints, setCurrentPoints] = useState<LatLng[]>([]);

  // Separate WebSocket effect from map initialization

  const checkPointValidity = (point: LatLng) => {
    return (
      point.lat !== undefined &&
      point.lng !== undefined &&
      point.lat !== null &&
      point.lng !== null &&
      !isNaN(point.lat) &&
      !isNaN(point.lng) &&
      point.lat !== Infinity &&
      point.lng !== Infinity &&
      point.lat !== -Infinity &&
      point.lng !== -Infinity &&
      point.lat < 90 &&
      point.lat > -90 &&
      point.lng < 180 &&
      point.lng > -180
    );
  };

  useEffect(() => {
    const reconnectDelay = [1000, 2000, 4000, 8000, 16000]; // Exponential backoff
    let reconnectAttempt = 0;

    const connectWebSocket = () => {
      if (reconnectAttempt >= reconnectDelay.length) {
        console.log("Max reconnection attempts reached");
        return;
      }

      console.log("Attempting to connect to WebSocket...");
      wsRef.current = new WebSocket("ws://localhost:8000/ws");

      wsRef.current.onopen = () => {
        console.log("WebSocket connection established");
        reconnectAttempt = 0; // Reset on successful connection
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "ping") return;

        // ensure valid longitude and latitude

        if (checkPointValidity({ lat: data.latitude, lng: data.longitude })) {
          setGpsData((prevData) => {
            const newData = [
              ...prevData,
              { lat: data.latitude, lng: data.longitude },
            ];
            return newData.slice(-50);
          });
          
          setCurrentLocation({ lat: data.latitude, lng: data.longitude });
        } else {
          console.log("Invalid point:", data);
        }
      };

      wsRef.current.onclose = () => {
        console.log(
          `WebSocket closed. Reconnecting in ${reconnectDelay[reconnectAttempt]}ms...`
        );
        setTimeout(connectWebSocket, reconnectDelay[reconnectAttempt]);
        reconnectAttempt++;
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Separate effect for handling map updates when location changes
  useEffect(() => {
    if (!mapRef.current || !currentLocation) return;
    mapRef.current.panTo(currentLocation);
  }, [currentLocation]);

  // Initial location setup
  useEffect(() => {
    const getInitialLocation = async () => {
      try {
        setLocationError(null);
        const position = await getLocation();

        const userLocation = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        setCurrentLocation(userLocation);

        // Only update map if it's mounted
        if (mapRef.current) {
          mapRef.current.panTo(userLocation);
          mapRef.current.setZoom(15);
        }
      } catch (error) {
        console.error("Error getting initial location:", error);
        setLocationError(
          error instanceof Error ? error.message : "Unable to get your location"
        );
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

  const onLoad = useCallback(
    (map: google.maps.Map) => {
      mapRef.current = map;
      if (currentLocation) {
        map.panTo(currentLocation);
        map.setZoom(15);
      }
    },
    [currentLocation]
  );

  const onUnmount = () => {
    mapRef.current = null;
  };

  return (
    <div className="w-full h-full relative">
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

      {locationError && (
        <div className="absolute top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p className="font-bold">Location Error</p>
          <p>{locationError}</p>
        </div>
      )}

      <div className="absolute top-4 left-4 bg-white p-4 rounded shadow">
        {gpsData.map((point, index) => (
          <div key={index}>
            <p>Point {index + 1}</p>
            <p>Latitude: {point.lat}</p>
            <p>Longitude: {point.lng}</p>
          </div>
        ))}
      </div>

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
