interface Coordinates {
  latitude: number;
  longitude: number;
}

async function generateWaypoints(
  start: Coordinates,
  end: Coordinates,
  numWaypoints: number = 10
): Promise<Coordinates[]> {
  const API_KEY = import.meta.env.VITE_OPEN_ROUTE_SERVICE_KEY!;
  if (!API_KEY) {
    throw new Error("API key for OpenRouteService is not set.");
  }

  const url = `https://api.openrouteservice.org/v2/directions/foot-walking/geojson`;

  const requestBody = {
    coordinates: [
      [start.longitude, start.latitude],
      [end.longitude, end.latitude],
    ],
  };

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Accept:
        "application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8",
      "Content-Type": "application/json; charset=utf-8",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    throw new Error(
      `API call failed: ${response.status} ${response.statusText}`
    );
  }

  const data = await response.json();

  if (!data.features || data.features.length === 0) {
    throw new Error("No route data found.");
  }

  const routeCoords: Coordinates[] = data.features[0].geometry.coordinates;

  // Sample waypoints along the route
  const step = Math.max(1, Math.floor(routeCoords.length / numWaypoints));
  const waypoints = routeCoords.filter((_, index) => index % step === 0);

  return waypoints;
}

export { generateWaypoints };
