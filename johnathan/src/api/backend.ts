export async function searchLocation(query: string) {
  const response = await fetch("http://127.0.0.1:5000/api/search-location", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });
  return response.json();
}
