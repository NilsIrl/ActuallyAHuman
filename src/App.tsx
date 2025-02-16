import { useEffect, useState, useRef } from "react";
import MapView from "./MapView";
import { generate_order_instructions } from "./processing/open_ai_processing";
import { generateWaypoints } from "./processing/generate-waypoints";

interface OrderItemProps {
  orderNumber: number;
  text: string;
  active?: boolean;
}

const OrderItem: React.FC<OrderItemProps> = ({
  orderNumber,
  text,
  active = false,
}) => {
  return (
    <div
      className={`p-4 border rounded-md flex items-center gap-2 ${
        active ? "border-blue-400" : "border-gray-300"
      } ${!active ? "opacity-50" : ""}`}
    >
      {active && (
        <svg
          className="animate-spin h-5 w-5 text-blue-500"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
          />
        </svg>
      )}
      <span>
        {orderNumber}. {text}
      </span>
    </div>
  );
};

const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY!;

function App() {
  const [orderQueue, setOrderQueue] = useState<string[]>([]);

  const [text, setText] = useState("");

  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      setError(null); // Clear any previous errors
      setOrderQueue((prevOrders) => [...prevOrders, text]);
      setText("");

      // send a request to the server to add the order to the queue
      try {
        const orderResponse = await fetch("http://localhost:8000/add_order", {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            order: text,
          }),
        });
        const orderData = await orderResponse.json();
        console.log(orderData);

        const instructions = await generate_order_instructions(text);
        console.log(instructions);

        if (instructions) {
          const waypoints = await generateWaypoints(
            instructions.start,
            instructions.end
          );
          console.log(waypoints);

          const waypointsResponse = await fetch(
            "http://localhost:8000/send_waypoints",
            {
              method: "POST",
              headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                waypoints: waypoints,
              }),
            }
          );
          const waypointsData = await waypointsResponse.json();
          console.log(waypointsData);
        }
      } catch (error) {
        console.error("Error:", error);
        if (error instanceof Error) {
          setError(error.message);
        } else {
          setError("An unknown error occurred");
        }
      }
    }
  };

  return (
    <div className="max-w-2xl mx-auto bg-white shadow-md rounded-md p-6 mt-10">

      <h1 className="text-3xl font-bold mb-4 text-center">Order on OmNom</h1>
      <input
        type="text"
        placeholder="Enter your order"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            handleSubmit(e);
          }
        }}
        className="w-full border-2 border-gray-300 rounded-md p-2 mb-4"
      />
      {error && (
        <div className="text-red-500 mb-4 p-2 bg-red-100 rounded">{error}</div>
      )}
      {orderQueue.length > 0 ? (
        orderQueue.length > 1 ? (
          <div className="flex gap-4">
            <div className="flex-1">
              <h2 className="text-xl font-semibold mb-2">Current Order</h2>
              <OrderItem
                orderNumber={1}
                text={orderQueue[0]}
                active
              />
            </div>
            <div className="flex-1">
              <h2 className="text-xl font-semibold mb-2">Upcoming Orders</h2>
              <div className="space-y-2">
                {orderQueue.slice(1).map((order, index) => (
                  <OrderItem
                    key={index}
                    orderNumber={index + 2}
                    text={order}
                  />
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div>
            <h2 className="text-xl font-semibold mb-2">Current Order</h2>
            <OrderItem
              orderNumber={1}
              text={orderQueue[0]}
              active
            />
          </div>
        )
      ) : (
        <div className="text-center text-gray-500">No orders in queue</div>
      )}
      <div className="mt-6 h-[500px] w-full border border-gray-300 rounded-md overflow-hidden">
        <MapView />
      </div>
    </div>
  );
}

export default App;
