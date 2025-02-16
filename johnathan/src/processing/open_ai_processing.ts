import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod.mjs";
import { z } from "zod";

// These imports assume you have langchainjs installed
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { ChatOpenAI } from "@langchain/openai";
import { DynamicTool } from "langchain/tools";

const openai = new OpenAI({
  organization: import.meta.env.VITE_OPENAI_ORGANIZATION,
  apiKey: import.meta.env.VITE_OPENAI_API_KEY,
  dangerouslyAllowBrowser: true,
});

const processedInstructionsSchema = z.object({
  start: z.object({
    longitude: z.number(),
    latitude: z.number(),
  }),
  end: z.object({
    longitude: z.number(),
    latitude: z.number(),
  }),
  order: z.array(
    z.object({
      name: z.string(),
      quantity: z.number(),
    })
  ),
});

// Configure logging
const logPrefix = "[OpenAI Processing]";
const log = {
  info: (msg: string, ...args: any[]) =>
    console.log(`${logPrefix} INFO: ${msg}`, ...args),
  error: (msg: string, ...args: any[]) =>
    console.error(`${logPrefix} ERROR: ${msg}`, ...args),
  warn: (msg: string, ...args: any[]) =>
    console.warn(`${logPrefix} WARN: ${msg}`, ...args),
  debug: (msg: string, ...args: any[]) =>
    console.debug(`${logPrefix} DEBUG: ${msg}`, ...args),
};

// Updated getLocation to return a Promise
function getLocation(): Promise<GeolocationPosition> {
  log.info("Getting user location");
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      log.error("Geolocation not supported");
      reject(new Error("Geolocation is not supported by your browser."));
    } else {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          log.info("Location obtained", {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
          resolve(position);
        },
        (error) => {
          log.error("Failed to get location", error);
          reject(error);
        }
      );
    }
  });
}

/**
 * Helper function to query the Google Maps API for a destination.
 *
 * Note: Replace the URL and parameters with your preferred endpoint (e.g., Places API).
 * Also ensure your Google Maps API key is set (consider using an environment variable).
 */
async function getDestinationFromGoogleMaps(
  query: string
): Promise<{ lat: number; lng: number }> {
  log.info("Querying Google Maps API", { query });
  try {
    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    const url = `https://maps.googleapis.com/maps/api/place/textsearch/json?query=${encodeURIComponent(
      query
    )}&key=${apiKey}`;

    const response = await fetch(url);
    const data = await response.json();

    if (!data.results || data.results.length === 0) {
      log.error("No results found from Google Maps", { query });
      throw new Error("No destination found using Google Maps.");
    }

    // We simply choose the first result here.
    const location = data.results[0].geometry.location;
    log.info("Destination found", location);
    return location; // format: { lat: number, lng: number }
  } catch (error) {
    log.error("Google Maps API error", { error, query });
    throw error;
  }
}

/**
 * Define a DynamicTool for the agent that uses Google Maps.
 */
const googleMapsTool = new DynamicTool({
  name: "GoogleMapsAPI",
  description:
    "Query Google Maps to determine a suitable destination given a text query.",
  func: async (input: string) => {
    try {
      const location = await getDestinationFromGoogleMaps(input);
      // Return a formatted string the agent can interpret.
      return `Destination found: latitude ${location.lat}, longitude ${location.lng}`;
    } catch (error: any) {
      return `Error querying Google Maps: ${error.message}`;
    }
  },
});

/**
 * A new function that uses a LangChain agent to generate order instructions.
 * The agent uses the GoogleMaps tool to autonomously decide on the end point.
 */
async function generateOrderInstructionsWithAgent(order: string) {
  log.info("Generating order instructions with agent", { order });
  try {
    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini",
      openAIApiKey: import.meta.env.VITE_OPENAI_API_KEY,
    });
    const tools = [googleMapsTool];

    log.debug("Initializing agent executor");
    const executor = await initializeAgentExecutorWithOptions(tools, llm, {
      agentType: "chat-conversational-react-description",
      verbose: true,
    });

    const position = await getLocation();
    const { latitude, longitude } = position.coords;

    const prompt = `
      I have a delivery order with the following details: ${order}.
      The current location is latitude: ${latitude}, longitude: ${longitude}.
      Please decide on a suitable destination for this order.
    `;

    log.debug("Running agent executor", { prompt });
    const result = await executor.run(prompt);
    log.info("Agent execution completed", { result });
    return result;
  } catch (error) {
    log.error("Agent execution failed", { error, order });
    throw error;
  }
}

async function generate_order_instructions(order: string) {
  log.info("Generating order instructions", { order });
  try {
    const position = await getLocation();
    const { latitude, longitude } = position.coords;

    log.debug("Making OpenAI API call", {
      model: "gpt-4o-mini",
      location: { latitude, longitude },
    });

    const endLocationStr = (
      await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "user",
            content: `Order: ${order}. What is the end location? Suggest a generic food location that is close to the order if no specific location is mentioned. Only return the location, no other text.`,
          },
        ],
      })
    ).choices[0].message.content;

    console.log("endLocationStr", endLocationStr);

    const endLocation = await fetch(
      "http://127.0.0.1:5000/api/search-location",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: endLocationStr }),
      }
    );

    const endLocationData = await endLocation.json();
    if (!endLocationData.success) {
      throw new Error(`Failed to find location: ${endLocationData.error}`);
    }

    // Now we can safely use the location data
    const { endLatitude, endLongitude } = endLocationData.location;

    console.log(
      `Current location: ${latitude}, ${longitude}. Order: ${order}. Please use the current location as the starting point. Please use ${endLatitude}, ${endLongitude} as your ending point`
    );

    const response = await openai.beta.chat.completions.parse({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant that generates instructions for a delivery driver.",
        },
        {
          role: "user",
          content: `Current location: ${latitude}, ${longitude}. Order: ${order}. Please use the current location as the starting point. Please use ${endLatitude}, ${endLongitude} as your ending point"`,
        },
      ],
      response_format: zodResponseFormat(
        processedInstructionsSchema,
        "processed_instructions"
      ),
    });

    log.info("Successfully generated instructions");
    log.debug("API response", response.choices[0].message.parsed);
    return response.choices[0].message.parsed;
  } catch (error) {
    log.error("Failed to generate instructions", { error, order });
    throw error;
  }
}

async function generate_order_instructions2(order: string) {
  log.info("Generating order instructions", { order });
  try {
    const position = await getLocation();
    const { latitude, longitude } = position.coords;

    const response = await openai.beta.chat.completions.parse({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant that generates instructions for a delivery driver.",
        },
        {
          role: "user",
          content: `Current location: ${latitude}, ${longitude}. Order: ${order}. Please use the current location as the starting point. Please use 37.424950518222445, -122.17637949085739 as your ending point"`,
        },
      ],
      response_format: zodResponseFormat(
        processedInstructionsSchema,
        "processed_instructions"
      ),
    });

    log.info("Successfully generated instructions");
    log.debug("API response", response.choices[0].message.parsed);
    return response.choices[0].message.parsed;
  } catch (error) {
    log.error("Failed to generate instructions", { error, order });
    throw error;
  }
}

// Export both implementations so you can choose which one to use.
export {
  generate_order_instructions,
  generate_order_instructions2,
  generateOrderInstructionsWithAgent,
  getLocation,
};
