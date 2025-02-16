import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod.mjs";
import { z } from "zod";

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

// Updated getLocation to return a Promise
function getLocation(): Promise<GeolocationPosition> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("Geolocation is not supported by your browser."));
    } else {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          console.log(
            "Located position:",
            position.coords.latitude,
            position.coords.longitude
          );
          resolve(position);
        },
        (error) => {
          console.error("Error retrieving location:", error);
          reject(error);
        }
      );
    }
  });
}

// Function to get current location
// const getCurrentLocation = (): Promise<GeolocationPosition> => {
//   return new Promise((resolve, reject) => {
//     if (!navigator.geolocation) {
//       reject(new Error("Geolocation is not supported by your browser"));
//     } else {
//       const options = {
//         enableHighAccuracy: true,
//         timeout: 10000, // Increased timeout
//         maximumAge: 0,
//       };

//       // First check the permission status
//       const checkPermission = async () => {
//         try {
//           // Check if the Permissions API is supported
//           if (navigator.permissions && navigator.permissions.query) {
//             const result = await navigator.permissions.query({
//               name: "geolocation",
//             });
//             console.log("Geolocation permission status:", result.state);

//             if (result.state === "denied") {
//               reject(
//                 new Error(
//                   "Location permission was denied. Please enable location access in your browser settings."
//                 )
//               );
//               return;
//             }
//           }

//           // Proceed with getting location
//           navigator.geolocation.getCurrentPosition(
//             (position) => {
//               console.log("Got position:", position.coords);
//               resolve(position);
//             },
//             (err) => {
//               console.warn(`ERROR(${err.code}): ${err.message}`);
//               switch (err.code) {
//                 case err.PERMISSION_DENIED:
//                   reject(
//                     new Error(
//                       "Please allow location access to use this feature"
//                     )
//                   );
//                   break;
//                 case err.POSITION_UNAVAILABLE:
//                   reject(
//                     new Error(
//                       "Location information is unavailable. Please check your device's location settings."
//                     )
//                   );
//                   break;
//                 case err.TIMEOUT:
//                   reject(
//                     new Error("Location request timed out. Please try again.")
//                   );
//                   break;
//                 default:
//                   reject(err);
//               }
//             },
//             options
//           );
//         } catch (error) {
//           reject(
//             new Error(
//               "Error accessing location services. Please check your browser settings."
//             )
//           );
//         }
//       };

//       checkPermission();
//     }
//   });
// };

async function generate_order_instructions(order: string) {
  try {
    // Use await here since getLocation now returns a Promise
    const position = await getLocation();
    console.log("OpenAI Position:", position);
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
          content: `Current location: ${latitude}, ${longitude}. Order: ${order}. Please use the current location as the starting point.`,
        },
      ],
      response_format: zodResponseFormat(
        processedInstructionsSchema,
        "processed_instructions"
      ),
    });
    return response.choices[0].message.parsed;
  } catch (error) {
    console.error("Error generating order instructions:", error);
    throw error;
  }
}

// Export the updated getLocation so it can be used in other components
export { generate_order_instructions, getLocation };
