import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import {
  generate_order_instructions,
  generateOrderInstructionsWithAgent,
} from "./processing/open_ai_processing";
import { Request, Response, RequestHandler } from "express";

// Load environment variables
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Configure CORS
const corsOptions = {
  origin: process.env.FRONTEND_URL || "http://localhost:5173", // Vite's default port
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type", "Authorization"],
};

// Middleware
app.use(cors(corsOptions));
app.use(express.json());

// Routes
app.post("/api/generate-instructions", (async (req: Request, res: Response) => {
  try {
    const { order } = req.body;

    if (!order) {
      return res.status(400).json({ error: "Order is required" });
    }

    const instructions = await generate_order_instructions(order);
    res.json(instructions);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
}) as RequestHandler);

app.post("/api/generate-instructions-with-agent", (async (
  req: Request,
  res: Response
) => {
  try {
    const { order } = req.body;

    if (!order) {
      return res.status(400).json({ error: "Order is required" });
    }

    const instructions = await generateOrderInstructionsWithAgent(order);
    res.json(instructions);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
}) as RequestHandler);

// Health check endpoint
app.get("/health", ((req: Request, res: Response) => {
  res.json({ status: "ok" });
}) as RequestHandler);

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
