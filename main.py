from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import joblib
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Test Gemini connection at startup
try:
    logger.info("Gemini AI connected successfully")
except Exception as e:
    logger.error(f"Error connecting to Gemini AI: {e}")

# Load models at startup (better performance)
try:
    loaded_preprocessor = joblib.load("preprocessor_joblib.pkl")
    loaded_model = joblib.load("./random_forest_model_joblib.pkl")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    loaded_preprocessor = None
    loaded_model = None

# Define column names
COLUMNS = [
    "N_req_kg_per_ha",
    "P_req_kg_per_ha",
    "K_req_kg_per_ha",
    "Temperature_C",
    "Humidity_%",
    "pH",
    "Crop_cotton",
    "Crop_maize",
    "Crop_rice",
]

OPTIMAL_RANGES = {
    "cotton": {
        "N_req_kg_per_ha": (100, 180),
        "P_req_kg_per_ha": (20, 60),
        "K_req_kg_per_ha": (50, 80),
        "Temperature_C": (21, 37),
        "Humidity_%": (60, 80),
        "pH": (5.8, 8.0),
    },
    "maize": {
        "N_req_kg_per_ha": (110, 180),
        "P_req_kg_per_ha": (60, 90),
        "K_req_kg_per_ha": (75, 120),
        "Temperature_C": (20, 33),
        "Humidity_%": (65, 85),
        "pH": (5.5, 7.5),
    },
    "rice": {
        "N_req_kg_per_ha": (114, 220),
        "P_req_kg_per_ha": (56, 149),
        "K_req_kg_per_ha": (121, 230),
        "Temperature_C": (20, 36),
        "Humidity_%": (70, 90),
        "pH": (5.0, 7.0),
    },
    "chickpea": {
        "N_req_kg_per_ha": (15, 30),
        "P_req_kg_per_ha": (40, 60),
        "K_req_kg_per_ha": (17, 30),
        "Temperature_C": (18, 29),
        "Humidity_%": (21, 41),
        "pH": (6.0, 7.5),
    },
}


def validate_input_data(data: List[List]) -> tuple[bool, str]:
    """
    Validate the input data format and values
    """
    if not isinstance(data, list):
        return False, "Data must be a list"

    if len(data) == 0:
        return False, "Data cannot be empty"

    for i, row in enumerate(data):
        if not isinstance(row, list):
            return False, f"Row {i} must be a list"

        if len(row) != len(COLUMNS):
            return (
                False,
                f"Row {i} must have exactly {len(COLUMNS)} values, got {len(row)}",
            )

        # Check data types for first 6 numerical columns
        for j in range(6):
            if not isinstance(row[j], (int, float)):
                return False, f"Row {i}, column {j} ({COLUMNS[j]}) must be a number"

        # Check boolean values for crop columns
        for j in range(6, 9):
            if not isinstance(row[j], bool):
                return False, f"Row {i}, column {j} ({COLUMNS[j]}) must be a boolean"

    return True, "Valid"


def get_crop_type(crop_cotton: bool, crop_maize: bool, crop_rice: bool) -> str:
    """Determine the crop type from boolean flags"""
    if crop_cotton:
        return "cotton"
    elif crop_maize:
        return "maize"
    elif crop_rice:
        return "rice"
    else:
        return "chickpea"  # Default when all three are False


# @app.route("/predict", methods=["POST"])
# def predict():
#    """
#    Predict using the trained model
#
#    Expected JSON format:
#    {
#        "data": [
#            [6.75, 3.0, 6.75, 28, 65, 6.0, true, false, false],
#            [22.00975, 10.56468, 19.36858, 25, 80, 6.5, false, false, true]
#        ]
#    }
#    """
#    try:
#        # Check if models are loaded
#        if loaded_preprocessor is None or loaded_model is None:
#            return jsonify(
#                {
#                    "error": "Models not loaded. Please check if model files exist.",
#                    "success": False,
#                }
#            ), 500
#
#        # Get JSON data from request
#        if not request.is_json:
#            return jsonify({"error": "Request must be JSON", "success": False}), 400
#
#        json_data = request.get_json()
#
#        # Validate JSON structure
#        if "data" not in json_data:
#            return jsonify(
#                {"error": 'JSON must contain "data" field', "success": False}
#            ), 400
#
#        input_data = json_data["data"]
#
#        # Validate input data
#        is_valid, validation_message = validate_input_data(input_data)
#        if not is_valid:
#            return jsonify(
#                {"error": f"Invalid input data: {validation_message}", "success": False}
#            ), 400
#
#        # Create DataFrame
#        df = pd.DataFrame(input_data, columns=COLUMNS)
#        logger.info(f"Processing {len(df)} rows of data")
#
#        # Transform data using preprocessor
#        transformed_data = loaded_preprocessor.transform(df)
#        logger.info(f"Data transformed successfully, shape: {transformed_data.shape}")
#
#        # Make predictions
#        predictions = loaded_model.predict(transformed_data)
#        logger.info(f"Predictions made successfully")
#
#        # Convert numpy types to Python types for JSON serialization
#        predictions_list = [
#            float(pred)
#            if isinstance(pred, np.floating)
#            else int(pred)
#            if isinstance(pred, np.integer)
#            else pred
#            for pred in predictions
#        ]
#
#        return jsonify(
#            {
#                "predictions": predictions_list,
#                "num_predictions": len(predictions_list),
#                "success": True,
#            }
#        )
#
#    except Exception as e:
#        logger.error(f"Error during prediction: {str(e)}")
#        return jsonify({"error": f"Prediction failed: {str(e)}", "success": False}), 500
#


@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Predict for a single data point

    Expected JSON format (frontend):
    {
        "nReq": 6.75,
        "pReq": 3.0,
        "kReq": 6.75,
        "temperature": 28,
        "humidity": 65,
        "pH": 6.0,
        "cropCotton": true,
        "cropMaize": false,
        "cropRice": false
    }
    """
    try:
        # Check if models are loaded
        if loaded_preprocessor is None or loaded_model is None:
            return jsonify(
                {
                    "error": "Models not loaded. Please check if model files exist.",
                    "success": False,
                }
            ), 500

        # Ensure request is JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "success": False}), 400

        json_data = request.get_json()

        # Map frontend keys to backend model column names
        key_mapping = {
            "nReq": "N_req_kg_per_ha",
            "pReq": "P_req_kg_per_ha",
            "kReq": "K_req_kg_per_ha",
            "temperature": "Temperature_C",
            "humidity": "Humidity_%",
            "pH": "pH",
            "cropCotton": "Crop_cotton",
            "cropMaize": "Crop_maize",
            "cropRice": "Crop_rice",
        }

        # Translate frontend input to backend column names
        translated_data = {
            backend_key: json_data[frontend_key]
            for frontend_key, backend_key in key_mapping.items()
            if frontend_key in json_data
        }

        # Validate that all required columns are present
        missing_columns = [col for col in COLUMNS if col not in translated_data]
        if missing_columns:
            return jsonify(
                {
                    "error": f"Missing required fields: {missing_columns}",
                    "success": False,
                }
            ), 400

        # Create single row data
        single_row = [translated_data[col] for col in COLUMNS]

        # Validate input
        is_valid, validation_message = validate_input_data([single_row])
        if not is_valid:
            return jsonify(
                {"error": f"Invalid input data: {validation_message}", "success": False}
            ), 400

        # Create DataFrame
        df = pd.DataFrame([single_row], columns=COLUMNS)

        # Transform and predict
        transformed_data = loaded_preprocessor.transform(df)
        prediction = loaded_model.predict(transformed_data)[0]

        # Convert numpy type to Python type
        prediction_value = (
            float(prediction)
            if isinstance(prediction, np.floating)
            else int(prediction)
            if isinstance(prediction, np.integer)
            else prediction
        )

        return jsonify(
            {"prediction": prediction_value, "input_data": json_data, "success": True}
        )

    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}", "success": False}), 500


def generate_farming_prompt(
    input_data: Dict[str, Any],
    current_yield: float,
    target_yield: Optional[float] = None,
) -> str:
    """
    Generate a farmer-friendly prompt for Gemini AI to provide concise farming recommendations
    """
    crop_type = get_crop_type(
        input_data.get("Crop_cotton", False),
        input_data.get("Crop_maize", False),
        input_data.get("Crop_rice", False),
    )

    # Use chickpea as default if crop type not found
    optimal_ranges = OPTIMAL_RANGES.get(crop_type, OPTIMAL_RANGES["chickpea"])

    target_info = f" Target yield: {target_yield} units." if target_yield else ""

    # Add special note for chickpea about nitrogen fixation
    nitrogen_note = ""
    if crop_type == "chickpea":
        nitrogen_note = (
            " (Note: Chickpea fixes nitrogen naturally, requires less N fertilizer)"
        )

    prompt = f"""
You are an agricultural expert helping farmers. A farmer grows {crop_type} and needs simple, actionable advice.

CURRENT CONDITIONS:
- Crop: {crop_type.title()}{nitrogen_note}
- Nitrogen (N): {input_data.get("N_req_kg_per_ha", 0)} kg/ha (Optimal: {optimal_ranges["N_req_kg_per_ha"][0]}-{optimal_ranges["N_req_kg_per_ha"][1]})
- Phosphorus (P): {input_data.get("P_req_kg_per_ha", 0)} kg/ha (Optimal: {optimal_ranges["P_req_kg_per_ha"][0]}-{optimal_ranges["P_req_kg_per_ha"][1]})
- Potassium (K): {input_data.get("K_req_kg_per_ha", 0)} kg/ha (Optimal: {optimal_ranges["K_req_kg_per_ha"][0]}-{optimal_ranges["K_req_kg_per_ha"][1]})
- Temperature: {input_data.get("Temperature_C", 0)}°C (Optimal: {optimal_ranges["Temperature_C"][0]}-{optimal_ranges["Temperature_C"][1]}°C)
- Humidity: {input_data.get("Humidity_%", 0)}% (Optimal: {optimal_ranges["Humidity_%"][0]}-{optimal_ranges["Humidity_%"][1]}%)
- Soil pH: {input_data.get("pH", 0)} (Optimal: {optimal_ranges["pH"][0]}-{optimal_ranges["pH"][1]})
- Current yield: {current_yield} units{target_info}

PROVIDE SIMPLE RECOMMENDATIONS IN THIS EXACT FORMAT:

## 🎯 MAIN PROBLEMS
• List 2-3 biggest issues (in bullet points)

## 🌱 FERTILIZER FIXES
• **Nitrogen**: [Specific amount] kg/ha - Use [fertilizer name]
• **Phosphorus**: [Specific amount] kg/ha - Use [fertilizer name] 
• **Potassium**: [Specific amount] kg/ha - Use [fertilizer name]

## 🌡️ ENVIRONMENT
• Temperature: [Action needed or "Good as is"]
• Humidity: [Action needed or "Good as is"]
• Soil pH: [Action needed or "Good as is"]

## 📈 TOP 3 PRIORITIES
1. [Most important action]
2. [Second priority]
3. [Third priority]

## 💰 EXPECTED RESULTS
• Yield increase: [X]% (from {current_yield} to [predicted] units)
• Investment: [High/Medium/Low cost]
• Return timeframe: [Next season/6 months/etc.]

## ⏰ WHEN TO ACT
• **NOW**: [Immediate actions]
• **NEXT SEASON**: [Actions for next planting]
• **ONGOING**: [Regular monitoring tasks]

Keep each point SHORT (maximum 10-15 words). Use simple farming language, not technical jargon. Focus on ACTIONABLE steps farmers can take immediately.
{"Special note for chickpea: Remember that chickpea fixes nitrogen from air, so avoid over-fertilizing with nitrogen." if crop_type == "chickpea" else ""}
"""

    return prompt


def parse_ai_recommendations(ai_response: str) -> Dict[str, Any]:
    """
    Parse the AI response and extract key information for better API response structure
    """
    try:
        # Extract main sections using simple string parsing
        sections = {
            "main_problems": [],
            "fertilizer_recommendations": {},
            "environment_actions": {},
            "top_priorities": [],
            "expected_results": {},
            "action_timeline": {},
        }

        lines = ai_response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Identify sections
            if "MAIN PROBLEMS" in line:
                current_section = "main_problems"
            elif "FERTILIZER FIXES" in line:
                current_section = "fertilizer"
            elif "ENVIRONMENT" in line:
                current_section = "environment"
            elif "TOP 3 PRIORITIES" in line:
                current_section = "priorities"
            elif "EXPECTED RESULTS" in line:
                current_section = "results"
            elif "WHEN TO ACT" in line:
                current_section = "timeline"
            elif line.startswith("•") or line.startswith("-"):
                # Parse bullet points
                content = line[1:].strip()
                if current_section == "main_problems":
                    sections["main_problems"].append(content)
                elif current_section == "fertilizer" and "**" in content:
                    # Parse fertilizer recommendations
                    if "Nitrogen" in content:
                        sections["fertilizer_recommendations"]["nitrogen"] = content
                    elif "Phosphorus" in content:
                        sections["fertilizer_recommendations"]["phosphorus"] = content
                    elif "Potassium" in content:
                        sections["fertilizer_recommendations"]["potassium"] = content
                elif current_section == "environment":
                    if "Temperature" in content:
                        sections["environment_actions"]["temperature"] = (
                            content.split(":")[1].strip() if ":" in content else content
                        )
                    elif "Humidity" in content:
                        sections["environment_actions"]["humidity"] = (
                            content.split(":")[1].strip() if ":" in content else content
                        )
                    elif "pH" in content:
                        sections["environment_actions"]["ph"] = (
                            content.split(":")[1].strip() if ":" in content else content
                        )
                elif current_section == "results":
                    if "Yield increase" in content:
                        sections["expected_results"]["yield_increase"] = content
                    elif "Investment" in content:
                        sections["expected_results"]["investment_level"] = content
                    elif "Return timeframe" in content:
                        sections["expected_results"]["timeframe"] = content
            elif line.startswith(("1.", "2.", "3.")):
                # Parse numbered priorities
                if current_section == "priorities":
                    sections["top_priorities"].append(line[2:].strip())
            elif line.startswith(("**NOW**", "**NEXT SEASON**", "**ONGOING**")):
                # Parse timeline actions
                if "NOW" in line:
                    sections["action_timeline"]["immediate"] = (
                        line.split(":")[1].strip() if ":" in line else ""
                    )
                elif "NEXT SEASON" in line:
                    sections["action_timeline"]["next_season"] = (
                        line.split(":")[1].strip() if ":" in line else ""
                    )
                elif "ONGOING" in line:
                    sections["action_timeline"]["ongoing"] = (
                        line.split(":")[1].strip() if ":" in line else ""
                    )

        return sections
    except Exception as e:
        logger.warning(f"Could not parse AI response structure: {e}")
        return {"raw_response": ai_response}


@app.route("/recommendations", methods=["POST"])
def get_farming_recommendations():
    """
    Get AI-powered farming recommendations using Gemini AI with farmer-friendly format
    """
    try:
        # [Keep existing validation code as is]
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "success": False}), 400

        json_data = request.get_json()

        if "current_yield" not in json_data:
            return jsonify(
                {"error": "current_yield field is required", "success": False}
            ), 400

        missing_columns = [col for col in COLUMNS if col not in json_data]
        if missing_columns:
            return jsonify(
                {
                    "error": f"Missing required fields: {missing_columns}",
                    "success": False,
                }
            ), 400

        current_yield = json_data["current_yield"]
        target_yield = json_data.get("target_yield")

        farming_data = {col: json_data[col] for col in COLUMNS}
        single_row = [farming_data[col] for col in COLUMNS]

        is_valid, validation_message = validate_input_data([single_row])
        if not is_valid:
            return jsonify(
                {
                    "error": f"Invalid farming data: {validation_message}",
                    "success": False,
                }
            ), 400

        # Generate the farmer-friendly prompt
        prompt = generate_farming_prompt(farming_data, current_yield, target_yield)
        logger.info("Generated farmer-friendly prompt for Gemini AI")

        # Get recommendations from Gemini AI
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            raw_recommendations = response.text
            logger.info("Successfully received recommendations from Gemini AI")

        except Exception as gemini_error:
            logger.error(
                f"Error getting recommendations from Gemini AI: {gemini_error}"
            )
            return jsonify(
                {
                    "error": f"Failed to get AI recommendations: {str(gemini_error)}",
                    "success": False,
                }
            ), 500

        # Parse the structured recommendations
        parsed_recommendations = parse_ai_recommendations(raw_recommendations)

        # Get ML prediction if available
        predicted_yield = None
        if loaded_preprocessor is not None and loaded_model is not None:
            try:
                df = pd.DataFrame([single_row], columns=COLUMNS)
                transformed_data = loaded_preprocessor.transform(df)
                prediction = loaded_model.predict(transformed_data)[0]
                predicted_yield = (
                    float(prediction)
                    if isinstance(prediction, np.floating)
                    else int(prediction)
                    if isinstance(prediction, np.integer)
                    else prediction
                )
            except Exception as ml_error:
                logger.warning(f"ML prediction failed: {ml_error}")

        # Enhanced response structure
        response_data = {
            "recommendations": {
                "formatted_advice": raw_recommendations,
                "structured_data": parsed_recommendations,
            },
            "input_data": {
                "farming_parameters": farming_data,
                "current_yield": current_yield,
                "target_yield": target_yield,
            },
            "crop_type": get_crop_type(
                farming_data.get("Crop_cotton", False),
                farming_data.get("Crop_maize", False),
                farming_data.get("Crop_rice", False),
            ),
            "summary": {
                "main_issues_count": len(
                    parsed_recommendations.get("main_problems", [])
                ),
                "priority_actions": parsed_recommendations.get("top_priorities", [])[
                    :3
                ],
                "quick_wins": parsed_recommendations.get("action_timeline", {}).get(
                    "immediate", "Check recommendations"
                ),
            },
            "success": True,
        }

        if predicted_yield is not None:
            response_data["ml_predicted_yield"] = predicted_yield

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {str(e)}")
        return jsonify(
            {"error": f"Recommendation generation failed: {str(e)}", "success": False}
        ), 500


@app.route("/schema", methods=["GET"])
def get_schema():
    """Get the expected data schema"""
    return jsonify(
        {
            "columns": COLUMNS,
            "data_types": {
                "N_req_kg_per_ha": "float",
                "P_req_kg_per_ha": "float",
                "K_req_kg_per_ha": "float",
                "Temperature_C": "int/float",
                "Humidity_%": "int/float",
                "pH": "float",
                "Crop_cotton": "boolean",
                "Crop_maize": "boolean",
                "Crop_rice": "boolean",
            },
            "crop_logic": {
                "cotton": "Crop_cotton = true, others = false",
                "maize": "Crop_maize = true, others = false",
                "rice": "Crop_rice = true, others = false",
                "chickpea": "All crop flags = false (default)",
            },
            "example_batch": {
                "data": [
                    [6.75, 3.0, 6.75, 28, 65, 6.0, True, False, False],  # Cotton
                    [
                        22.00975,
                        10.56468,
                        19.36858,
                        25,
                        80,
                        6.5,
                        False,
                        False,
                        True,
                    ],  # Rice
                    [30.5, 45.0, 25.0, 22, 60, 6.8, False, False, False],  # Chickpea
                ]
            },
            "example_single_cotton": {
                "N_req_kg_per_ha": 6.75,
                "P_req_kg_per_ha": 3.0,
                "K_req_kg_per_ha": 6.75,
                "Temperature_C": 28,
                "Humidity_%": 65,
                "pH": 6.0,
                "Crop_cotton": True,
                "Crop_maize": False,
                "Crop_rice": False,
            },
            "example_single_chickpea": {
                "N_req_kg_per_ha": 30.5,
                "P_req_kg_per_ha": 45.0,
                "K_req_kg_per_ha": 25.0,
                "Temperature_C": 22,
                "Humidity_%": 60,
                "pH": 6.8,
                "Crop_cotton": False,
                "Crop_maize": False,
                "Crop_rice": False,
            },
            "example_recommendations": {
                "N_req_kg_per_ha": 80.5,
                "P_req_kg_per_ha": 25.0,
                "K_req_kg_per_ha": 45.5,
                "Temperature_C": 28,
                "Humidity_%": 65,
                "pH": 6.2,
                "Crop_cotton": True,
                "Crop_maize": False,
                "Crop_rice": False,
                "current_yield": 45.2,
                "target_yield": 60.0,
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
