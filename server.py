from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Any

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict using the trained model

    Expected JSON format:
    {
        "data": [
            [6.75, 3.0, 6.75, 28, 65, 6.0, true, false, false],
            [22.00975, 10.56468, 19.36858, 25, 80, 6.5, false, false, true]
        ]
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

        # Get JSON data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "success": False}), 400

        json_data = request.get_json()

        # Validate JSON structure
        if "data" not in json_data:
            return jsonify(
                {"error": 'JSON must contain "data" field', "success": False}
            ), 400

        input_data = json_data["data"]

        # Validate input data
        is_valid, validation_message = validate_input_data(input_data)
        if not is_valid:
            return jsonify(
                {"error": f"Invalid input data: {validation_message}", "success": False}
            ), 400

        # Create DataFrame
        df = pd.DataFrame(input_data, columns=COLUMNS)
        logger.info(f"Processing {len(df)} rows of data")

        # Transform data using preprocessor
        transformed_data = loaded_preprocessor.transform(df)
        logger.info(f"Data transformed successfully, shape: {transformed_data.shape}")

        # Make predictions
        predictions = loaded_model.predict(transformed_data)
        logger.info(f"Predictions made successfully")

        # Convert numpy types to Python types for JSON serialization
        predictions_list = [
            float(pred)
            if isinstance(pred, np.floating)
            else int(pred)
            if isinstance(pred, np.integer)
            else pred
            for pred in predictions
        ]

        return jsonify(
            {
                "predictions": predictions_list,
                "num_predictions": len(predictions_list),
                "success": True,
            }
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}", "success": False}), 500


@app.route("/predict/single", methods=["POST"])
def predict_single():
    """
    Predict for a single data point

    Expected JSON format:
    {
        "N_req_kg_per_ha": 6.75,
        "P_req_kg_per_ha": 3.0,
        "K_req_kg_per_ha": 6.75,
        "Temperature_C": 28,
        "Humidity_%": 65,
        "pH": 6.0,
        "Crop_cotton": true,
        "Crop_maize": false,
        "Crop_rice": false
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

        # Get JSON data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "success": False}), 400

        json_data = request.get_json()

        # Validate that all required columns are present
        missing_columns = [col for col in COLUMNS if col not in json_data]
        if missing_columns:
            return jsonify(
                {
                    "error": f"Missing required fields: {missing_columns}",
                    "success": False,
                }
            ), 400

        # Create single row data
        single_row = [json_data[col] for col in COLUMNS]

        # Validate the single row
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


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    model_status = loaded_preprocessor is not None and loaded_model is not None
    return jsonify(
        {
            "status": "healthy" if model_status else "unhealthy",
            "models_loaded": model_status,
            "message": "API is running" if model_status else "Models not loaded",
        }
    )


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
            "example_batch": {
                "data": [
                    [6.75, 3.0, 6.75, 28, 65, 6.0, True, False, False],
                    [22.00975, 10.56468, 19.36858, 25, 80, 6.5, False, False, True],
                ]
            },
            "example_single": {
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
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
