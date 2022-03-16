"""json_handlers:
An internal utility module to handle and validate JSON.
"""
import logging
import sys

from jsonschema import validate
from jsonschema.exceptions import ValidationError


def validate_json(extracted_json: dict) -> None:
    """Validate a given JSON object against a standard schema.

    Args:
      extracted_json: The JSON object to be validated.
    """

    json_schema = {
        "type": "object",
        "properties": {
            "simulations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model_size": {
                            "type": "string",
                            "enum": ["big", "regular", "small", "tiny"],
                        },
                        "optimiser": {"type": "string", "enum": ["adam", "sgd", "rmsprop"]},
                        "conductance_drifting": {"type": "boolean"},
                        "G_off": {"type": "number", "minimum": 0},
                        "G_on": {"type": "number", "minimum": 0},
                        "k_V": {"type": "number", "minimum": 0},
                        "discretisation": {"type": "boolean"},
                        "number_conductance_levels": {"type": "integer", "minimum": 0},
                        "excluded_weights_proportion": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "number_hidden_layers": {"type": "integer", "enum": [1, 2, 3, 4]},
                        "nonidealities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "StuckAtValue",
                                            "StuckDistribution",
                                            "D2DVariability",
                                            "IVNonlinear",
                                        ],
                                    },
                                    "parameters": {
                                        "type": "array",
                                        "items": {"type": ["array", "number"]},
                                    },
                                },
                            },
                        },
                        "noise_variance": {"type": "number", "minimum": 0},
                        "number_simulations": {"type": "integer", "minimum": 1},
                    },
                    "required": [
                        "model_size",
                        "optimiser",
                        "conductance_drifting",
                        "discretisation",
                        "G_off",
                        "G_on",
                        "k_V",
                        "number_hidden_layers",
                        "nonidealities",
                        "noise_variance",
                        "number_simulations",
                    ],
                    "allOf": [
                        {
                            "if": {"properties": {"conductance_drifting": {"const": True}}},
                            "then": {"properties": {"discretisation": {"const": False}}},
                        },
                        {
                            "if": {"properties": {"discretisation": {"const": True}}},
                            "then": {"properties": {"conductance_drifting": {"const": False}}},
                        },
                    ],
                },
            },
        },
        "required": ["simulations"],
    }

    try:
        validate(extracted_json, json_schema)
    except ValidationError as err:
        logging.exception("An exception occurred while validating the JSON file.")
        print(f"{err.message}")
        sys.exit(0)
