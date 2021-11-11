"""
json_handlers:
An internal utility module to handle and validate JSON.
"""
import sys
from jsonschema import validate
from jsonschema.exceptions import ValidationError

def validate_json(extracted_json):
    """
    Validate a given JSON object against a standard schema.

    Args:
      extracted_json: The JSON object to be validated.
    """
    json_schema = {
        "type": "object",
        "properties" : {
            "HRS_LRS_ratio": {"type" : "integer"},
            "number_of_conductance_levels": {"type": "integer"},
            "excluded_weights_proportion" : {"type": "number"},
            "number_hidden_layers": {"type": "integer"},
            "fault_type": {"type": "string"},
            "noise_variance": {"type": "number"},
            "number_ANNs": {"type": "integer"},
            "number_simulations": {"type": "integer"}
        }
    }
    try:
        validate(extracted_json, json_schema)
    except ValidationError as err:
        print(f"{err.message}")
        sys.exit(0)
