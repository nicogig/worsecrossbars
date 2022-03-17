"""json_handlers:
An internal utility module to handle and validate JSON.
"""
import logging
import sys

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from worsecrossbars.utilities.json_schemas import simulation_schema


def validate_json(extracted_json: dict, json_schema: dict = simulation_schema) -> None:
    """Validate a given JSON object against a standard schema.

    Args:
      extracted_json: The JSON object to be validated.
    """

    try:
        validate(extracted_json, json_schema)
    except ValidationError as err:
        logging.exception("An exception occurred while validating the JSON file.")
        print(f"{err.message}")
        sys.exit(0)
