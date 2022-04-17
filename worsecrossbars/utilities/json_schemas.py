"""json_schemas:
A module containing the JSON schemas used for plotting and simulation JSONs.
"""

PLOT_SCHEMA = {
    "type": "object",
    "properties": {
        "plots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "keyFeatures": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

SIMULATION_SCHEMA = {
    "type": "object",
    "properties": {
        "simulations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "modelSize": {
                        "type": "string",
                        "enum": ["big", "regular", "small", "tiny"],
                    },
                    "optimiser": {"type": "string", "enum": ["adam", "sgd", "rmsprop"]},
                    "doubleWeights": {"type": "boolean"},
                    "conductanceDrifting": {"type": "boolean"},
                    "gOff": {"type": "number", "minimum": 0},
                    "gOn": {"type": "number", "minimum": 0},
                    "kV": {"type": "number", "minimum": 0},
                    "discretisation": {"type": "boolean"},
                    "numberConductanceLevels": {"type": "integer", "minimum": 0},
                    "excludedWeightsProportion": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "numberHiddenLayers": {"type": "integer", "enum": [1, 2, 3, 4]},
                    "nonidealitiesAfterTraining": {"type": "integer", "minimum": 1},
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
                    "noiseVariance": {"type": "number", "minimum": 0},
                    "numberSimulations": {"type": "integer", "minimum": 1},
                    "epochs": {"type": "integer", "minimum": 1},
                },
                "required": [
                    "modelSize",
                    "optimiser",
                    "doubleWeights",
                    "conductanceDrifting",
                    "discretisation",
                    "gOff",
                    "gOn",
                    "kV",
                    "numberHiddenLayers",
                    "nonidealities",
                    "noiseVariance",
                    "numberSimulations",
                ],
                "allOf": [
                    {
                        "if": {"properties": {"conductanceDrifting": {"const": True}}},
                        "then": {"properties": {"discretisation": {"const": False}}},
                    },
                    {
                        "if": {"properties": {"discretisation": {"const": True}}},
                        "then": {
                            "properties": {"conductanceDrifting": {"const": False}},
                            "required": [
                                "numberConductanceLevels",
                                "excludedWeightsProportion",
                            ],
                        },
                    },
                ],
            },
        },
    },
    "required": ["simulations"],
}
