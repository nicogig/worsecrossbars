"""json_schemas:
A module containing the JSON schemas used for plotting and simulation JSONs.
"""

plot_schema = {
    "type": "object",
    "properties": {
        "plots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "key_features": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

simulation_schema = {
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
                    "double_weights": {"type": "boolean"},
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
                    "nonidealities_after_training": {"type": "integer", "minimum": 1},
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
                    "epochs": {"type": "integer", "minimum": 1},
                },
                "required": [
                    "model_size",
                    "optimiser",
                    "double_weights",
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
                        "then": {
                            "properties": {"conductance_drifting": {"const": False}},
                            "required": [
                                "number_conductance_levels",
                                "excluded_weights_proportion",
                            ],
                        },
                    },
                ],
            },
        },
    },
    "required": ["simulations"],
}
