"""
parameter_validator:
A utilities module used to assess the validity of the simulation parameters loaded from JSON dict.
"""


def validate_parameters(simulation_parameters):
    """
    """

    if not isinstance(simulation_parameters, dict):
        raise ValueError("\"simulation_parameters\" argument should be a dictionary object.")

    if simulation_parameters["fault_type"] not in ["STUCK_ZERO", "STUCK_HRS", "STUCK_LRS"]:
        raise ValueError("\"fault_type\" argument should be valid string. Acceptable values \
                         include STUCK_ZERO, STUCK_HRS and STUCK_LRS.")

    if simulation_parameters["number_hidden_layers"] not in [1, 2, 3, 4]:
        raise ValueError("\"number_hidden_layers\" argument should be an integer between 1 and 4.")