"""
parameter_validator:
A utilities module used to assess the validity of the simulation parameters loaded from JSON dict.
"""


def validate_parameters(simulation_parameters):
    """
    This function validates the dictionary containing the simulation parameters, by checking whether
    the values provided are of the correct type, and take on legal values.

    Args:
      simulation_parameters: Dictionary containing simulation parameters to be validated.
    """

    if not isinstance(simulation_parameters, dict):
        raise ValueError("\"simulation_parameters\" argument should be a dictionary object.")

    hrs_lrs_ratio = simulation_parameters["HRS_LRS_ratio"]
    number_conductance_levels = simulation_parameters["number_conductance_levels"]
    excluded_weights_proportion = simulation_parameters["excluded_weights_proportion"]
    number_hidden_layers = simulation_parameters["number_hidden_layers"]
    fault_type = simulation_parameters["fault_type"]
    noise_variance = simulation_parameters["noise_variance"]
    number_anns = simulation_parameters["number_ANNs"]
    number_simulations = simulation_parameters["number_simulations"]

    if fault_type not in ["STUCK_ZERO", "STUCK_HRS", "STUCK_LRS"]:
        raise ValueError("\"fault_type\" argument should be valid string. Acceptable values " +
                         "include STUCK_ZERO, STUCK_HRS and STUCK_LRS.")

    if number_hidden_layers not in [1, 2, 3, 4]:
        raise ValueError("\"number_hidden_layers\" argument should be an integer between 1 and 4.")

    if not isinstance(number_anns, int) or number_anns < 1:
        raise ValueError("\"number_ANNs\" argument should be an integer greater than " +
                         "or equal to 1.")

    if not isinstance(number_simulations, int) or number_simulations < 1:
        raise ValueError("\"number_simulations\" argument should be an integer greater than " +
                         "or equal to 1.")

    if not isinstance(number_conductance_levels, int) or number_conductance_levels < 1:
        raise ValueError("\"number_conductance_levels\" argument should be an integer " +
                         "greater than or equal to 1.")

    if isinstance(noise_variance, int):
        noise_variance = float(noise_variance)
    if not isinstance(noise_variance, float) or noise_variance < 0:
        raise ValueError("\"noise_variance\" argument should be a positive real number.")

    if isinstance(hrs_lrs_ratio, int):
        hrs_lrs_ratio = float(hrs_lrs_ratio)
    if not isinstance(hrs_lrs_ratio, float) or hrs_lrs_ratio < 0:
        raise ValueError("\"hrs_lrs_ratio\" argument should be a positive real number.")

    if isinstance(excluded_weights_proportion, int):
        excluded_weights_proportion = float(excluded_weights_proportion)
    if not isinstance(excluded_weights_proportion, float) or excluded_weights_proportion < 0 or \
        excluded_weights_proportion > 1:
        raise ValueError("\"excluded_weights_proportion\" argument should be a " +
                         "real number between 0 and 1.")
