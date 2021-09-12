import numpy as np

def compute_output_currents(device_resistances, applied_voltages):
    """
    """

    # TODO: Make sure that inputs are good
    # TODO: At the moment, we assume all node voltages are known.

    voltage_matrix = np.zeros((2*device_resistances.size, applied_voltages.shape[1]))
    voltage_matrix[:device_resistances.size, ] = np.repeat(applied_voltages, device_resistances.shape[1], axis=0)
    word_line_voltages = distribute_array(voltage_matrix[:device_resistances.size, ], device_resistances)
    bit_line_voltages = distribute_array(voltage_matrix[device_resistances.size:, ], device_resistances)

    # Extracted voltages, proceeding to currents
    if word_line_voltages.ndim > 2:
        device_resistances_currents = np.repeat(device_resistances[:, :, np.newaxis],
                                                word_line_voltages.shape[2],
                                                axis=2)
    voltage_diff = word_line_voltages - bit_line_voltages
    device_currents = voltage_diff / device_resistances

    # Extracting output currents
    output_currents = np.sum(device_currents, axis=0)
    if output_currents.ndim == 1:
        output_currents = output_currents.reshape(1, output_currents.shape[0])
    return (word_line_voltages, bit_line_voltages, output_currents)



def distribute_array (flattened_array, model_array):
    """
    """
    
    reshaped_array = flattened_array.reshape((model_array.shape[0], model_array.shape[1], flattened_array.shape[1]))
    if reshaped_array.ndim == 3:
        if reshaped_array.shape[2] == 1:
            reshaped_array = np.squeeze(reshaped_array, axis=2)
    return reshaped_array
