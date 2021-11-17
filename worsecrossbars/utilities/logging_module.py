"""
logging_module:
An internal module used to create and write to a log file.
"""
from datetime import datetime
from pathlib import Path


class Logging:
    """
    A class holding the name and location of the log file.

    Args:
      simulation_parameters: The JSON dictionary object containing the simulation parameters.
      output_folder: The output folder of the module.
    """

    def __init__(self, **kwargs):
        self.simulation_parameters = kwargs["simulation_parameters"]
        fault_type = self.simulation_parameters["fault_type"]
        number_hidden_layers = self.simulation_parameters["number_hidden_layers"]
        noise_variance = self.simulation_parameters["noise_variance"]
        self.file_object = str(
            Path.home().joinpath(
                "worsecrossbars",
                "outputs",
                kwargs["output_folder"],
                "logs",
                f"spruce_{fault_type}"
                + f"_{number_hidden_layers}HL"
                + f"_{noise_variance}NV.log",
            )
        )

    def __call__(self):
        pass

    def write(self, string="", special=None):
        """
        Writes a given string to the log.

        Args:
          string: The string to write.
          special: Enables a bypass of `string` to write specific recurring strings.
        """

        if special == "begin":
            write_string = (
                f"----- Begin log {datetime.now().__str__()} -----\n"
                + "Attempting simulation with following parameters:\n"
                + f"{self.simulation_parameters}"
            )
        elif special == "end":
            write_string = f"[{datetime.now().strftime('%H:%M:%S')}] Saved accuracies to file. Ending.\n----- End log {datetime.now().__str__()} -----"
        elif special == "abruptend":
            write_string = (
                f"[{datetime.now().strftime('%H:%M:%S')}] Abruptly Ending.\n"
                + f"----- End log {datetime.now().__str__()} -----"
            )
        else:
            write_string = f"[{datetime.now().strftime('%H:%M:%S')}] " + string
        with open(self.file_object, "a", encoding="utf8") as file:
            file.write(f"{write_string}\n")
