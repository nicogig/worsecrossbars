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

    def __init__ (self, simulation_parameters, output_folder):
        self.simulation_parameters = simulation_parameters
        self.file_object = str(Path.home().joinpath("worsecrossbars",
                                "outputs", output_folder, "logs", "spruce.log"))

    def __call__ (self):
        pass

    def write (self, string="", special=None):
        """
        Writes a given string to the log.

        Args:
          string: The string to write.
          special: Enables a bypass of `string` to write specific recurring strings.
        """
        if special == "begin":
            write_string = f"----- Begin log {datetime.now().__str__()} -----\n" + \
                            "Attempting simulation with following parameters:\n" + \
                            f"{self.simulation_parameters}"
        elif special == "end":
            write_string = f"[{datetime.now().strftime('%H:%M:%S')}] Saved accuracies to file." + \
                    f" Ending.\n----- End log {datetime.now().__str__()} -----"
        elif special == "abruptend":
            write_string = f"[{datetime.now().strftime('%H:%M:%S')}] Abruptly Ending.\n"+ \
                    f"----- End log {datetime.now().__str__()} -----"
        else:
            write_string = f"[{datetime.now().strftime('%H:%M:%S')}] " + string
        with open(self.file_object, "a", encoding="utf8") as file:
            file.write(f"{write_string}\n")
