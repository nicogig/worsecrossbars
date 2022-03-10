"""
logging_module:
An internal module used to create and write to a log file.
"""
from datetime import datetime
from pathlib import Path

import os


class Logging:
    """
    A class holding the name and location of the log file.

    Args:
      simulation_parameters: The JSON dictionary object containing the simulation parameters.
      output_folder: The output folder of the module.
    """

    def __init__ (self, output_folder):
        self.file_object = str(Path.home().joinpath("worsecrossbars", "outputs",
                                                    output_folder, "logs",
                                                    f"run_{os.getpid()}.log"))

    def __call__ (self):
        pass

    def write (self, string="", level="INFO"):
        """
        Writes a given string to the log.

        Args:
          string: The string to write.
          special: Enables a bypass of `string` to write specific recurring strings.
        """

        write_string = f"[{datetime.now().strftime('%H:%M:%S')}] [{os.getpid()}] [{level}] " + string
        with open(self.file_object, "a", encoding="utf8") as file:
            file.write(f"{write_string}\n")