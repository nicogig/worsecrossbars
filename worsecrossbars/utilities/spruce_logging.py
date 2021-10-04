import glob
from datetime import datetime

from worsecrossbars import configs

class Logging:

    def __init__ (self, number_hidden_layers, fault_type, number_ANNs, number_simulations):
        """

        """

        log_number = 1
        for name in glob.glob(str(configs.working_dir.joinpath("outputs", "logs", f"spruce_faultType{fault_type}_{number_hidden_layers}HL-?.log"))):
            log_number += 1
        self.file_object = str(configs.working_dir.joinpath("outputs", "logs", f"spruce_faultType{fault_type}_{number_hidden_layers}HL-{log_number}.log"))
        self.hidden_layers = number_hidden_layers
        self.fault_type = fault_type
        self.number_ANNs = number_ANNs
        self.number_simulations = number_simulations



    def write (self, string="", special=None):
        """
        
        """

        if special == "begin":
            with open(self.file_object, 'a') as file:
                file.write(f"----- Begin log {datetime.now().__str__()} -----\nAttempting simulation with following parameters:\nnumber_hidden_layers: {self.hidden_layers}\nfault_type: {self.fault_type}\nnumber_ANNs: {self.number_ANNs}\nnumber_simulations: {self.number_simulations}\n\n")
        elif special == "end":
            with open(self.file_object, 'a') as file:
                file.write(f"[{datetime.now().strftime('%H:%M:%S')}] Saved accuracies to file. Ending.\n----- End log {datetime.now().__str__()} -----")
        elif special == "abruptend":
            with open(self.file_object, 'a') as file:
                file.write(f"[{datetime.now().strftime('%H:%M:%S')}] Abruptly Ending.\n----- End log {datetime.now().__str__()} -----")
        else:
            with open(self.file_object, 'a') as file:
                file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {string}\n")

    def close (self):
        # Archaic, here until the next code cleanup.
        return
