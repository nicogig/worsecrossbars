import glob
from datetime import datetime

class Logging:

    def __init__ (self, number_hidden_layers, fault_type, number_ANNs, number_simulations):
        log_number = 1
        for name in glob.glob(f"../../outputs/logs/spruce_faultType{fault_type}_{number_hidden_layers}HL-?.log"):
            log_number += 1
        self.file_object = open(f"../../outputs/logs/spruce_faultType{fault_type}_{number_hidden_layers}HL-{log_number}.log", "a")
        self.hidden_layers = number_hidden_layers
        self.fault_type = fault_type
        self.number_ANNs = number_ANNs
        self.number_simulations = number_simulations

    def write (self, string="", special=None):
        if special is "begin":
            self.file_object.write(f"----- Begin log {datetime.now().__str__()} -----\n\
            Attempting simulation with following parameters:\n\
            number_hidden_layers: {self.hidden_layers}\n\
            fault_type: {self.fault_type}\n\
            number_ANNs: {self.number_ANNs}\n\
            number_simulations: {self.number_simulations}\n\n")
        elif special is "end":
            self.file_object.write(f"[{datetime.now().strftime('%H:%M:%S')}] Saved accuracies to file. Ending.\n\
                ----- End log {datetime.now().__str__()} -----")
        else:
            self.file_object.write(f"[{datetime.now().strftime('%H:%M:%S')}] {string}\n")

    def close (self):
        self.file_object.close()
