"""
"""
import glob
from datetime import datetime
from worsecrossbars import configs

class Logging:

    def __init__ (self, extracted_json):
        """
        """
        log_number = 1
        self.number_hidden_layers = extracted_json["number_hidden_layers"]
        self.fault_type = extracted_json["fault_type"]
        self.number_anns = extracted_json["number_ANNs"]
        self.number_simulations = extracted_json["number_simulations"]
        for _ in glob.glob(str(configs.working_dir.joinpath("outputs", "logs", 
        f"spruce_faultType{self.fault_type}_{self.number_hidden_layers}HL-?.log"))):
            log_number += 1
        self.file_object = str(configs.working_dir.joinpath("outputs", "logs", 
        f"spruce_faultType{self.fault_type}_{self.number_hidden_layers}HL-{log_number}.log"))

    def write (self, string="", special=None):
        """
        """
        if special == "begin":
            write_string = f"----- Begin log {datetime.now().__str__()} -----\n" + \
                    f"Attempting simulation with following parameters:\nnumber_hidden_layers: {self.number_hidden_layers}" + \
                        f"\nfault_type: {self.fault_type}\nnumber_ANNs: {self.number_anns}\nnumber_simulations: {self.number_simulations}\n\n"
        elif special == "end":
            write_string = f"[{datetime.now().strftime('%H:%M:%S')}] Saved accuracies to file." + \
                    f" Ending.\n----- End log {datetime.now().__str__()} -----"
        elif special == "abruptend":
            write_string = f"[{datetime.now().strftime('%H:%M:%S')}] Abruptly Ending.\n"+ \
                    f"----- End log {datetime.now().__str__()} -----"
        else:
            write_string = string
        with open(self.file_object, "a", encoding="utf8") as file:
            file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {write_string}\n")
