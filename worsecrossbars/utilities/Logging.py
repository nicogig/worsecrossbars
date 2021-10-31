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
        self.number_ANNs = extracted_json["number_ANNs"]
        self.number_simulations = extracted_json["number_simulations"]
        for _ in glob.glob(str(configs.working_dir.joinpath("outputs", "logs", f"spruce_faultType{self.fault_type}_{self.number_hidden_layers}HL-?.log"))):
            log_number += 1
        self.file_object = str(configs.working_dir.joinpath("outputs", "logs", f"spruce_faultType{self.fault_type}_{self.number_hidden_layers}HL-{log_number}.log"))

    def write (self, string="", special=None):
        """
        
        """
        if special == "begin":
            with open(self.file_object, 'a') as file:
                file.write(f"----- Begin log {datetime.now().__str__()} -----\nAttempting simulation with following parameters:\nnumber_hidden_layers: {self.number_hidden_layers}\nfault_type: {self.fault_type}\nnumber_ANNs: {self.number_ANNs}\nnumber_simulations: {self.number_simulations}\n\n")
        elif special == "end":
            with open(self.file_object, 'a') as file:
                file.write(f"[{datetime.now().strftime('%H:%M:%S')}] Saved accuracies to file. Ending.\n----- End log {datetime.now().__str__()} -----")
        elif special == "abruptend":
            with open(self.file_object, 'a') as file:
                file.write(f"[{datetime.now().strftime('%H:%M:%S')}] Abruptly Ending.\n----- End log {datetime.now().__str__()} -----")
        else:
            with open(self.file_object, 'a') as file:
                file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {string}\n")
