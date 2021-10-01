import glob

class Logging:

    def __init__ (self, number_hidden_layers, fault_type):
        log_number = 1
        for name in glob.glob(f"../../outputs/logs/spruce_faultType{fault_type}_{number_hidden_layers}HL-?.log"):
            log_number += 1
        self.file_object = open(f"../../outputs/logs/spruce_faultType{fault_type}_{number_hidden_layers}HL-{log_number}.log", "a")

    def write (self, string):
        self.file_object.write(string)

    def close (self):
        self.file_object.close()
