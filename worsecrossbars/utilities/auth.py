import sys, os, json
from worsecrossbars import configs

def require_webhook ():
    """

    """

    global webhook_url
    webhook_url = input("Please enter the MS Teams Webhook URL: ")
    jsonified_webhook = {"msteams_webhook":webhook_url}
    with open(str(configs.working_dir.joinpath("config", "msteams.json")), 'w') as outfile:
        json.dump(jsonified_webhook, outfile)




def check_webhook_presence ():
    """

    """

    global webhook_present, webhook_url
    if not os.path.exists(configs.working_dir.joinpath("config", "msteams.json")):
        print("Please run this module with --setup before using Internet options!")
        sys.exit(0)
    else:
        with open(str(configs.working_dir.joinpath("config", "msteams.json"))) as json_file:
            webhook_url = json.load(json_file)["msteams_webhook"]
            webhook_present = True