import pymsteams
import json
import os

webhook_url = ""
webhook_present = False

def require_webhook ():
    global webhook_url
    webhook_url = input("Please enter the MS Teams Webhook URL: ")
    jsonified_webhook = {"msteams_webhook":webhook_url}
    with open("./config/msteams.json", 'w') as outfile:
        json.dump(jsonified_webhook, outfile)

def check_webhook_presence ():
    global webhook_present, webhook_url
    if not os.path.exists("./config/msteams.json"):
        require_webhook()
        check_webhook_presence()
    else:
        with open("./config/msteams.json") as json_file:
            webhook_url = json.load(json_file)["msteams_webhook"]
            webhook_present = True

def send_message(message, title=None, color=None):
    if webhook_present:
        msteams_message = pymsteams.connectorcard(webhook_url)
        msteams_message.text(message)
        if title:
            msteams_message.title(title)
        if color:
            msteams_message.color(color)
        msteams_message.send() 