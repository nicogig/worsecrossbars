import shutil
import os
import dropbox
import auth_dropbox
import json
import secrets

dropbox_secrets = {}
auth_checked = False

def check_auth_presence ():
    global auth_key, auth_checked, dropbox_secrets
    if not os.path.exists("user_secrets.json"):
        auth_dropbox.authenticate()
        check_auth_presence()
    else:
        with open("user_secrets.json") as json_file:
            dropbox_secrets = json.load(json_file)
            auth_checked = True


def zip_output (fault_type, number_hidden_layers):
    shutil.make_archive(f"output_faultType{fault_type}_{number_hidden_layers}HL", "zip", "../../outputs")

def upload (fault_type, number_hidden_layers):
    if auth_checked:
        dbx = dropbox.Dropbox(oauth2_refresh_token=dropbox_secrets["dropbox_refresh"], app_key=secrets.APP_KEY, app_secret=secrets.APP_SECRET)
        zip_output(fault_type, number_hidden_layers)
        with open(f"output_faultType{fault_type}_{number_hidden_layers}HL.zip", 'rb') as f:
            data = f.read()
        try:
            res = dbx.files_upload(
                    data, f"/output_faultType{fault_type}_{number_hidden_layers}HL.zip",
                    mute=True)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err)
            return None
        print('uploaded as', res.name.encode('utf8'))
        os.remove(f"./output_faultType{fault_type}_{number_hidden_layers}HL.zip")
    else:
        check_auth_presence()
        upload(fault_type, number_hidden_layers)

