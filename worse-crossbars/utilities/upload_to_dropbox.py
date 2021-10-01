import shutil
import os
import dropbox
import json
from dropbox import DropboxOAuth2FlowNoRedirect
import secrets

dropbox_secrets = {}
auth_checked = False

def authenticate():
    auth_flow = DropboxOAuth2FlowNoRedirect(secrets.APP_KEY, consumer_secret=secrets.APP_SECRET, token_access_type='offline',
                                         scope=['account_info.read', 'files.content.read', 'files.content.write'],)

    authorize_url = auth_flow.start()
    print("1. Go to: " + authorize_url)
    print("2. Click \"Allow\" (you might have to log in first).")
    print("3. Copy the authorization code.")
    auth_code = input("Enter the authorization code here: ").strip()

    try:
        oauth_result = auth_flow.finish(auth_code)
    except Exception as e:
        print('Error: %s' % (e,))
        exit(1)

    with dropbox.Dropbox(oauth2_access_token=oauth_result.access_token) as dbx:
        dbx.users_get_current_account()
        print("Successfully set up client!")

    secret = {"dropbox_auth_code": oauth_result.access_token,
              "dropbox_expiration": oauth_result.expires_at.__str__(),
              "dropbox_refresh":oauth_result.refresh_token}

    with open("./config/user_secrets.json", 'w') as outfile:
        json.dump(secret, outfile)

def check_auth_presence ():
    global auth_key, auth_checked, dropbox_secrets
    if not os.path.exists("./config/user_secrets.json"):
        authenticate()
        check_auth_presence()
    else:
        with open("./config/user_secrets.json") as json_file:
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
        