from dropbox import DropboxOAuth2FlowNoRedirect, Dropbox
import json
import os

from worsecrossbars import configs

app_keys = {}

def obtain_keys():
    """

    """

    global app_keys

    app_key = input("Enter the APP KEY from your Dropbox App: ").strip()
    app_secret = input("Enter the APP SECRET from your Dropbox App: ").strip()
    app_keys = {"APP_KEY":app_key, "APP_SECRET":app_secret}

    with open(str(configs.working_dir.joinpath("config", "app_keys.json")), 'w') as outfile:
        json.dump(app_keys, outfile)



def authenticate():
    """

    """

    global app_keys

    if not os.path.exists(configs.working_dir.joinpath("config", "app_keys.json")):
        obtain_keys()
        authenticate()
        return
    else:
        with open(str(configs.working_dir.joinpath("config", "app_keys.json"))) as json_file:
            app_keys = json.load(json_file)

    
    auth_flow = DropboxOAuth2FlowNoRedirect(app_keys["APP_KEY"], consumer_secret=app_keys["APP_SECRET"], token_access_type='offline',
                                         scope=['account_info.read', 'files.content.read', 'files.content.write'],)

    authorize_url = auth_flow.start()
    print(f"1. Go to: {authorize_url}")
    print("2. Click \"Allow\" (you might have to log in first).")
    print("3. Copy the authorization code.")
    auth_code = input("Enter the authorization code here: ").strip()

    try:
        oauth_result = auth_flow.finish(auth_code)
    except Exception as e:
        print('Error: %s' % (e,))
        exit(1)

    with Dropbox(oauth2_access_token=oauth_result.access_token) as dbx:
        dbx.users_get_current_account()
        print("Successfully set up client!")

    secret = {"dropbox_auth_code": oauth_result.access_token,
              "dropbox_expiration": oauth_result.expires_at.__str__(),
              "dropbox_refresh":oauth_result.refresh_token}

    with open(str(configs.working_dir.joinpath("config", "user_secrets.json")), 'w') as outfile:
        json.dump(secret, outfile)
