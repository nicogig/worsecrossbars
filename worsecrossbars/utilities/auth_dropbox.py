import os, json
from pathlib import Path
from dropbox import DropboxOAuth2FlowNoRedirect, Dropbox
from worsecrossbars.utilities import io_operations

def authenticate():
    """

    """
    if not os.path.exists(
        Path.home().joinpath("worsecrossbars", "config", "app_keys.json")):
        io_operations.store_dropbox_keys()
        authenticate()
        return
    else:
        with open(
            str(Path.home().joinpath("worsecrossbars", "config", "app_keys.json")),
            encoding="utf8") as json_file:
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

    with open(
        str(Path.home().joinpath("worsecrossbars", "config", "user_secrets.json")),
        'w',
        encoding="utf8") as outfile:
        json.dump(secret, outfile)
