import dropbox
from dropbox import DropboxOAuth2FlowNoRedirect
from . import secret_keys
import json

def authenticate():
    auth_flow = DropboxOAuth2FlowNoRedirect(secret_keys.APP_KEY, consumer_secret=secret_keys.APP_SECRET, token_access_type='offline',
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
