import shutil
import os
import sys
import dropbox
import json

from worsecrossbars import configs

class DropboxUpload:
    def __init__(self, folder) -> None:
        if not os.path.exists(configs.working_dir.joinpath("config", "user_secrets.json")):
            print("Please run this module with --setup before using Internet options!")
            sys.exit(0)
        else:
            self.output_folder = folder
            with open(str(configs.working_dir.joinpath("config", "user_secrets.json"))) as json_file:
                self.dropbox_secrets = json.load(json_file)
                self.auth_checked = True
            with open(str(configs.working_dir.joinpath("config", "app_keys.json"))) as json_file:
                self.app_keys = json.load(json_file)

    def upload(self):
        if self.auth_checked:
            dbx = dropbox.Dropbox(oauth2_refresh_token=self.dropbox_secrets["dropbox_refresh"], app_key=self.app_keys["APP_KEY"], app_secret=self.app_keys["APP_SECRET"])
            shutil.make_archive(f"output_{self.output_folder}", "zip", str(configs.working_dir.joinpath("outputs", self.output_folder)))
            with open(f"output_{self.output_folder}.zip", 'rb') as f:
                data = f.read()
            try:
                res = dbx.files_upload(
                        data, f"/output_{self.output_folder}.zip",
                        mute=True,
                        mode=dropbox.files.WriteMode('overwrite'))
            except dropbox.exceptions.ApiError as err:
                print('*** API error', err)
                return None
            print('uploaded as', res.name.encode('utf8'))
            os.remove(f"./output_{self.output_folder}.zip")


# def check_auth_presence ():
#     """

#     """
    
#     global auth_key, auth_checked, dropbox_secrets, app_keys
#     if not os.path.exists(configs.working_dir.joinpath("config", "user_secrets.json")):
#         print("Please run this module with --setup before using Internet options!")
#         sys.exit(0)
#     else:
#         with open(str(configs.working_dir.joinpath("config", "user_secrets.json"))) as json_file:
#             dropbox_secrets = json.load(json_file)
#             auth_checked = True
#         with open(str(configs.working_dir.joinpath("config", "app_keys.json"))) as json_file:
#             app_keys = json.load(json_file)

# def upload (output_folder):
#     """

#     """

#     if auth_checked:
#         dbx = dropbox.Dropbox(oauth2_refresh_token=dropbox_secrets["dropbox_refresh"], app_key=app_keys["APP_KEY"], app_secret=app_keys["APP_SECRET"])
#         shutil.make_archive(f"output_{output_folder}", "zip", str(configs.working_dir.joinpath("outputs", output_folder)))
#         with open(f"output_{output_folder}.zip", 'rb') as f:
#             data = f.read()
#         try:
#             res = dbx.files_upload(
#                     data, f"/output_{output_folder}.zip",
#                     mute=True,
#                     mode=dropbox.files.WriteMode('overwrite'))
#         except dropbox.exceptions.ApiError as err:
#             print('*** API error', err)
#             return None
#         print('uploaded as', res.name.encode('utf8'))
#         os.remove(f"./output_{output_folder}.zip")
#     else:
#         check_auth_presence()
#         upload(output_folder)
