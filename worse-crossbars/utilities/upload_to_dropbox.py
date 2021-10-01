import secrets
import shutil
import os
from os.path import basename

def zipOutput ():
    shutil.make_archive("test", "zip", "../../outputs")
