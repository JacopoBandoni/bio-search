import io
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from googleapiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools



file_id = '1gAY_NUUJSZKPqCLEDvCy0QpsB0xebMYB'
drive_service = discovery.build('drive', 'v3')

request = drive_service.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()
    print("Download %d%%." % int(status.progress() * 100))