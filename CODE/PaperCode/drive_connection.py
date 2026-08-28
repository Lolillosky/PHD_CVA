import os
import httplib2
import oauth2client.transport as transport

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pathlib import Path


def connect_drive(use_work_proxy=False):
    if use_work_proxy:
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:8999"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8999"

        proxy_info = httplib2.proxy_info_from_environment("https")

        def get_http_object(*args, **kwargs):
            return httplib2.Http(proxy_info=proxy_info)

        transport.get_http_object = get_http_object

    gauth = GoogleAuth(
        settings_file="../../CONFIG/settings.yaml"
    )

    gauth.LoadCredentialsFile("../../CONFIG/credentials.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("../../CONFIG/credentials.json")

    return GoogleDrive(gauth)


def upload_folder_to_drive(
    drive,
    local_folder,
    drive_folder_id,
    recursive=False,
):
    """
    Upload all files from a local folder to a Google Drive folder.

    Parameters
    ----------
    drive : GoogleDrive
        Authenticated PyDrive2 GoogleDrive object.
    local_folder : str or Path
        Local folder containing the files to upload.
    drive_folder_id : str
        Destination Google Drive folder ID.
    recursive : bool
        If True, also uploads files inside subfolders.

    Returns
    -------
    list
        List of tuples: (local_path, drive_file_id)
    """

    local_folder = Path(local_folder)

    if not local_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {local_folder}")

    if not local_folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {local_folder}")

    if recursive:
        files = [p for p in local_folder.rglob("*") if p.is_file()]
    else:
        files = [p for p in local_folder.iterdir() if p.is_file()]

    uploaded = []

    for path in files:
        print(f"Uploading {path.name}...")

        f = drive.CreateFile({
            "title": path.name,
            "parents": [{"id": drive_folder_id}],
        })

        f.SetContentFile(str(path))
        f.Upload()

        uploaded.append((path, f["id"]))

    return uploaded


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def download_drive_folder(
    drive,
    drive_folder_id,
    local_folder,
):
    """
    Recursively download a Google Drive folder to a local folder,
    preserving the Drive subfolder structure.

    Parameters
    ----------
    drive : GoogleDrive
        Authenticated PyDrive2 GoogleDrive object.
    drive_folder_id : str
        ID of the Google Drive folder to download.
    local_folder : str or Path
        Local destination folder.
    """

    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)

    items = drive.ListFile({
        "q": (
            f"'{drive_folder_id}' in parents "
            "and trashed=false"
        )
    }).GetList()

    for item in items:
        name = item["title"]

        if item["mimeType"] == FOLDER_MIME_TYPE:
            # Recreate subfolder locally
            subfolder = local_folder / name
            subfolder.mkdir(parents=True, exist_ok=True)

            print(f"Entering folder: {name}")

            download_drive_folder(
                drive=drive,
                drive_folder_id=item["id"],
                local_folder=subfolder,
            )

        else:
            local_path = local_folder / name

            print(f"Downloading: {local_path}")

            item.GetContentFile(str(local_path))