import requests

from pathlib import Path
from typing import List, Union


class File:
    def __init__(self) -> None:
        self.filename = None
        self.filepath = None

    @staticmethod
    def process_remote_url(url: str) -> Path:
        """Download the remote file to the current directory and return its full path."""
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the download was successful

        # Use the file name from the URL, defaulting if not available
        filename = url.split("/")[-1] or "downloaded_file"
        file_path = Path(filename)

        # Save the file content
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return file_path.resolve()  # Return the absolute path


class InputFile(File):
    def __init__(self, file):

        self.remote_url = None
        self.local_file = None
        self.other_task_file = None

        # File will be downloaded
        if file.startswith('https') or file.startswith('http'):
            self.remote_url = file
        # File is local and will be staged in
        elif '/' in file:
            self.local_file = file
        # File produced from another task
        else:
            self.other_task_file = file

        if self.remote_url:
            # the default URL path would be the task sandbox
            # FIXME: maybe instead of downloading it and then stage it
            # inject a download command to the task.pre_exec?
            self.filepath = self.process_remote_url(self.remote_url)

        elif self.local_file and Path(self.local_file).exists():
            # local file to stage in to the task sandbox
            self.filepath = Path(self.local_file).resolve()  # Convert to absolute path
        
        elif self.other_task_file:
            self.filepath = Path(self.other_task_file)

        else:
            raise Exception(f'Failed to find/resolve InputFile: {file}')

        if not self.filepath:
            raise Exception('Failed to obtain the input file localy or remotley')

        self.filename = self.filepath.name


class OutputFile(File):
    def __init__(self, filename):
        self.filename = filename

        if '/' in self.filename:
            self.filename = self.filename.split('/')[-1]
