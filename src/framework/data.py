import os
import requests
import radical.pilot as rp

from pathlib import Path
from typing import List, Union


class File:
    def __init__(self) -> None:
        pass

    @staticmethod
    def process_remote_url(self, url: str) -> Path:
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


class XInputFile(File):
    def __new__(cls, urls: List[Union[os.PathLike, str]]) -> dict:
        
        input_files = []

        # Process each URL
        for url in urls:
            # remote file
            if url.startswith('https') or url.startswith('http'):
                file = cls.process_remote_url(url)

            # local file - get absolute path
            elif Path(url).exists():
                file = Path(url).resolve()  # Convert to absolute path

            # not supported
            else:
                raise TypeError('Unrecognized file type, make sure your file is remote (https/http) or local')

            if file:
                # Populate the stage_in dictionary with full path
                stage_in= {'source': str(file), 'target': f'task://{file.name}', 'action': rp.TRANSFER}
                input_files.append(stage_in)
            else:
                raise Exception('Could not resolve file, please check your file')
        
        return input_files  # Return the list directly instead of an instance


class XOutputFile(File):
    def __new__(cls, urls: List[Union[os.PathLike, str]], target_task_sandbox) -> dict:

        output_files = []
        for url in urls:
            stage_in= {'source': f'task://{url}', 'target': f'{target_task_sandbox}/{url}', 'action': rp.TRANSFER}
            output_files.append(stage_in)


class InputFile:
    def __init__(self, filename):
        self.filename = filename

class OutputFile:
    def __init__(self, filename):
        self.filename = filename


