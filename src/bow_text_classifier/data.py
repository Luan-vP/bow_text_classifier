from logging import Logger
from pathlib import Path

import requests

logger = Logger(__name__)

data_dir = Path(__file__).parent.parent.parent / "data"


def download_dataset(url: str, output_file: Path | str):
    output_file = Path(output_file)

    if not output_file.exists():
        logger.info(f"Downloading {url} to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url) as response:
            response.raise_for_status()

            with output_file.open("wb") as f:
                f.write(response.content)

    return output_file
