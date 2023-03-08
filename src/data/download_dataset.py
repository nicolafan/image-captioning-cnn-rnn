# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import kaggle


@click.command()
def main():
    """Download script for downloading the dataset (Flickr8k) from Kaggle"""
    logger = logging.getLogger(__name__)
    logger.info(
        "downloading Flickr8k dataset from kaggle, please wait until new log message"
    )

    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / "data" / "raw"

    kaggle.api.dataset_download_files(
        "adityajn105/flickr8k", path=raw_data_dir, force=True, quiet=True, unzip=True
    )

    logger.info("download finished")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
