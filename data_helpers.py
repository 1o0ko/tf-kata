'''
Module with sample data
'''
import logging
import os
import shutil
import zipfile

import numpy as np
import xlrd

from six.moves import urllib

FIRE_THEFT_PATH = './data/fire_theft.xls'
TEXT8_PATH = './data/text8'
TEXT8_URL = 'http://mattmahoney.net/dc/text8'
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)


def fetch_text8(data_url=TEXT8_URL, data_path=TEXT8_PATH):
    '''
    Downloads and unzips text8 data
    '''
    logger = logging.getLogger(__name__)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if os.path.exists(os.path.join(TEXT8_PATH, 'wiki.txt')):
        logging.info("File already exists")
        return

    zip_path = os.path.join(data_path, "text8.zip")
    logger.info("Downloading file...")
    urllib.request.urlretrieve(data_url, zip_path)
    logger.info("File downloaded.")

    logger.info("Extracting file.")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TEXT8_PATH)
    logger.info("File extracted.")

    # rename the file
    shutil.move(os.path.join(TEXT8_PATH, 'text8'),
                os.path.join(TEXT8_PATH, 'wiki.txt'))

    # clean-up
    os.remove(zip_path)


def load_fire_theft(file_path=FIRE_THEFT_PATH):
    '''
    Loads fire and theft data
    '''
    book = xlrd.open_workbook(file_path, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1
    x_data = data[:, 0].reshape(n_samples, 1)
    y_data = data[:, 1].reshape(n_samples, 1)

    return x_data, y_data


if __name__ == '__main__':
    fetch_text8()
