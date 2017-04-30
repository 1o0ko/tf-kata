'''
Module with sample data
'''
import numpy as np
import xlrd


DATA_FILE = 'data/fire_theft.xls'


def load_fire_theft(file_path=DATA_FILE):
    '''
    Loads fire and theft data
    '''
    book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1
    x_data = data[:, 0].reshape(n_samples, 1)
    y_data = data[:, 1].reshape(n_samples, 1)

    return x_data, y_data
