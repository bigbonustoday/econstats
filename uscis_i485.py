import glob, os
import pandas as pd


def read_all_files():
    data = {}
    os.chdir("d:\\Data\\uscis_i485")
    for file in glob.glob("*.csv"):
        print(file)
        date_label = file.split('_')[2] + file.split('_')[3]
        data[date_label] = pd.read_csv(file, encoding = "ISO-8859-1")

    # find NYC row
    row = data['fy2015qtr1.csv']['Unnamed: 2'].str.find('NYC').idxmax()
    df = pd.DataFrame({
        label: data[label].ix[data[label]['Unnamed: 2'].str.find('NYC').idxmax()] for label in data.keys()
    })
