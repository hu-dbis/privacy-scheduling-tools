import csv
import json
import pickle


def write_csv_data(path, data, field_names):
    with open(path, "a") as file:
        writer = csv.DictWriter(file, field_names)
        writer.writeheader()
        writer.writerows(data)
        file.flush()
    return


def write_json_data(path, data):
    with open(path, 'a') as f:
        json.dump(data, f)
        f.flush()
    return


def pickle_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        f.flush()
    return


def load_pickled_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
