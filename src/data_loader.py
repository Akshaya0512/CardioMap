import pandas as pd


def load_data(ems_path, hospital_path):

    ems = pd.read_excel(ems_path)
    hospitals = pd.read_excel(hospital_path)

    print("EMS shape:", ems.shape)
    print("Hospitals shape:", hospitals.shape)

    return ems, hospitals
