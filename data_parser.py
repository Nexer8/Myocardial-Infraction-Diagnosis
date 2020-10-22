import re
import numpy as np


class Data:
    def __init__(self, pain_of_non_heart_origin: list, angina_prectoris: list,
                 angina_prectoris_prinzmetal_variant: list,
                 myocardial_infraction_transmural: list, myocardial_infraction_subendocardial: list):
        self.pain_of_non_heart_origin = pain_of_non_heart_origin
        self.angina_prectoris = angina_prectoris
        self.angina_prectoris_prinzmetal_variant = angina_prectoris_prinzmetal_variant
        self.myocardial_infraction_transmural = myocardial_infraction_transmural
        self.myocardial_infraction_subendocardial = myocardial_infraction_subendocardial

    def print_classes_strength(self):
        print(f'Pain of non heart origin: {len(self.pain_of_non_heart_origin)}')
        print(f'Angina prectoris: {len(self.angina_prectoris)}')
        print(f'Angina prectoris - Prinzmetal variant: {len(self.angina_prectoris_prinzmetal_variant)}')
        print(f'Myocardial infraction (transmural): {len(self.myocardial_infraction_transmural)}')
        print(f'Myocardial infraction (subendocardial): {len(self.myocardial_infraction_subendocardial)}')


FILES_PATHS = {
    'data/inne.txt': 0,
    'data/ang_prect.txt': 1,
    'data/ang_prct_2.txt': 2,
    'data/mi.txt': 3,
    'data/mi_np.txt': 4
}


def load_and_transpose_file(path: str) -> np.array:
    return np.loadtxt(path, dtype=int).transpose()


def load_all_files() -> list:
    return [load_and_transpose_file(path) for path in FILES_PATHS]


def load_data(datasets: list) -> Data:
    return Data(pain_of_non_heart_origin=datasets[FILES_PATHS['data/inne.txt']],
                angina_prectoris=datasets[FILES_PATHS['data/ang_prect.txt']],
                angina_prectoris_prinzmetal_variant=datasets[FILES_PATHS['data/ang_prct_2.txt']],
                myocardial_infraction_transmural=datasets[FILES_PATHS['data/mi.txt']],
                myocardial_infraction_subendocardial=datasets[FILES_PATHS['data/mi_np.txt']])


def load_features_names() -> list:
    features_names = []

    with open('data/MYOCARDIAL INFRACTION DIAGNOSIS.txt') as file:
        for line in file.readlines():
            if line.__contains__('DIAGNOSES'):
                break
            if re.search(r'^(\d+)\.', line):
                line = line.split(' ', 1)[1]
                line = line.split('(')[0]
                line = line.split(':')[0]
                line = line.strip()
                features_names.append(line)

    return features_names
