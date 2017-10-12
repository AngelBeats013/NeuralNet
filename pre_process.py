import sys
import pandas as pd

if len(sys.argv) != 3:
    print('2 arguments required. Got: %s' % (len(sys.argv) - 1))
    exit(1)

raw_data_path = sys.argv[1]
output_path = sys.argv[2]
print('Using raw data from: %s' % raw_data_path)
print('Output data path: %s' % output_path)

def car_pre_process(raw_data_path):
    df = pd.read_csv(raw_data_path, header=None)
    df.columns = ['buy', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'cls']

    df.buy = df.buy.map({'vhigh': 0.0, 'high': 0.3, 'med': 0.6, 'low': 0.9})
    df.maint = df.maint.map({'vhigh': 0.0, 'high': 0.3, 'med': 0.6, 'low': 0.9})
    df.doors = df.doors.map({'2': 0.0, '3': 0.3, '4': 0.6, '5more': 0.9})
    df.persons = df.persons.map({'2': 0.0, '4': 0.5, 'more': 1.0})
    df.lug_boot = df.lug_boot.map({'small': 0.0, 'med': 0.5, 'big': 1.0})
    df.safety = df.safety.map({'low': 0.0, 'med': 0.5, 'high': 1.0})
    df.cls = df.cls.map({'unacc': 0.0, 'acc': 0.3, 'good': 0.6, 'vgood': 0.9})

    df.to_csv(output_path)


def iris_pre_process(raw_data_path):
    df = pd.read_csv(raw_data_path, header=None)
    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'cls']
    sepal_len_mean = df.sepal_len.mean()
    sepal_wid_mean = df.sepal_wid.mean()
    petal_len_mean = df.petal_len.mean()
    petal_wid_mean = df.petal_wid.mean()
    sepal_len_std = df.sepal_len.std()
    sepal_wid_std = df.sepal_wid.std()
    petal_len_std = df.petal_len.std()
    petal_wid_std = df.petal_wid.std()

    df.sepal_len = df.sepal_len.apply(lambda x: (x - sepal_len_mean) / sepal_len_std)
    df.sepal_wid = df.sepal_wid.apply(lambda x: (x - sepal_wid_mean) / sepal_wid_std)
    df.petal_len = df.petal_len.apply(lambda x: (x - petal_len_mean) / petal_len_std)
    df.petal_wid = df.petal_wid.apply(lambda x: (x - petal_wid_mean) / petal_wid_std)
    df.cls = df.cls.map({'Iris-setosa': 0.0, 'Iris-versicolor': 0.5, 'Iris-virginica': 1.0})

    df.to_csv(output_path)


def adult_pre_process(raw_data_path):

    pass

if 'car' in raw_data_path:
    car_pre_process(raw_data_path)
elif 'iris' in raw_data_path:
    iris_pre_process(raw_data_path)
elif 'adult' in raw_data_path:
    adult_pre_process(raw_data_path)
else:
    print('Unrecognized data set!')
