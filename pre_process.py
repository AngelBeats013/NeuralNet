import sys
import pandas as pd
import numpy as np

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
    df = pd.read_csv(raw_data_path, header=None, skiprows=1 if open(raw_data_path, 'r').read(1) == '|' else 0)
    df.columns = ['age', 'work_class', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                  'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                  'cls']
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x == '?' else x)
    df = df.dropna(axis=0, how='any')

    age_mean = df.age.mean()
    age_std = df.age.std()
    fnlwgt_mean = df.fnlwgt.mean()
    fnlwgt_std = df.fnlwgt.std()
    education_num_mean = df.education_num.mean()
    education_num_std = df.education_num.std()
    capital_gain_mean = df.capital_gain.mean()
    capital_gain_std = df.capital_gain.std()
    capital_loss_mean = df.capital_loss.mean()
    capital_loss_std = df.capital_loss.std()
    hours_per_week_mean = df.hours_per_week.mean()
    hours_per_week_std = df.hours_per_week.std()

    df.age = df.age.apply(lambda x: (x - age_mean) / age_std)
    df.fnlwgt = df.fnlwgt.apply(lambda x: (x - fnlwgt_mean) / fnlwgt_std)
    df.education_num = df.education_num.apply(lambda x: (x - education_num_mean) / education_num_std)
    df.capital_gain = df.capital_gain.apply(lambda x: (x - capital_gain_mean) / capital_gain_std)
    df.capital_loss = df.capital_loss.apply(lambda x: (x - capital_loss_mean) / capital_loss_std)
    df.hours_per_week = df.hours_per_week.apply(lambda x: (x - hours_per_week_mean) / hours_per_week_std)

    df.work_class = df.work_class.map({ 'Private': -1.0, 'Self-emp-not-inc': -0.75, 'Self-emp-inc': -0.5,
                                        'Federal-gov': -0.25, 'Local-gov': 0, 'State-gov': 0.25, 'Without-pay': 0.5,
                                        'Never-worked': 0.75 })
    df.education = df.education.map({ 'Bachelors': -1.0, 'Some-college': -0.875, '11th': -0.75, 'HS-grad': -0.625,
                                      'Prof-school': -0.5, 'Assoc-acdm': -0.375, 'Assoc-voc': -0.25, '9th': -0.125,
                                      '7th-8th': 0.0, '12th': 0.125, 'Masters': 0.25, '1st-4th': 0.375, '10th': 0.5,
                                      'Doctorate': 0.625, '5th-6th': 0.75, 'Preschool': 0.875})
    df.marital_status = df.marital_status.map({ 'Married-civ-spouse': -0.9, 'Divorced': -0.6, 'Never-married': -0.3,
                                                'Separated': 0.0, 'Widowed': 0.3, 'Married-spouse-absent': 0.6,
                                                'Married-AF-spouse': 0.9 })
    df.occupation = df.occupation.map({ 'Tech-support': -0.75, 'Craft-repair': -0.625, 'Other-service': -0.5,
                                        'Sales': -0.375, 'Exec-managerial': -0.25, 'Prof-specialty': -0.125,
                                        'Handlers-cleaners': 0.0, 'Machine-op-inspct': 0.125, 'Adm-clerical': 0.25,
                                        'Farming-fishing': 0.375, 'Transport-moving': 0.5, 'Priv-house-serv': 0.625,
                                        'Protective-serv': 0.75, 'Armed-Forces': 0.825 })
    df.relationship = df.relationship.map({ 'Wife': -0.9, 'Own-child': -0.6, 'Husband': -0.3, 'Not-in-family': 0,
                                            'Other-relative': 0.3, 'Unmarried': 0.6 })
    df.race = df.race.map({ 'White': -1.0, 'Asian-Pac-Islander': -0.5, 'Amer-Indian-Eskimo': 0.0, 'Other': 0.5,
                           'Black': 1.0 })
    df.sex = df.sex.map({ 'Female': 0.0, 'Male': 1.0 })
    df.native_country = df.native_country.map({ 'United-States': -1.0, 'Cambodia': -0.95, 'England': -0.9,
                                                'Puerto-Rico': -0.85, 'Canada': -0.8, 'Germany': -0.75,
                                                'Outlying-US(Guam-USVI-etc)': -0.7, 'India': -0.65, 'Japan': -0.6,
                                                'Greece': -0.55, 'South': -0.5, 'China': -0.45, 'Cuba': -0.4,
                                                'Iran': -0.35, 'Honduras': -0.3, 'Philippines': -0.25, 'Italy': -0.2,
                                                'Poland': -0.15, 'Jamaica': -0.1, 'Vietnam': -0.05, 'Mexico': 0.00,
                                                'Portugal': 0.05, 'Ireland': 0.1, 'France': 0.15,
                                                'Dominican-Republic': 0.2, 'Laos': 0.25, 'Ecuador': 0.3, 'Taiwan': 0.35,
                                                'Haiti': 0.4, 'Columbia': 0.45, 'Hungary': 0.5, 'Guatemala': 0.55,
                                                'Nicaragua': 0.6, 'Scotland': 0.65, 'Thailand': 0.7, 'Yugoslavia': 0.75,
                                                'El-Salvador': 0.8, 'Trinadad&Tobago': 0.85, 'Peru': 0.9, 'Hong': 0.95,
                                                'Holand-Netherlands': 1.0 })
    df.cls = df.cls.apply(lambda x: x.replace('.', '') if isinstance(x, str) else x)
    df.cls = df.cls.map({ '<=50K': 0.0, '>50K': 1.0 })
    df.to_csv(output_path)

if 'car' in raw_data_path:
    car_pre_process(raw_data_path)
elif 'iris' in raw_data_path:
    iris_pre_process(raw_data_path)
elif 'adult' in raw_data_path:
    adult_pre_process(raw_data_path)
else:
    print('Unrecognized data set!')
