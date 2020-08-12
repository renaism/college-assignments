import csv
import os
from random import shuffle
from pprint import pprint

# File name constant
DATA_TEST_FILE = 'DataTest_Tugas3_AI.csv'
DATA_TRAIN_FILE = 'DataTrain_Tugas3_AI.csv'
DATA_TRAIN_FILE_SHUFFLED = 'DataTrain_Tugas3_AI_Shuffled.csv'
OUTPUT_FILE = 'TebakanTugas3.csv'
OUTPUT_FILE_DETAIL = 'TebakanTugas3_Detailed.csv'

# Read from csv file, return a list of dicts
def read_file(f, y=False):
    data = []
    with open(f) as csv_file:
        csv_reader = csv.DictReader(csv_file, skipinitialspace=True)
        for row in csv_reader:
            data.append({
                'Index': int(row['Index']),
                'X1': float(row['X1']),
                'X2': float(row['X2']),
                'X3': float(row['X3']),
                'X4': float(row['X4']),
                'X5': float(row['X5']),
                'Y': int(row['Y']) if y else row['Y']
            })
    return data

# Write to csv file from a list of dicts (rows of class)
def write_file(f, data):
    with open(f, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)

# Write to csv file from a list of dicts (complete output)
def write_file_detail(f, data):
    with open(f, mode='w', newline='') as csv_file:
        field_names = [*data[0]]
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()
        csv_writer.writerows(data)

# Calculate the estimate distance of two points (Manhattan)
def distance(p1, p2):
    d1 = abs(p1['X1'] - p2['X1'])
    d2 = abs(p1['X2'] - p2['X2'])
    d3 = abs(p1['X3'] - p2['X3'])
    d4 = abs(p1['X4'] - p2['X4'])
    d5 = abs(p1['X5'] - p2['X5'])
    return d1 + d2 + d3 + d4 + d5

# Predict data class based on the k-nearest neighbours class
def predict_class(point, data, k):
    distance_list = [{'distance': float('inf')}]
    for d in data:
        dist = distance(point, d)
        if dist < distance_list[-1]['distance']:
            if len(distance_list) >= k:
                distance_list.pop()
            i = 0
            while i < len(distance_list)-1 and dist >= distance_list[i]['distance']:
                i = i + 1
            distance_list.insert(i, {'distance': dist, 'Y': d['Y']})
    y_list = list(map(lambda x: x['Y'], distance_list))
    return max(y_list, key=y_list.count)

# Shuffle data and save to new file
def shuffle_train_data():
    data = read_file(DATA_TRAIN_FILE, y=True)
    shuffle(data)
    write_file_detail(DATA_TRAIN_FILE_SHUFFLED, data)
    print('Shuffled training data written to "{}"'.format(DATA_TRAIN_FILE_SHUFFLED))

# Test the error rate of K value
def cross_validate(start_k, end_k, ratio=0.25, shuffle_data=True):
    if shuffle_data:
        if not os.path.isfile(DATA_TRAIN_FILE_SHUFFLED):
            shuffle_train_data()
        data_train = read_file(DATA_TRAIN_FILE_SHUFFLED, y=True)
    else:
        data_train = read_file(DATA_TRAIN_FILE, y=True)
    slice_point = int(ratio * len(data_train))
    data_A = data_train[:slice_point]
    data_B = data_train[slice_point:]
    print("---Error Rate---")
    for k in range(start_k, end_k+1):
        error = 0
        for d_A in data_A:
            p_Y = predict_class(d_A, data_B, k)
            if p_Y != d_A['Y']:
                error = error + 1
        print('k={0:2d} | {2} / {3} | {1:.3f}%'.format(k, 100*error/slice_point, error, slice_point))

# Predict classes in DataTest based on DataTrain
def classify_data(k):
    data_test = read_file(DATA_TEST_FILE)
    data_train = read_file(DATA_TRAIN_FILE, y=True)
    for d_test in data_test:
        d_test['Y'] = predict_class(d_test, data_train, k)
    write_file(OUTPUT_FILE, map(lambda x: [x['Y']], data_test))
    print('Classification written to "{}"'.format(OUTPUT_FILE))
    write_file_detail(OUTPUT_FILE_DETAIL, data_test)
    print('Detailed information written to "{}"'.format(OUTPUT_FILE_DETAIL))

if __name__ == '__main__':
    classify_data(int(input('K: ')))