import csv

INPUT_FILE = 'DataTugas2.csv'
OUTPUT_FILE = 'TebakanTugas2.csv'
OUTPUT_DETAIL_FILE = 'TebakanTugas2_detail.csv'

input_data = []
output_data = []

# Z-shaped membership functions
def zmf(x, a, b): 
    if x <= a:
        return 1
    elif x <= (a+b) / 2:
        return 1 - 2 * ((x-a) / (b-a))**2
    elif x <= b:
        return 2 * ((x-b) / (b-a))**2
    return 0

# S-shaped membership functions
def smf(x, a, b): 
    if x <= a:
        return 0
    elif x <= (a+b) / 2:
        return 2 * ((x-a) / (b-a))**2
    elif x <= b:
        return 1 - 2 * ((x-b) / (b-a))**2
    return 1

# Pi-shaped membership functions
def pimf(x, a, b, c, d):
    if x <= a:
        return 0
    elif x <= (a+b) / 2:
        return 2 * ((x-a) / (b-a))**2
    elif x <= b:
        return 1 - 2 * ((x-b) / (b-a))**2
    elif x <= c:
        return 1
    elif x <= (c+d) / 2:
        return 1 - 2 * ((x-c) / (d-c))**2
    elif x <= d:
        return 2 * ((x-d) / (d-c))**2
    return 0

# Membership functions (fuzzy)
def pendapatan_rendah(x):
    return zmf(x, 0.5, 1.5)

def pendapatan_sedang(x):
    return pimf(x, 0.5, 1, 1, 1.5)

def pendapatan_tinggi(x):
    return smf(x, 0.5, 1.5)

def hutang_sedikit(x):
    return zmf(x, 25, 75)

def hutang_sedang(x):
    return pimf(x, 25, 50, 50, 75)

def hutang_banyak(x):
    return smf(x, 25, 75)

# Output functions (crisp)
def ekonomi_buruk(pendapatan, hutang):
    return 36 * pendapatan - hutang 

def ekonomi_biasa(pendapatan, hutang):
    return 24 * pendapatan - hutang

def ekonomi_bagus(pendapatan, hutangx):
    return 12 * pendapatan - hutang

# Read from file
with open(INPUT_FILE) as csv_file:
    csv_reader = csv.DictReader(csv_file, skipinitialspace=True)
    for data in csv_reader:
        no = int(data['No'])
        pendapatan = float(data['Pendapatan'])
        hutang = float(data['Hutang'])
        input_data.append({'no': no, 'pendapatan': pendapatan, 'hutang': hutang})

# Fuzzy Inference (Sugeno)
result_data = []
for data in input_data:
    # Fuzzification
    fuzzify_result = {
        'pRendah': pendapatan_rendah(data['pendapatan']),
        'pSedang': pendapatan_sedang(data['pendapatan']),
        'pTinggi': pendapatan_tinggi(data['pendapatan']),
        'hSedikit': hutang_sedikit(data['hutang']),
        'hSedang': hutang_sedang(data['hutang']),
        'hBanyak': hutang_banyak(data['hutang'])
    }

    # Rule application with logic operators
    weight = {
        'w_eBuruk': max(
            min(fuzzify_result['pRendah'], fuzzify_result['hSedang']),
            min(fuzzify_result['pRendah'], fuzzify_result['hBanyak']),
            min(fuzzify_result['pSedang'], fuzzify_result['hBanyak'])
        ),
        'w_eBiasa': max(
            min(fuzzify_result['pRendah'], fuzzify_result['hSedikit']),
            min(fuzzify_result['pSedang'], fuzzify_result['hSedang']),
            min(fuzzify_result['pTinggi'], fuzzify_result['hBanyak']),
        ),
        'w_eBagus': max(
            min(fuzzify_result['pSedang'], fuzzify_result['hSedikit']),
            min(fuzzify_result['pTinggi'], fuzzify_result['hSedikit']),
            min(fuzzify_result['pTinggi'], fuzzify_result['hSedang'])
        )
    }
    eBuruk = weight['w_eBuruk'] * ekonomi_buruk(data['pendapatan'], data['hutang'])
    eBiasa = weight['w_eBiasa'] * ekonomi_biasa(data['pendapatan'], data['hutang'])
    eBagus = weight['w_eBagus'] * ekonomi_bagus(data['pendapatan'], data['hutang'])

    # Defuzzification
    total_weight = weight['w_eBuruk'] + weight['w_eBiasa'] + weight['w_eBagus']
    weighted_average = (eBuruk + eBiasa + eBagus) / total_weight

    # Append to result data
    result_data.append({
        'no': data['no'],
        'pendapatan': data['pendapatan'],
        'hutang': data['hutang'],
        'ekonomi': weighted_average
    })

result_data.sort(key = lambda x: x['ekonomi'])
output_data = map(lambda x: [x['no']], result_data[:20])

# Write to file (detailed info)
with open(OUTPUT_DETAIL_FILE, mode='w', newline='') as csv_file:
    field_names = [*result_data[0]]
    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()
    csv_writer.writerows(result_data)
print('Detailed output written to', OUTPUT_DETAIL_FILE)

# Write to file (requested info)
with open(OUTPUT_FILE, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(output_data)
print('Output written to', OUTPUT_FILE)