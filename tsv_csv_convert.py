import csv

def tsv_to_csv(tsv_file_path, csv_file_path):

    with open(tsv_file_path, 'r', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            
            for row in tsv_reader:
                csv_writer.writerow(row)

tsv_file_path = 'test.tsv'
csv_file_path = 'new_test.csv'
tsv_to_csv(tsv_file_path, csv_file_path)