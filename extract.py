import csv

# Read input CSV file and prepare output CSV data
with open('input.csv', 'r') as infile, open('output.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=';')
    writer = csv.writer(outfile, delimiter=';')
    # Iterate through each row in the input CSV and process the columns
    for row in reader:
        # Extract the last and second last columns
        last_column = row[-2].strip()
        second_last_column = row[-3].strip()

        # Merge the last and second last columns into one and write to output CSV
        #merged_column = f'{last_column} {second_last_column}'
        writer.writerow([last_column])
        writer.writerow([second_last_column])

print("Output CSV file created successfully.")
