from sentence_transformers import SentenceTransformer
import csv

model = SentenceTransformer("BAAI/bge-m3")

# Read the sentence from a CSV file
with open('1000.csv', mode='r') as file:  # Update the file name as needed
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Get the header if there is one

    # Open the output CSV file for writing
    with open('1000embeddings.csv', mode='w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        # Write new header with original fields and embeddings
        csv_writer.writerow(header + ['Embeddings'])  # Add 'Embeddings' to the header

        for row in csv_reader:
            sentence = row[2]  # Assuming the sentence is in the first column
            embeddings = model.encode(sentence)  # Update this line
            # Convert embeddings to a string representation
            embeddings_str = ','.join(map(str, embeddings.tolist()))  # Convert to string
            # Write original fields and embeddings to the output CSV
            csv_writer.writerow(row + [embeddings_str])  # Append embeddings to the original row

