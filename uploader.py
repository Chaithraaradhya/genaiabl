import singlestoredb as s2
import csv
import numpy as np

# Function to create a connection to SingleStoreDB
def create_connection():
    return s2.connect('admin:X8MBbWxI1NiuG6RGPhyIQcr7lz4oseOY@svc-8bd4e6d7-dd92-449e-b8af-56828e3aea12-dml.aws-mumbai-1.svc.singlestore.com:3306/miniDB')
# admin:X8MBbWxI1NiuG6RGPhyIQcr7lz4oseOY@svc-8bd4e6d7-dd92-449e-b8af-56828e3aea12-dml.aws-mumbai-1.svc.singlestore.com:3306/
# Insert embeddings into SingleStoreDB
try:
    with create_connection() as conn:
        with conn.cursor() as cur:
            # Create a table to store embeddings if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS webgen (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    text TEXT,
                    llm_generated_idea TEXT,
                    embedding VECTOR(1024, F32) NOT NULL
                )
            """)

            # Read data from CSV and insert into the database
            with open('1000embeddings.csv', mode='r') as csvfile:  # Replace with your CSV file path
                csvreader = csv.reader(csvfile)
                next(csvreader)  # Skip the header row

                for row in csvreader:
                    text = row[1]
                    llm_generated_idea = row[2]
                    embedding = np.fromstring(row[3], sep=',').tolist()  # Convert string to numpy array and then to list

                    # Format embedding as a JSON array string
                    embedding_json = f"[{','.join(map(str, embedding))}]"

                    # Insert into the table
                    cur.execute("""
                        INSERT INTO webgen (text, llm_generated_idea, embedding) VALUES (%s, %s, %s)
                    """, (text, llm_generated_idea, embedding_json))  # Pass the JSON array string

            # Commit the transaction
            conn.commit()
            print("Insertion successful.")
except Exception as e:
    print(f"Insertion failed: {e}")
