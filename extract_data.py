import pandas as pd 
import psycopg

# Define the connection parameters
dbname = 'hmisdb'
user = 'postgres'
password = 'thapa9860'
host = 'localhost'  # Usually 'localhost' if running locally
port = '5432'  # Usually 5432
branch_schema = 'branch_gnh'  # Replace with the name of the schema for the branch you want to access
table_name = 'customer_customer'
csv_filename = '/Users/sunilthapa/Desktop/My_projects/meraki/datas/customers.csv'


# Establish a connection to the PostgreSQL database
try:
    connection = psycopg.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    print("Connected to the database successfully!")
    
    # Create a cursor object
    cursor = connection.cursor()
    
    # Define the SQL query to select data from the table
    query = f"SELECT * FROM {branch_schema}.{table_name};"
    
    # Execute the SQL query
    cursor.execute(query)
    
    # Fetch all rows from the result set
    rows = cursor.fetchall()

    col_names = [desc[0] for desc in cursor.description]
    
    
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

    df = pd.DataFrame(rows,columns = col_names)

    df.to_csv(csv_filename, index=False)
    
    
except psycopg.Error as e:
    print("Error connecting to the database:", e)

