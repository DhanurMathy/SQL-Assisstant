from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Setup database connection
def setup_database():
    # Database Credentials
    db_user = os.getenv('db_user')
    db_password = os.getenv('db_password')
    db_host = os.getenv('db_host')
    port = os.getenv('port')
    db_name = os.getenv('db_name')
    # Create SQLAlchemy connection string
    connection_string = (
        f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{port}/{db_name}'
    )

    try:
        # Create SQLDatabase instance
        db = SQLDatabase.from_uri(connection_string)
        return db
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return None