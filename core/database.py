import psycopg2
import os
from typing import Optional, List, Tuple
from dotenv import load_dotenv

load_dotenv()

# Database configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "csv")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

def get_db_connection():
    """
    Create and return a PostgreSQL database connection.
    """
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def create_chatbot_table():
    """
    Create the rc_chatbot table if it doesn't exist.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS rc_chatbot (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255) NOT NULL,
            role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'bot')),
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversation_id ON rc_chatbot(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_created_at ON rc_chatbot(created_at);
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        print("rc_chatbot table created successfully or already exists")
        
    except Exception as e:
        print(f"Error creating table: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def store_message(conversation_id: str, role: str, message: str) -> bool:
    """
    Store a message in the rc_chatbot table.
    
    Args:
        conversation_id: The conversation identifier
        role: Either 'user' or 'bot'
        message: The message content
        
    Returns:
        bool: True if successful, False otherwise
    """
    if role not in ['user', 'bot']:
        raise ValueError("Role must be either 'user' or 'bot'")
    
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO rc_chatbot (conversation_id, role, message)
        VALUES (%s, %s, %s)
        """
        
        cursor.execute(insert_query, (conversation_id, role, message))
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error storing message: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_conversation_history(conversation_id: str) -> str:
    """
    Retrieve conversation history for a given conversation_id and format it as a string.
    
    Args:
        conversation_id: The conversation identifier
        
    Returns:
        str: Formatted conversation history as "user: abc, bot: abc, user: ..." or empty string if no history
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        select_query = """
        SELECT role, message FROM rc_chatbot 
        WHERE conversation_id = %s 
        ORDER BY created_at ASC
        """
        
        cursor.execute(select_query, (conversation_id,))
        rows = cursor.fetchall()
        
        if not rows:
            return ""
        
        # Format as "user: message, bot: message, user: message, ..."
        formatted_history = []
        for role, message in rows:
            formatted_history.append(f"{role}: {message}")
        
        return ", ".join(formatted_history)
        
    except Exception as e:
        print(f"Error retrieving conversation history: {e}")
        return ""
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_conversation_messages(conversation_id: str) -> List[Tuple[str, str, str]]:
    """
    Retrieve conversation messages for a given conversation_id as a list of tuples.
    
    Args:
        conversation_id: The conversation identifier
        
    Returns:
        List[Tuple[str, str, str]]: List of (role, message, timestamp) tuples
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        select_query = """
        SELECT role, message, created_at FROM rc_chatbot 
        WHERE conversation_id = %s 
        ORDER BY created_at ASC
        """
        
        cursor.execute(select_query, (conversation_id,))
        rows = cursor.fetchall()
        
        return [(row[0], row[1], str(row[2])) for row in rows]
        
    except Exception as e:
        print(f"Error retrieving conversation messages: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Initialize the table when the module is imported
try:
    create_chatbot_table()
except Exception as e:
    print(f"Warning: Could not initialize rc_chatbot table: {e}")
