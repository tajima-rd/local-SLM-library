import sys
from pathlib import Path
import sqlite3

# Assume core modules are in sys.path or accessible
# Adjust import based on your project structure if necessary
try:
    # Assuming this script is alongside the core/ and sample/ directories
    # Or adjust the path finding logic below if structure is different
    # For instance, if this script is inside modules/sample/scripts/
    # then adjust the parents calculation.
    from core import document_utils as du
    from core.objects import (
        database,
        categories,
        projects,
        documents,
        paragraphs
    )
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Please ensure the 'core' directory is accessible and contains database.py, document_utils.py, chain_factory.py, and retriever_utils.py.")
    print("Also ensure your Python environment is set up to find these modules.")
    sys.exit(1)

def _db_connect(db_path: str):
    """
    Connect to the SQLite database at the specified path.
    
    Parameters:
    - db_path: Path to the SQLite database file.
    
    Returns:
    - Connection object if successful, None otherwise.
    """
    try:
        conn = database.db_connect(db_path)
        if conn is None:
            print(f"❌ Error: Could not connect to the database at {db_path}.")
            print("Please ensure sample_03_staging.py has been run successfully to create the database.")
            sys.exit(1)
        print(f"✅ Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def load_project(db_path: str, project_name: str):
    conn = _db_connect(db_path)
    if conn is None:
        print("❌ Error: Could not establish a database connection.")
        sys.exit(1)
    print(f"✅ Successfully connected to the database at {db_path}")

    # Load project by name
    loaded_project = projects.load_project_by_name(db_path, project_name)
    if loaded_project is None:
        print(f"❌ Error: Project '{project_name}' not found in the database.")
        print("Please ensure the project exists and has been created by sample_03_staging.py.")
        sys.exit(1)
    print(f"✅ Loaded project: {loaded_project.name}")
    
    # Load documents and paragraphs for the project
    project_documents = documents.get_documents_by_project_id(conn, loaded_project.id)
    if not project_documents:
        print(f"❌ Error: No documents found for project '{project_name}'.")
        print("Please ensure the project has documents associated with it.")
        sys.exit(1)
    print(f"✅ Found {len(project_documents)} documents for project '{project_name}'.")

    project_paragraphs = []
    for doc in project_documents:
        project_paragraphs = paragraphs.get_paragraphs_by_document_id(conn, doc.id)  
        for project_paragraph in project_paragraphs:
            doc.paragraphs.append(project_paragraph) 

    return loaded_project, project_documents

def load_categories(db_path: str, project_name: str):
    conn = _db_connect(db_path)
    if conn is None:
        print("❌ Error: Could not establish a database connection.")
        sys.exit(1)
    print(f"✅ Successfully connected to the database at {db_path}")
    
    categories.Category.get_all_categories(conn)
