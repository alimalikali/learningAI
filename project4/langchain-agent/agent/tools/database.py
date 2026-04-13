"""
Natural language to SQL tool for querying the company database.

CONCEPT: This is Text-to-SQL — one of the most powerful and practical
agent capabilities. The agent writes SQL based on natural language,
executes it, and returns results. Used in enterprise AI applications constantly.

IMPORTANT: Read-only queries only. Never allow INSERT/UPDATE/DELETE.
"""

import os
import sqlite3
from langchain_core.tools import tool
from google import genai
from google.genai import types


# Dangerous SQL keywords to block
BLOCKED_KEYWORDS = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]


def generate_sql(natural_language_query: str, api_key: str) -> str:
    """
    Use Gemini to convert natural language to SQL.

    Args:
        natural_language_query: The user's question
        api_key: Gemini API key

    Returns:
        str: SQL query
    """
    client = genai.Client(api_key=api_key)

    system_prompt = """You are a SQL expert. Convert the question to a SQLite SELECT query.
Return ONLY the SQL query, nothing else. No markdown, no explanation, no code blocks.
Only SELECT queries are allowed."""

    user_message = f"""Tables:
- employees(id, name, department, salary, hire_date, city)
- products(id, name, category, price, stock)
- sales(id, product_id, employee_id, amount, sale_date)

Question: {natural_language_query}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )

    return response.text.strip()


def execute_query(sql: str, db_path: str) -> list:
    """
    Execute SQL query and return results.

    Args:
        sql: SQL query to execute
        db_path: Path to SQLite database

    Returns:
        list: Query results as list of tuples
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    return columns, results


def format_results(columns: list, rows: list) -> str:
    """
    Format query results as a readable table.

    Args:
        columns: Column names
        rows: Data rows

    Returns:
        str: Formatted table string
    """
    if not rows:
        return "No results found."

    # Calculate column widths
    widths = [len(str(col)) for col in columns]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Build table
    header = " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
    separator = "-+-".join("-" * w for w in widths)
    data_rows = [
        " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
        for row in rows
    ]

    return f"{header}\n{separator}\n" + "\n".join(data_rows)


@tool
def query_database(natural_language_query: str) -> str:
    """
    Query the company database using natural language.
    The database has three tables:
    - employees(id, name, department, salary, hire_date, city)
    - products(id, name, category, price, stock)
    - sales(id, product_id, employee_id, amount, sale_date)
    Use this when asked about employees, salaries, products, sales data, or company information.
    Input should be a natural language question about the data.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        db_path = os.getenv("DB_PATH", "data/company.db")

        if not api_key:
            return "Database query failed: GEMINI_API_KEY not configured"

        # Check if database exists
        if not os.path.exists(db_path):
            return f"Database not found at {db_path}. Run 'python3 data/seed_db.py' first."

        # Generate SQL from natural language
        sql = generate_sql(natural_language_query, api_key)

        # Security check: block dangerous operations
        sql_upper = sql.upper()
        for keyword in BLOCKED_KEYWORDS:
            if keyword in sql_upper:
                return f"Error: {keyword} operations are not allowed. Only SELECT queries permitted."

        # Execute query
        columns, results = execute_query(sql, db_path)

        # Format and return results
        formatted = format_results(columns, results)

        return f"SQL: {sql}\n\n{formatted}"

    except sqlite3.Error as e:
        return f"Database error: {str(e)}"

    except Exception as e:
        return f"Query failed: {str(e)}"
