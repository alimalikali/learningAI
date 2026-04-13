"""
Creates a SQLite database with realistic sample data.
Run this once before starting the agent: python3 data/seed_db.py
"""

import sqlite3
import os


def create_database():
    """Create and seed the company database."""

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    db_path = "data/company.db"

    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date TEXT NOT NULL,
            city TEXT NOT NULL
        )
    """)

    # Create products table
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        )
    """)

    # Create sales table
    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            employee_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            sale_date TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    """)

    # Seed employees (10 employees across 4 departments)
    employees = [
        (1, "Alice Chen", "Engineering", 125000, "2021-03-15", "San Francisco"),
        (2, "Bob Smith", "Engineering", 115000, "2022-01-10", "Seattle"),
        (3, "Carol Davis", "Marketing", 95000, "2020-06-01", "New York"),
        (4, "David Wilson", "Marketing", 85000, "2023-02-20", "Chicago"),
        (5, "Emma Brown", "Sales", 90000, "2021-09-05", "Los Angeles"),
        (6, "Frank Miller", "Sales", 88000, "2022-04-12", "Miami"),
        (7, "Grace Lee", "HR", 78000, "2020-11-30", "Boston"),
        (8, "Henry Taylor", "Engineering", 135000, "2019-08-22", "Austin"),
        (9, "Ivy Martinez", "Sales", 92000, "2023-01-08", "Denver"),
        (10, "Jack Anderson", "HR", 72000, "2022-07-15", "Portland"),
    ]
    cursor.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)",
        employees
    )

    # Seed products (8 products across 3 categories)
    products = [
        (1, "Cloud Platform Pro", "Software", 299.99, 500),
        (2, "Data Analytics Suite", "Software", 499.99, 250),
        (3, "Smart Monitor 27\"", "Electronics", 349.99, 75),
        (4, "Wireless Keyboard", "Electronics", 79.99, 200),
        (5, "Enterprise Support", "Services", 999.99, 100),
        (6, "Training Workshop", "Services", 1499.99, 50),
        (7, "Security Scanner", "Software", 199.99, 300),
        (8, "USB-C Hub Pro", "Electronics", 59.99, 150),
    ]
    cursor.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
        products
    )

    # Seed sales (20 sales records)
    sales = [
        (1, 1, 5, 299.99, "2024-01-15"),
        (2, 2, 5, 499.99, "2024-01-18"),
        (3, 3, 6, 349.99, "2024-01-20"),
        (4, 1, 9, 599.98, "2024-01-22"),
        (5, 5, 5, 999.99, "2024-01-25"),
        (6, 4, 6, 159.98, "2024-02-01"),
        (7, 2, 9, 999.98, "2024-02-05"),
        (8, 6, 5, 1499.99, "2024-02-10"),
        (9, 7, 6, 399.98, "2024-02-12"),
        (10, 1, 9, 299.99, "2024-02-15"),
        (11, 8, 5, 119.98, "2024-02-18"),
        (12, 3, 6, 699.98, "2024-02-20"),
        (13, 2, 5, 499.99, "2024-02-22"),
        (14, 5, 9, 1999.98, "2024-02-25"),
        (15, 4, 6, 79.99, "2024-03-01"),
        (16, 1, 5, 899.97, "2024-03-05"),
        (17, 6, 9, 2999.98, "2024-03-08"),
        (18, 7, 6, 199.99, "2024-03-10"),
        (19, 8, 5, 179.97, "2024-03-12"),
        (20, 2, 9, 1499.97, "2024-03-15"),
    ]
    cursor.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
        sales
    )

    conn.commit()
    conn.close()

    print(f"Database created at {db_path}")
    print("Tables:")
    print("  - employees: 10 records")
    print("  - products: 8 records")
    print("  - sales: 20 records")


if __name__ == "__main__":
    create_database()
