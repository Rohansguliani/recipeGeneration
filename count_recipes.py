import sqlite3

conn = sqlite3.connect('recipes.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM recipes")
count = cursor.fetchone()[0]
print(f"Total recipes in the database: {count}")
conn.close() 