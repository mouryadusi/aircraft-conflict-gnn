import trino
from trino.auth import OAuth2Authentication

print("=" * 60)
print("CONNECTING TO OPENSKY TRINO")
print("=" * 60)

conn = trino.dbapi.connect(
    host="trino.opensky-network.org",
    port=443,
    http_scheme="https",
    user="mdusi",  
    auth=OAuth2Authentication(),
    catalog="minio",
    schema="osky",
)

cursor = conn.cursor()

print("Connected successfully!")

cursor.execute("SHOW TABLES")

tables = cursor.fetchall()

print("\nAvailable tables:\n")

for t in tables:
    print(t[0])