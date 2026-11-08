import os

print(f"version={os.getenv('APP_VERSION')} greeting={os.getenv('GREETING')} debug={os.getenv('DEBUG')}")
