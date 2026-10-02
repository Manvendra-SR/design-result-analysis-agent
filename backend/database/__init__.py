"""
backend/database
=================
connection.py  the SQLAlchemy engine (from DATABASE_URL)
models.py      ORM tables: datasets, sessions, experiments
repository.py  Repository - the only module that reads or writes them
init_db.py     python -m backend.database.init_db [--reset]
"""
