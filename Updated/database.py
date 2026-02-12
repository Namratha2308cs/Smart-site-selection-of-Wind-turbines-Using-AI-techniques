import sqlite3
from flask import g
import os

DATABASE = 'wind_turbine.db'

def get_db():
    """Get database connection"""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Close database connection"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db(app):
    """Initialize database with tables"""
    with app.app_context():
        db = get_db()
        
        # Create users table
        db.execute('''
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create prediction history table
        db.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                input_data TEXT NOT NULL,
                feature_importance TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user (id)
            )
        ''')
        
        db.commit()

# Model classes for type hints
class User:
    def __init__(self, id, username, email, created_at):
        self.id = id
        self.username = username
        self.email = email
        self.created_at = created_at

class PredictionHistory:
    def __init__(self, id, user_id, prediction, probability, input_data, feature_importance, created_at):
        self.id = id
        self.user_id = user_id
        self.prediction = prediction
        self.probability = probability
        self.input_data = input_data
        self.feature_importance = feature_importance
        self.created_at = created_at