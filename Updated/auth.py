from flask import session, flash, redirect, url_for
from database import get_db, User
import hashlib
import secrets
from functools import wraps

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return stored_password == hash_password(provided_password)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_id():
    """Get current user ID from session"""
    return session.get('user_id')

def register_user(username, email, password):
    """Register a new user"""
    db = get_db()
    
    # Check if user already exists
    existing_user = db.execute(
        'SELECT id FROM user WHERE username = ? OR email = ?', 
        (username, email)
    ).fetchone()
    
    if existing_user:
        return False, "Username or email already exists"
    
    # Create new user
    hashed_password = hash_password(password)
    db.execute(
        'INSERT INTO user (username, email, password_hash) VALUES (?, ?, ?)',
        (username, email, hashed_password)
    )
    db.commit()
    
    return True, "User registered successfully"

def login_user(username, password):
    """Login user"""
    db = get_db()
    
    user = db.execute(
        'SELECT * FROM user WHERE username = ?', (username,)
    ).fetchone()
    
    if user and verify_password(user['password_hash'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return True, "Login successful"
    
    return False, "Invalid username or password"

def logout_user():
    """Logout user"""
    session.clear()