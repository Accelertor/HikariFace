from flask import Flask, request, jsonify, render_template, session, redirect, url_for, abort
from db import add_or_update_user, get_embedding, mark_attendance, get_all_users, get_attendance_for_today
from face_recognition import extract_embedding, compare_embeddings
from datetime import datetime
import csv
import os
from functools import wraps
import numpy as np

app = Flask(__name__, static_folder="static")
app.secret_key = "your_secret_key_here"  # Change to a strong secret key

ADMIN_EMBED_CSV = "adminEmbed.csv"
EMBEDDING_THRESHOLD = 0.8  # similarity threshold

# --- Helper functions to save/load admin embedding from CSV ---

def save_admin_embedding(embedding):
    """Save the admin face embedding (numpy array) to CSV."""
    with open(ADMIN_EMBED_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(embedding.tolist())

def load_admin_embedding():
    """Load the admin face embedding from CSV and return as numpy array."""
    if not os.path.exists(ADMIN_EMBED_CSV):
        return None
    with open(ADMIN_EMBED_CSV, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            return np.array([float(x) for x in row])
    return None

# --- Admin login decorator ---

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('admin_login'))
        if session.get('role') != 'admin':
            abort(403)  # Forbidden
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

@app.route("/")
def attendance_page():
    return render_template("attend.html")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        photo = request.files.get("photo")
        if not photo:
            return render_template("admin_addUser.html", error="Please upload a photo")

        try:
            submitted_emb = extract_embedding(photo.read())
            admin_emb = load_admin_embedding()

            if admin_emb is None:
                return render_template("admin_addUser.html", error="Admin face is not enrolled. Please enroll first.")

            similarity = compare_embeddings(submitted_emb, admin_emb)
            if similarity >= EMBEDDING_THRESHOLD:
                session['username'] = 'admin'
                session['role'] = 'admin'
                return redirect(url_for('admin_page'))
            else:
                return render_template("admin_addUser.html", error="Face not recognized as admin")

        except Exception as e:
            return render_template("admin_addUser.html", error=f"Error: {str(e)}")

    return render_template("admin_addUser.html")

@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))

@app.route("/admin/enroll", methods=["GET", "POST"])
def admin_enroll():
    """
    Route to enroll admin face embedding and save it to CSV.
    Protect this route with a simple token or password in production.
    """
    if request.method == "POST":
        photo = request.files.get("photo")
        if not photo:
            return render_template("admin_enroll.html", error="Please upload a photo")

        try:
            embedding = extract_embedding(photo.read())
            save_admin_embedding(embedding)
            return render_template("admin_enroll.html", success="Admin face enrolled successfully.")
        except Exception as e:
            return render_template("admin_enroll.html", error=f"Error: {str(e)}")

    return render_template("admin_enroll.html")

@app.route("/admin")
@admin_required

def admin_page():
    return render_template("addUser.html")

@app.route("/admin/add_user", methods=["POST"])
@admin_required
def add_user():
    roll = request.form.get("roll")
    name = request.form.get("name")
    photo = request.files.get("photo")
    if not (roll and name and photo):
        return jsonify({"error": "Missing data"}), 400

    try:
        embedding = extract_embedding(photo.read())
        add_or_update_user(roll, name, embedding)
        return jsonify({"message": "User added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/attendance/submit", methods=["POST"])
def submit_attendance():
    roll = request.form.get("roll")
    name = request.form.get("name")
    photo = request.files.get("photo")
    if not (roll and name and photo):
        return jsonify({"error": "Missing data"}), 400

    try:
        attendance_record = get_attendance_for_today(roll)
        if attendance_record:
            return jsonify({
                "status": "present",
                "message": f"Already marked present at {attendance_record['timestamp']}"
            })
        submitted_emb = extract_embedding(photo.read())
        stored_emb = get_embedding(roll)
        if stored_emb is None:
            return jsonify({"error": "User not found"}), 404

        similarity = compare_embeddings(submitted_emb, stored_emb)
        status = "present" if similarity >= EMBEDDING_THRESHOLD else "absent"
        mark_attendance(roll, status)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"status": status, "timestamp": timestamp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/dashboard")
@admin_required
def dashboard():
    users = get_all_users()  # fetch users from db.py
    return render_template("dashboard.html", users=users)

if __name__ == "__main__":
    app.run()
