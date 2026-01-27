import os
from flask import Blueprint, request, session, redirect, url_for, render_template, jsonify
from app.config import USERNAME, PASSWORD

bp_auth = Blueprint("auth", __name__)

@bp_auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pwd = request.form.get("password")
        if user == USERNAME and pwd == PASSWORD:
            session["user"] = user
            return redirect(url_for("core.index"))
        return render_template("login.html", error="Usuário ou senha incorretos")
    return render_template("login.html")

@bp_auth.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("auth.login"))

# Decorator para proteger rotas
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function
