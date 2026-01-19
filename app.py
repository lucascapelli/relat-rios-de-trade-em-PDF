from flask import Flask, render_template
import threading
import webbrowser
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def open_browser():
    time.sleep(1)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=False)