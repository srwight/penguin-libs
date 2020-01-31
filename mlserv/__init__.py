from flask import Flask
# Import Bootstrap, an open source CSS framework
from flask_bootstrap import Bootstrap
import os

# Instantiate the Flask object used to run the web server
app = Flask(__name__)
# Add a cryptographically secure, randomized secret key used for form validation.
app.config['SECRET_KEY'] = os.urandom(32)

# Apply the app to the Bootstrap and CSRF objects.
Bootstrap(app)

from mlserv import views