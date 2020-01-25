from flask import Flask
# Import CSRFProtect (Cross-Site Request Forgery Protection) object
from flask_wtf.csrf import CSRFProtect
# Import Bootstrap, an open source CSS framework
from flask_bootstrap import Bootstrap
import os

# Instantiate CSRF Protection object
csrf = CSRFProtect()
# Instantiate the Flask object used to run the web server
app = Flask(__name__)
# Add a cryptographically secure, randomized secret key used for form validation.
app.config['SECRET_KEY'] = os.urandom(32)

# Apply the app to the Bootstrap and CSRF objects.
Bootstrap(app)
csrf.init_app(app)

from mlserv import views