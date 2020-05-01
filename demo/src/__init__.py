import os
from flask import Flask

app = Flask(__name__, instance_relative_config=True)

from views.api import demo_bp

# register the blueprints
app.register_blueprint(demo_bp)
