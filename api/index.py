import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app

def handler(request):
    return flask_app(request.environ, lambda start_response, response: (start_response(response[0], response[1]), response[2]))