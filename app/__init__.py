from flask import Flask
from app.api.dataset_routes import dataset_bp


def create_app():
    app=Flask(__name__)
    app.register_blueprint(dataset_bp,url_prefix='/api/v1/datasets')
    return app

