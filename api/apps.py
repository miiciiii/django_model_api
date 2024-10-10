from django.apps import AppConfig
# from .model_loader import initialize_models

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    # def ready(self):
    #     initialize_models()