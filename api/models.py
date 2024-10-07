from django.db import models

# Create your models here.

class Passage(models.Model):
    content = models.TextField()
