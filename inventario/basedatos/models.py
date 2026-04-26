from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.validators import UnicodeUsernameValidator
# Create your models here.

class Articulo(models.Model):
    nombre=models.CharField(max_length=255, primary_key=True)
    existencias = models.PositiveIntegerField(default=0)

class User(AbstractUser):
    rol = models.CharField(max_length=255)
    pass
    