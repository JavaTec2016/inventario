from django.db import models

# Create your models here.
class Articulo(models.Model):
    nombre=models.CharField(max_length=255, primary_key=True)
    marca=models.CharField(max_length=255, null=False)

class Existencias(models.Model):
    nombre=models.ForeignKey(Articulo.__name__, on_delete=models.CASCADE)

class Precio(models.Model):
    nombre=models.ForeignKey(Articulo.__name__, on_delete=models.CASCADE)