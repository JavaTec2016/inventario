from django.urls import path

from basedatos.views import *

urlpatterns = [
    path('', Login.as_view(template_name='login.html'), name='login'),
    path('inventario', ListarArticulos.as_view(template_name='inventario/listado.html'), name='listado'),
    path('inventario/agregar', CrearArticulo.as_view(template_name='inventario/agregar.html'), name='agregar'),
    path('inventario/modificar', ModificarArticulo.as_view(template_name='inventario/modificar.html'), name='modificar'),
    path('inventario/subes', procesarBatch),
]