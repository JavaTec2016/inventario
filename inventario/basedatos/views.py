from django.http import HttpRequest
from django.shortcuts import render

from django.views.generic import DetailView, ListView
from django.views.generic.edit import CreateView, DeleteView, UpdateView

from .cv.foto_parser import procesar, convertir

from .models import *
from django.urls import reverse
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib.auth.views import LoginView
# Create your views here.

class CrearArticulo(SuccessMessageMixin, CreateView):
    model = Articulo
    form = Articulo
    fields = '__all__'
    success_message = 'Articulo registrado'
    
    def get_success_url(self):
        return reverse('listado')

class ListarArticulos(ListView):
    model = Articulo
    context_object_name = 'articulos'

class ModificarArticulo(UpdateView):
    model = Articulo
    form = Articulo
    fields = '__all__'
    success_message = 'Articulo modificado'
    def get_success_url(self):
        return reverse('listado')

#===autenticacion en fa
class Login(LoginView):
    redirect_authenticated_user=True
    def get_success_url(self):
        return reverse('listado')
    
    def form_invalid(self, form):
        messages.error(self.request, 'Usuario o contraseña incorrectos.')
        return self.render_to_response(self.get_context_data(form=form))
    

#===procesar fotos

def procesarBatch(request:HttpRequest):
    if request.method == 'GET':
        return render(request, 'inventario/subes.html', {})
    elif request.method == 'POST':
        archivos = request.FILES.getlist('fotos[]')
        if not archivos:
            return render(request, 'inventario/subes.html', {'error':'No se subieron imagenes.'})
        for arch in archivos:
            arch.read()
        objetos = procesar(convertir(archivos))
        return render(request, 'inventario/subes.html' ,{'msg': 'todo bien', 'conteo':objetos})