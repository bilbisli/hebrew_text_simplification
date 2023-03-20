from django.urls import path,  re_path
import TextSimplification.views as views

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^get_simplified/$', views.get_simplified, name='get_simplified'),
]