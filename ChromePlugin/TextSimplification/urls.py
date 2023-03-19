from django.urls import path,  re_path
import TextSimplification.views as views

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^get_wiki_summary/$', views.get_wiki_summary, name='get_wiki_summary'),
]