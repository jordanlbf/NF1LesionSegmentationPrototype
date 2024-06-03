# segformer_project/urls.py
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from segformer_app.views import upload_image

urlpatterns = [
    path('admin/', admin.site.urls),
    path('segformer/upload/', upload_image, name='upload_image'),
    path('', upload_image, name='upload_image'),  # Add this line to set the upload view as the default
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
