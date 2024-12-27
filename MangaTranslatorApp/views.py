from django.shortcuts import render
from MangaTranslatorApp.predict_model import predict
from MangaTranslatorApp.read_translate_draw import process_text
import os
from django.conf import settings
import matplotlib.pyplot as plt
def HomePage(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("image")
        selected_language = request.POST.get("language")
        if not uploaded_file:
            return render(request, 'homePage.html', {'error': "Lütfen bir görsel yükleyin."})

        static_dir = os.path.join(settings.STATICFILES_DIRS[0], "img")
        os.makedirs(static_dir, exist_ok=True)
        original_name, file_extension = os.path.splitext(uploaded_file.name)

        saved_file_path = os.path.join(static_dir, uploaded_file.name)

        with open(saved_file_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        try:
            boxes = predict(saved_file_path)
            translated_file_name = f"{original_name}_translated{file_extension}"
            processed_image_path = process_text(saved_file_path, boxes, selected_language, translated_file_name)
            processed_image_url = f"/static/img/{translated_file_name}"
            original_image_url = f"/static/img/{uploaded_file.name}"
            return render(request, 'homePage.html', {
                'processed_image_url': processed_image_url,
                'original_image_url': original_image_url
            })
        except Exception as e:
            if os.path.exists(saved_file_path):
                os.remove(saved_file_path)
            return render(request, 'homePage.html', {'error': str(e)})

    return render(request, 'homePage.html')

def AboutPage(request):
    return render(request, 'About.html')