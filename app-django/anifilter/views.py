import datetime

import requests
from django.shortcuts import render
from django.http import HttpResponse, HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt  # Если используете CSRF
import os
import uuid


def start(request: HttpRequest):
    return render(request, 'start.html')


def analyz(request):
    return render(request, 'analyz.html')


def filtration(request):
    # Получаем ID из параметров GET-запроса
    upload_id = request.GET.get('id')
    print(upload_id)
    if not upload_id:
        # Если ID не найден, возвращаем ошибку
        return HttpResponse('ID не предоставлен', status=400)

    # Здесь вы можете выполнить обработку с использованием upload_id
    # Например, начать процесс фильтрации или другой логики
    # Ваш код обработки...

    # Передаем upload_id в шаблон, если это необходимо
    return render(request, 'filtration.html', {'upload_id': upload_id})

def handle_uploaded_file(f):
    """
    Сохранение файлов
    :param f: file object
    :return:
    """
    file_name = f.name
    path = "/upload_files/"
    if not os.path.exists('static/upload_files'):
        os.makedirs('static/upload_files')
    with open('static' + path + file_name.lower(), "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return 'static' + path + file_name

@csrf_exempt
def upload_files(request):
    context = {"page": 'index'}
    if request.method == "POST":
        context['files'] = []
        file_objs = request.FILES.getlist('files')
        headers = {
            'accept': 'application/json',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }
        files = []
        for file in file_objs:
            f_name = handle_uploaded_file(file)
            files.append(('files', (file.name, open(f_name, 'rb'))))

        response = requests.post('http://api-model-control:8001/uploadfiles/', headers=headers, files=files)
        return JsonResponse({'upload_id': response.json()['files_upload_id']})


def results(request: HttpRequest):
    return render(request, 'results.html')
