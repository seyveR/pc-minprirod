# Автоматическая фильтрация изображений животных

Мы представляем решение для атоматической фильтрации изображений, полученных с фотоловушек, на пригодность для дальнейшего анализа научными работниками.
Наше решение представляет собой веб-сервис, на котором пользователь с удобством и меньшими, в сравнении с ручной обработкой, временными затратами может получить необходимую информацию об изображениях  с фотоловушек. Пользователь загружает изображения и в реальном времени может отслеживать процесс обработки. Во время обработки фотографии проходят через несколько моделей:
1. SWIN transformer для классификации изображений на пустое/не пустое
2. Yolov8 для выделения области, где находится животное и дальнейшего обрезания изображения по контуру Bounding Box;
3.1 SWIN transformer для классификации обрезанного изображения - либо изображение, пригодное для дальнейшего анализа, либо вспомогательное изображение;
3.2 Yolov8-seg для сегментации частей тела животного, с помощью чего можно понять, какие части тела находятся в кадре и с помощью скрипта выяснить, подходит ли изображение для дальнейшего анализа или же является вспомогательным;
4. RandomForest для подведения итогов обработки. Эта модель проводит анализ полученных ответов от предыдущих моделей и выносит финальный вердикт, за счет нескольких признаков. 
После обработки моделями изображения выводятся на веб-сервис, где пользователь может просмотреть и скачать фотографии, посмотреть и выгрузить отчет по изображениям. 

## Доступ

Для проверки развертывания или тестирования моделей вы можете скачать наши обученные веса по ссылке с яндекс диска https://disk.yandex.ru/d/sG6D3SuWDfEX9Q

## Обучение

- yolo_seg_train.ipynb - юпитер для обучения модели сегментации yolov8
- torch_classifier_train.ipynb - юпитер для обучения моделей классификации
- preprocessor.ipynb - юпитер для сборки параметров в датасет
- ml_train.ipynb - юпитер для обучения Random Forest
- submit_creation.ipynb - юпитер для создания сабмитов
- req_train.txt - зависимости для обучения моделей

Чтобы запустить обучение или тестирование, установите в свое виртуальное окружение зависимости, например ``` pip install -r req_train.txt```


## api-modeling
Используется для соединения и управления получением нарезок из видео, по сокету начинает выполнение обработки передовая текущии состояния
Для установки api-modeling сервиса выполниете:
- используйте версию python11.9 , создайте env усновите зависимости:
- ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```
- ```pip install --no-cache-dir --upgrade -r /code/requirements.txt```
- ``` uvicorn main:app --port 8000 ```
- 
Обратите внимание на настройку переменной path и full_path в main.py
> path = '../vidcut/static/upload_videos/'
> full_path = 'D:/Studing/videocut/vidcut/static/upload_videos/' + ...

Основной путь для установления socet:
> @app.websocket("/{item_id}/ws")

Со стороны FastApi отрабатывают следующие модели:
-R2 (Highlight Detection), которая выдает тайминги интересных моментов;
-Whisper, которая транскрибирует полученные фрагменты в текст, для дальнейшей обработки;
-LLama (1), обрабатывает текст каждого из фрагментов, для идентификации самых интересных клипов, из предложенных R2;
-LLama (2), обрабатывает текст, полученный LLama(1), для уверенности в том, что полученную ей информацию можно в дальнейшем обработать для работы в Веб-сервисе;
-Yolov8, которая отцентровывает видео относительно "Самого привлекательного объекта" в видеоряде, будь то человек или что-то иное.

Также на FastApi кроме обработки видео происходит и обрезка видео moviepy

## Django
Небольшое приложение с templates и статикой
Используется для:
- фронтенд взаимодейсвия с пользователем 
- бэкенд взаимодейсвия с пользователем
- храенение видео

Для установки Django сервиса выполниете:
- используйте версию python11.9 , создайте env усновите зависимости:
- pip install -r requirements.txt
- python11.9
- ``` python manage.py runserver localhost:80  ```

Используемые urls:
Для начала работы:
> http://127.0.0.1/

Для вызова загрузки файла
> http://127.0.0.1/upload_video/
> method POST

Запрос на api-modeling лежит в файле 
> vidcut\main\templates\main\cutting.html
> var ws = new WebSocket("ws://localhost:8000/{{ video_link }}/ws");


