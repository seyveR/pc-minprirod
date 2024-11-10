import glob
import json
import os

from PIL import Image
from starlette.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from typing import Annotated

import aiofiles
from starlette.websockets import WebSocket
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

from ultralytics.engine.results import Results
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from torchvision.models import swin_v2_s
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import cv2

# API config
DOMAIN = os.getenv('DOMAIN', 'http://localhost:8001')
PATH_UPLOAD = 'upload'
counters = {'id_folder': max([int(i) for i in os.listdir(PATH_UPLOAD)] + [0]) + 1}
if not os.path.exists(PATH_UPLOAD):
    os.makedirs(PATH_UPLOAD)

app = FastAPI()
app.mount('/' + PATH_UPLOAD, StaticFiles(directory="upload"), name="upload")


# models config
model_seg = YOLO(model="finalseg.pt")
model_det = YOLO(model="best.pt")

best_rf = joblib.load('best_rf_model.pkl')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_labels = {1: 'good', 0: 'bad'}

# Трансформации для Swin
test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

rigins = ['null']  # NOT recommended - see details below


def get_swin_model():
    model = swin_v2_s(pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)  # Обновляем последний слой для 2 классов
    checkpoint = torch.load('best_efficient.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


swin_model = get_swin_model()


def save_image(id_folder, photo_name, detection_results):
    path = f'{PATH_UPLOAD}/{id_folder}/'
    photo_name = photo_name
    image = Image.open(path + photo_name)
    image_width, image_height = image.size

    label_mapping = model_det.names


    output_folder = path
    os.makedirs(output_folder, exist_ok=True)

    for result in detection_results:
        for box in result.boxes:
            label_target = [{
                'class': label_mapping.get(result.names[box.cls.item()], 'Unknown'),
                'conf': box.conf.item()
            }]

        annot = result.plot(boxes=True, labels=True, conf=True, line_width=2)

        im = Image.fromarray(annot[..., ::-1])  # Конвертируем из BGR в RGB для PIL

        output_path = os.path.join(output_folder, f"annotated_{photo_name}")
        im.save(output_path, format="JPEG")
    return path + f"annotated_{photo_name}"

# def save_image(id_folder, photo_name, detection_results):
#     path = f'{PATH_UPLOAD}/{id_folder}/'
#     image_path = os.path.join(path, photo_name)
#     image = Image.open(image_path)
#     image_width, image_height = image.size

#     label_mapping = model_det.names

#     output_folder = path
#     os.makedirs(output_folder, exist_ok=True)

#     img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     for result in detection_results:
#         for box in result.boxes:
#             x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
#             cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
#     img_annotated = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

#     output_path = os.path.join(output_folder, f"annotated_{photo_name}")
#     img_annotated.save(output_path, format="JPEG")
    
#     return output_path

def tensor_from_images(image):
    image = test_transform(image=np.array(image))['image'].unsqueeze(0).to(device)
    return image


def predict_swin(image, model):
    tensor = tensor_from_images(image)
    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
        return preds.item()


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    id_folder = counters['id_folder']
    counters['id_folder'] += 1
    saved_files = []
    folder = f'{PATH_UPLOAD}/{id_folder}/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in files:
        async with aiofiles.open(f'{folder}{file.filename}', 'wb+') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write
            saved_files.append(f'{folder}{file.filename}')
    return {'files_upload_id': id_folder, 'saved_files_paths': saved_files}


@app.websocket("/{item_id}/ws")
async def websocket_endpoint(
        websocket: WebSocket,
        item_id: str,
):
    await websocket.accept()
    await websocket.send_json({'type': 'start', 'text': 'Начинаем'})
    query_text = 'most viral moments more than 10 seconds'
    if item_id is None:
        await websocket.close()
        return 0
    id_folder = item_id
    images_folder = PATH_UPLOAD + '/' + str(id_folder)
    files = os.listdir(PATH_UPLOAD + '/' + str(id_folder))
    size_of_path = len(files)
    count = 0
    images_paths = []

    # Создаем пустой датафрейм для результатов
    df_results = pd.DataFrame(columns=["photo_name", "orig_width", "orig_height", "crop_width", "crop_height",
                                       "pred_seg_class", "pred_det_class", "is_head", "is_body", "is_legs",
                                       "is_tail", "area_head", "area_body", "area_legs", "area_tail", "Bbox"])

    for image_name in os.listdir(images_folder):
        if image_name.find('annotated') != -1:
            continue
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_folder, image_name)

            image = Image.open(image_path)
            image_width, image_height = image.size
            await websocket.send_json({
                'type': 'worker',
                'text': f"Обрабатывается изображение: {image_name}, размер: {image_width}x{image_height}"
            })
            # ДЕТЕКЦИЯ
            detection_results = model_det.predict(image, iou=0.5, conf=0.52)
            animal_count = sum(len(result.boxes) for result in detection_results)
            await websocket.send_json({
                'type': 'worker',
                'text': f"Количество животных на {image_name}: {animal_count}, "
            })

            img_link = save_image(id_folder, image_name, detection_results)
            images_paths.append('http://localhost:8001/' + img_link)
            await websocket.send_json({
                    'type': 'worker',
                    'text': f"Сохранение изображение с разметкой, ссылка: {img_link}"
            })


            # КРОП ПО ДЕТЕКЦИИ
            for result in detection_results:
                for i, box in enumerate(result.boxes):
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(image_width, x_max)
                    y_max = min(image_height, y_max)

                    # ВЫРЕЗАЕМ
                    cropped_image = image.crop((x_min, y_min, x_max, y_max))
                    cropped_image_np = np.array(cropped_image)

                    cropped_height, cropped_width = cropped_image_np.shape[:2]
                    await websocket.send_json({
                        'type': 'worker',
                        'text': f"Размер обрезанного изображения для объекта {i + 1}: {cropped_width}x{cropped_height}"
                    })

                    # Рассчитываем центр и размеры в нормализованной форме для Bbox
                    center_x = (x_min + x_max) / 2 / image_width
                    center_y = (y_min + y_max) / 2 / image_height
                    norm_width = (x_max - x_min) / image_width
                    norm_height = (y_max - y_min) / image_height
                    bbox_str = f"{center_x:.5f},{center_y:.5f},{norm_width:.5f},{norm_height:.5f}"

                    # Инициализация значений по умолчанию
                    is_head, is_body, is_legs, is_tail = 0, 0, 0, 0
                    area_head, area_body, area_legs, area_tail = 0, 0, 0, 0

                    if cropped_width < 180 and cropped_height < 180:
                        pred_seg_class = "bad"
                    else:
                        # СЕГМЕНТАЦИЯ
                        segmentation_results = model_seg.predict(source=cropped_image_np, iou=0.5, conf=0.3)

                        class_counts = {"head": 0, "body": 0, "leg": 0, "tail": 0}
                        class_areas = {"head": 0, "body": 0, "leg": 0, "tail": 0}

                        for segment in segmentation_results:
                            for j, label in enumerate(segment.boxes.cls):
                                class_name = model_seg.names[int(label)]
                                class_counts[class_name] += 1

                                mask_x_min, mask_y_min, mask_x_max, mask_y_max = map(int, segment.boxes.xyxy[j])
                                mask_area = (mask_x_max - mask_x_min) * (mask_y_max - mask_y_min)
                                class_areas[class_name] += mask_area

                        head_count = class_counts["head"]
                        body_count = class_counts["body"]
                        leg_count = class_counts["leg"]
                        tail_count = class_counts["tail"]

                        area_head = class_areas["head"]
                        area_body = class_areas["body"]
                        area_legs = class_areas["leg"]
                        area_tail = class_areas["tail"]

                        if head_count >= 1 and body_count >= 1 and (tail_count >= 1 and leg_count >= 1):
                            pred_seg_class = "good"
                        elif head_count >= 1 and body_count >= 1 and (tail_count == 0 and leg_count >= 2):
                            pred_seg_class = "good"
                        else:
                            pred_seg_class = "bad"

                        is_head, is_body, is_legs, is_tail = head_count, body_count, leg_count, tail_count

                    pred_seg_class_encoded = 1 if pred_seg_class == "good" else 0

                    # Используем модель Swin для предсказания класса вместо YOLO
                    swin_predicted_class = predict_swin(cropped_image, swin_model)
                    pred_det_class_encoded = 1 if swin_predicted_class == 1 else 0

                    new_row = pd.DataFrame([{
                        "photo_name": image_name,
                        "orig_width": image_width,
                        "orig_height": image_height,
                        "crop_width": cropped_width,
                        "crop_height": cropped_height,
                        "pred_seg_class": pred_seg_class_encoded,
                        "pred_det_class": pred_det_class_encoded,
                        "is_head": is_head,
                        "is_body": is_body,
                        "is_legs": is_legs,
                        "is_tail": is_tail,
                        "area_head": area_head,
                        "area_body": area_body,
                        "area_legs": area_legs,
                        "area_tail": area_tail,
                        "Bbox": bbox_str
                    }])
                    df_results = pd.concat([df_results, new_row], ignore_index=True)

    final_result = df_results.drop(
        columns=['photo_name', 'orig_width', 'orig_height', 'area_head', 'area_body', 'area_legs', 'area_tail', 'Bbox'])

    scaler = StandardScaler()
    final_result_scaled = scaler.fit_transform(final_result)
    predictions = best_rf.predict(final_result_scaled)

    statistics = pd.DataFrame({
        'photo_name': 'http://localhost:8001/' + images_folder + '/annotated_' + df_results['photo_name'],
        'class': predictions
    })
    await websocket.send_json({'type': 'stop', 'text': 'Завершаем'})
    await websocket.send_json({

        'type': 'result', 'images': images_paths, 'result': statistics.to_dict(),
        'count_obj': len(statistics.index),
        'count_annotated': len(images_paths),
        'count_all': size_of_path
    })
    await websocket.close()
