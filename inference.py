"""Выполнение сохранённых моделей и правил сортировки на центральном процессоре."""
import csv
import io
import json
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
FRUITS = ['Яблоко', 'Банан', 'Лайм']
FRUIT_IDS = np.array([0, 0, 1, 1, 2, 2])
BAD_IDS = [0, 2, 4]
DECISIONS = ['Принять', 'Отбраковать', 'Ручная проверка']
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BATCH_BYTES = 50 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000


def read_image(content):
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError('Размер фотографии должен быть от 1 байта до 10 МБ.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ValueError('Поддерживаются фотографии JPG, PNG и WebP.')
                source.load()
                return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as error:
        raise ValueError('Не удалось прочитать фотографию. Проверьте формат и размер.') from error


def preview_bytes(image):
    preview = image.copy()
    preview.thumbnail((900, 700), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    preview.save(output, format='JPEG', quality=85)
    return output.getvalue()


def create_session(path):
    # Два потока и последовательное выполнение ограничивают
    # расход ресурсов на общей бесплатной площадке.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=['CPUExecutionProvider'])


def softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def prepare_detector_input(image):
    # Сохраняются пропорции и поля исходного способа обработки.
    # Размер по большей стороне равен 960, поля кратны шагу сети 32.
    pixels = np.asarray(image)
    height, width = pixels.shape[:2]
    ratio = min(960 / height, 960 / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    horizontal = (960 - resized_width) % 32
    vertical = (960 - resized_height) % 32
    left, right = round(horizontal / 2 - .1), round(horizontal / 2 + .1)
    top, bottom = round(vertical / 2 - .1), round(vertical / 2 + .1)
    if (resized_width, resized_height) != (width, height):
        pixels = cv2.resize(pixels, (resized_width, resized_height),
                            interpolation=cv2.INTER_LINEAR)
    pixels = cv2.copyMakeBorder(pixels, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, ratio, left, top


def decode_regions(output, image_size, ratio, left, top):
    # Сеть возвращает центр, размеры и оценки двух категорий.
    # Сохраняются участки с оценкой от 0,25. Повторные рамки
    # одной категории подавляются при совпадении более 0,7.
    candidates = output[0].T
    scores = candidates[:, 4:6].max(axis=1)
    classes = candidates[:, 4:6].argmax(axis=1)
    selected = scores > .25
    boxes = candidates[selected, :4].copy()
    scores, classes = scores[selected], classes[selected]
    if not len(boxes):
        return []
    centers = boxes[:, :2].copy()
    sizes = boxes[:, 2:].copy()
    boxes[:, :2] = centers - sizes / 2
    boxes[:, 2:] = centers + sizes / 2
    order = np.argsort(-scores, kind='stable')[:30000]
    kept = []
    while len(order) and len(kept) < 300:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not len(remaining):
            break
        lower = np.maximum(boxes[current, :2], boxes[remaining, :2])
        upper = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection = np.maximum(upper - lower, 0).prod(axis=1)
        first_area = np.maximum(boxes[current, 2:] - boxes[current, :2], 0).prod()
        other_area = np.maximum(boxes[remaining, 2:] - boxes[remaining, :2], 0).prod(axis=1)
        overlap = intersection / np.maximum(first_area + other_area - intersection, 1e-7)
        suppress = (classes[remaining] == classes[current]) & (overlap > .7)
        order = remaining[~suppress]
    width, height = image_size
    regions = []
    for index in kept:
        box = boxes[index].copy()
        box[[0, 2]] = np.clip((box[[0, 2]] - left) / ratio, 0, width) / width
        box[[1, 3]] = np.clip((box[[1, 3]] - top) / ratio, 0, height) / height
        regions.append({'category': int(classes[index]), 'score': float(scores[index]),
                        'rectangle': box.tolist()})
    return regions


class FruitSystem:
    def __init__(self):
        self.settings = json.loads((ROOT / 'config/decisions.json').read_text())
        config = json.loads((ROOT / 'config/network.json').read_text())
        self.mean = np.asarray(config['normalization_mean'], dtype=np.float32)[:, None, None]
        self.std = np.asarray(config['normalization_std'], dtype=np.float32)[:, None, None]
        self.classifier = create_session(ROOT / 'models/sorting.onnx')
        self.detector = None
        # Один общий экземпляр используется разными посетителями.
        # Одновременная обработка партий ограничивается этим замком.
        self.lock = threading.Lock()

    def classification_logits(self, image):
        resized = image.resize((224, 224), Image.Resampling.BILINEAR)
        tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = np.ascontiguousarray(((tensor - self.mean) / self.std)[None])
        return self.classifier.run(None, {self.classifier.get_inputs()[0].name: tensor})[0][0]

    def detect_regions(self, image):
        if self.detector is None:
            self.detector = create_session(ROOT / 'models/apple_regions.onnx')
        tensor, ratio, left, top = prepare_detector_input(image)
        output = self.detector.run(None, {self.detector.get_inputs()[0].name: tensor})[0]
        return decode_regions(output, image.size, ratio, left, top)

    def analyze(self, entries, mode='neural', show_regions=False):
        if mode not in {'neural', 'batch'}:
            raise ValueError('Неизвестный режим обработки.')
        if not 1 <= len(entries) <= 20:
            raise ValueError('Добавьте от 1 до 20 фотографий.')
        if sum(len(entry['content']) for entry in entries) > MAX_BATCH_BYTES:
            raise ValueError('Общий размер фотографий должен быть не больше 50 МБ.')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Система обрабатывает другую партию. Повторите немного позже.')
        try:
            return self._analyze(entries, mode, show_regions)
        finally:
            self.lock.release()

    def _analyze(self, entries, mode, show_regions):
        started = time.perf_counter()
        stable = mode == 'batch'
        brightness_values = [1.0, .85, 1.15] if stable else [1.0]
        all_probabilities, results = [], []
        for entry in entries:
            try:
                image = read_image(entry['content'])
            except ValueError as error:
                raise ValueError(f"{entry['name']}: {error}") from error
            logits = np.stack([
                self.classification_logits(ImageEnhance.Brightness(image).enhance(value))
                for value in brightness_values
            ])
            variants = softmax(logits / self.settings['temperature'])
            all_probabilities.append(variants)
            combined = variants.mean(axis=0)
            fruit_scores = np.array([combined[FRUIT_IDS == index].sum() for index in range(3)])
            fruit_id = int(fruit_scores.argmax())
            regions = self.detect_regions(image) if show_regions and fruit_id == 0 else []
            preview = image.copy()
            preview.thumbnail((900, 700), Image.Resampling.LANCZOS)
            annotation = None
            if regions:
                annotated = preview.copy()
                drawing = ImageDraw.Draw(annotated)
                width, height = annotated.size
                for region in regions:
                    x1, y1, x2, y2 = region['rectangle']
                    color = '#e6c830' if region['category'] == 0 else '#24c9d0'
                    drawing.rectangle([x1 * width, y1 * height, x2 * width, y2 * height],
                                      outline=color, width=3)
                    drawing.text((x1 * width, y1 * height), str(region['category']), fill=color)
                annotation = preview_bytes(annotated)
            results.append({'filename': entry['name'], 'fruit': FRUITS[fruit_id],
                            'fruit_id': fruit_id, 'fruit_score': float(fruit_scores.max()),
                            'image': preview_bytes(preview), 'annotated': annotation,
                            'regions': regions, 'regions_checked': show_regions and fruit_id == 0})
            # Полное изображение не сохраняется в результате партии.
            del image, preview
        variants = np.stack(all_probabilities)
        combined = variants.mean(axis=1)
        defect = combined[:, BAD_IDS].sum(axis=1)
        instability = variants[:, :, BAD_IDS].sum(axis=2).std(axis=1)
        fruits = np.array([item['fruit_id'] for item in results])
        confidence = np.array([item['fruit_score'] for item in results])
        policy = self.settings['policies']['risk_stable' if stable else 'neural']
        thresholds = np.asarray(policy['thresholds'])[fruits]
        decisions = (defect >= thresholds).astype(int)
        quota = 0
        if stable:
            quota = int(np.floor(len(entries) * self.settings['review_limit']))
            priority = np.minimum(self.settings['miss_cost'] * defect,
                                  self.settings['reject_cost'] * (1 - defect))
            priority += policy['instability_weight'] * instability + .1 * (1 - confidence)
            if quota:
                decisions[np.argsort(-priority, kind='stable')[:quota]] = 2
        for index, result in enumerate(results):
            decision = int(decisions[index])
            result.update(decision=decision, decision_label=DECISIONS[decision],
                          defect_score=float(np.clip(defect[index], 0, 1)),
                          threshold=float(thresholds[index]))
        return {'items': results, 'review_quota': quota, 'mode': mode,
                'show_regions': show_regions, 'seconds': round(time.perf_counter() - started, 2),
                'counts': {'accepted': int(np.sum(decisions == 0)),
                           'rejected': int(np.sum(decisions == 1)),
                           'review': int(np.sum(decisions == 2))}}


def make_csv(result):
    def safe_value(value):
        text = str(value)
        return "'" + text if text.startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else text
    output = io.StringIO(newline='')
    writer = csv.writer(output, delimiter=';')
    writer.writerow(['Файл', 'Вид', 'Решение', 'Оценка дефекта, %', 'Граница решения, %',
                     'Найдено участков'])
    for item in result['items']:
        writer.writerow([safe_value(item['filename']), item['fruit'], item['decision_label'],
                         round(item['defect_score'] * 100, 2), round(item['threshold'] * 100, 2),
                         len(item['regions'])])
    return output.getvalue().encode('utf-8-sig')
