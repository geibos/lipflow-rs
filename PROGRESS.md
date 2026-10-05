# Lipflow → Rust: журнал

Оригинал (Python) — `lipflow/` (клон upstream, не трогаем, служит эталоном).
Rust-воркспейс — корень `lipflow-rs/`. Эталонные данные — `ref/` (генерирует `tools/baseline.py`).

## Цель и правила (из задачи)

- Цель — производительность и надёжность, не RIIR ради RIIR. C-биндинги/API можно оставлять.
- Инкрементами; каждый инкремент — замер против baseline. Несколько инкрементов без выигрыша → стоп.
- Модель (инференс + дообучение) тоже на Rust, если это даёт скорость/точность; иначе стоп по модели.
- Приложение полностью функционально: хоткеи, HUD, меню, онбординг.

## Baseline Python (M1 Pro 32 ГБ, macOS 15.7, 38 клипов bench, beam 4, nbest 5)

| метрика | значение |
|---|---|
| WER (beam 4) | 26.13% |
| encoder MPS, сумма / среднее | 10.33 s / 0.27 s |
| beam search CPU, сумма / среднее | 90.75 s / 2.39 s |
| загрузка моделей | 2.98 s |
| import python+torch+mediapipe (холодный) | 26.1 s |
| max RSS | 2404 MB |

## План

1. [ ] crate `vsr`: загрузка весов из .pth, encoder (Conv3d-ResNet + Conformer), CTC greedy. Сверка с PyTorch.
2. [ ] decoder + Transformer LM + CTC prefix scorer + batch beam search. Сверка гипотез, замер.
3. [ ] face: MediaPipe FaceLandmarker без Python.
4. [ ] приложение macOS: камера, хоткей, вставка, HUD, меню, настройки, онбординг, cleanup.
5. [ ] дообучение (front end на лице, LM на фразах) на Rust.
6. [ ] AV-режим (шёпот), Windows.

## Журнал инкрементов

### 1. Инференс VSR на candle (crate `crates/vsr`) — выигрыш есть

- Порт: Conv3d-ResNet18 front end, 12×Conformer (rel-pos MHA, macaron, conv module), decoder 6×,
  Transformer LM 16×, CTC prefix scorer (на f32-срезах), batch beam search ESPnet (pre-beam,
  end_detect, maxlen=T). Веса читаются прямо из `.pth` (candle pickle).
- Корректность: front end max|Δ| 7e-6, encoder 3e-5 против PyTorch; гипотезы beam
  **38/38 дословно совпадают** с Python, WER 26.13% = baseline.
- Скорость (38 клипов): Python enc+beam ≈ 101 s → Rust 24.2 s (beam на Metal) / 31.0 s (beam на CPU).
  Среднее на фразу ≈ 0.64 s против 2.66 s (×4.2).
- Что сделано для скорости:
  - candle Metal conv2d в ~20× медленнее своего matmul (strided-копии) → свои Metal-kernels
    (`metal_ops.rs`): NHWC, im2col, fused bias+residual+swish, maxpool; BN свёрнут в веса.
    Front end 0.75 s → 0.16 s на 192 кадра.
  - KV-кэш вместо пересчёта K/V всего префикса; cross-attention: все гипотезы одним matmul.
  - decoder и LM считаются параллельно (потоки).
- Узкое место сейчас: шаг decoder/LM на Metal упирается в CPU-стоимость диспетчеризации
  (~700 мелких candle-операций на шаг). Резерв: fused QKV, свой engine шага.

### 2. Лицо без MediaPipe (crate `crates/face`) — выигрыш по надёжности, скорость паритет

- Свой интерпретатор TFLite (flatbuffer-парсер + 10 операций, f16-веса), читает модели прямо
  из `face_landmarker.task`. Против LiteRT: относительная ошибка ~1e-6.
- Логика FaceLandmarker VIDEO-режима по исходникам MediaPipe: SSD-якоря, weighted NMS,
  ROI с поворотом (1.5×, square_long), трекинг без детектора, проекция, One-Euro сглаживание.
- Кроп рта (`mouth_rois`): fixed-point bilinear OpenCV (warpAffine/warpPerspective), LMedS
  `estimateAffinePartial2D` воспроизведён вплоть до RNG OpenCV.
- Против MediaPipe на клипе (231 кадр): точки mean 0.009 px / max 0.125 px; кропы из
  Python-якорей 99.96% пикселей бит-в-бит (max Δ 4), из Rust-точек MAE 0.026 уровня яркости.
- Скорость: 9.2 мс/кадр (MediaPipe 7.3 мс). Ключевое: переиспользование буферов
  (page faults), branch-free PReLU, параллельный depthwise.
- Убрана зависимость от mediapipe/opencv/numpy (≈ сотни МБ Python-колёс).

### 3. Сквозной путь «видео → текст» (crate `crates/app`, `lipflow file` / `lipflow bench`)

- Видео через AVFoundation (AVAssetReader, BGRA), без ffmpeg/OpenCV.
- `lipflow file sample 20.4–28.1`: тот же текст, что у Python; 8.8 s всего против 20.7 s.
- `lipflow bench` (38 фраз, полный путь из видео): WER 23.87% (Python с mediapipe-кропами: 26.13%;
  разница — шум декодера видео, не заявляю как улучшение), decode 0.63 s/фраза.

### 4. Приложение macOS (crates/app) — функционально, кроме обучения/whisper/local-LLM

- Нативно на objc2: меню-бар (NSStatusItem + меню + выбор камеры), глобальный хоткей
  (CGEventTap listen-only, свои события по метке), вставка (NSPasteboard + ⌘V + восстановление),
  камера (AVCaptureSession в своём потоке, делегат на serial-очереди, трекинг прямо по
  CVPixelBuffer без копий), HUD (NSPanel, Liquid Glass через runtime / HUD-материал, метр,
  видео рта с контурами), окно настроек, онбординг (разрешения, импорт Wispr, практика, обучение),
  контекст и обучение на исправлениях через AX API.
- Логика PTT — чистый автомат (`ptt.rs`), тесты портированы (7/7).
- Текст (субагент): vocab, visemes, personal (rusqlite), context, corrections (порт difflib,
  сверен с CPython на 3000 случайных парах), practice, cleanup (Claude/Ollama/basic). 28 тестов.
- Данные совместимы с Python-версией: settings.json, history.jsonl, клипы .npz (читаем
  savez_compressed, пишем то, что читает numpy — проверено в обе стороны).
- `lipflow selftest VIDEO`: видеофайл вместо камеры, «нажатие» клавиши по таймеру — прогоняет
  весь путь UI (превью, хвост, финал, HUD, история, клип). Прошёл: encode 0.10 s + beam 0.85 s.

### ⚠ Обучение на Metal перезагружает Mac

- Запуск шага обучения (candle 0.11, Metal backward через conv front end) дважды
  перезагрузил машину пользователя (02.10.2026). Обучение на GPU запрещено в коде
  (`Trainer::new` отказывает не-CPU). Не запускать GPU-эксперименты без явного согласия.
- Граф обучения на CPU сверен с PyTorch: loss/CTC/att/нормы градиентов совпадают до 4–5 знака.
  Проблема CPU — скорость: ~9 s на forward+backward клипа из 192 кадров.

### 5. Обучение на Rust (CPU) — корректно, но медленнее; оптимизацию остановил

- Граф обучения (frontend+encoder1, 0.1·CTC + 0.9·label smoothing, AdamW, clip 5): совпадает с
  PyTorch до 4–5 знака на loss/градиентах (детерминированный режим).
- Скорость шага (клип 192 кадра): Rust CPU 8.7 s, PyTorch CPU 4.4 s. Своя im2col-свёртка с
  быстрым backward — без выигрыша (сэмплы: время размазано по однопоточным поэлементным op
  candle). Metal — перезагрузка машины. По правилу задачи («не быстрее/не точнее — стоп»)
  дальше не оптимизирую. Тренер подключён в онбординг (работает без Python), это медленнее
  Python/MPS. Пик памяти ~14 GB (im2col-матрицы сохраняются для backward).
- Отличие от Python: нет dropout при дообучении (Python учит в train-режиме с dropout 0.1).
- Персональная LM (`train-lm`) не портирована: тот же тренировочный стек, выигрыша не жду;
  LM, обученная Python-версией (`lm_phrasing.pth`), Rust-версией загружается.

### 6. В работе (код написан, проверка — после завершения прогона обучения)

- Локальная LLM для cleanup (`crates/llm`): Qwen3-0.6B GGUF Q8 на candle, токенизатор
  `tokenizers` (fancy-regex, без C), скачивание при первом запуске, «auto» без ключа → local.
- Whisper-режим: аудио-front end ResNet1D, AV-модель (encoder + aux_encoder + fusion), микрофон
  на AVAudioEngine, `segment` с привязкой к часам видео (тест из Python портирован), ресемплинг.
- `lipflow doctor`, `lipflow import-wispr`, `scripts/setup.sh`, `scripts/make_app.sh`, лог в
  ~/Library/Logs/Lipflow.log при запуске из бандла.

### Решение (пользователь, 02.10.2026): дообучение остаётся на Python

- «Если на Python и быстрее и качественнее — оставляй на нём; нужны только веса».
- Приложение (Rust) на шаге «Train» запускает оригинальный `train_on_face` через
  `uv run --project <python checkout> python scripts/train_face.py` и подхватывает веса
  (`vsr_face.pth`, `lm_phrasing.pth` — Rust их читает). Rust-тренер (`crates/vsr/src/train.rs`)
  в приложении не используется; оставлен как сверенный с PyTorch код, GPU-путь запрещён.

### 7. Локальная LLM, whisper-режим, упаковка — проверено

- Локальная LLM (candle, Qwen3-0.6B GGUF Q8): 0.31 s на вызов против 0.38–0.41 s у Python/MLX,
  тот же ответ на одинаковом входе. Python в рантайме не нужен.
- Whisper (AV-модель): выход слитого encoder против PyTorch max|Δ| 3e-5 (масштаб 23.7), тот же
  текст; encode 0.19 s + beam 0.55 s. Аудио-путь (`segment`, ресемплинг) покрыт тестами;
  живой микрофон не проверял (нужен запрос разрешения у пользователя).
- `scripts/make_app.sh` собирает бандл (ad-hoc подпись app.lipflow.Lipflow, лог в
  ~/Library/Logs/Lipflow.log); `lipflow doctor`, `import-wispr`, `train`/`train-lm` (через Python).
- Регрессия после всех правок: 38/38 гипотез = Python, selftest проходит, тесты воркспейса зелёные.

### 8. Русский язык: MultiVSR вместо Auto-AVSR (02.10.2026)

Пользователь после живой проверки: «самое главное — русский язык». Auto-AVSR обучена только на
английском (LRS3), поэтому вместо неё взята MultiVSR (Prajwal, Hegde, Zisserman 2025,
github.com/Sindhu-Hegde/multivsr). Это открытые веса на 13 языков, у русского WER 39.5% на их
тесте. MuAViC (Meta) обучена всего на 49 часах русского, у MultiVSR их 845.

- Сначала оригинал на Python проверен на видео пользователя (5 фраз; беззвучно и вслух), без
  дообучения: WER 70% / 57%. Направление пригодное: у английской модели на английских фразах
  пользователя было 106%.
- Порт на candle (`crates/vsr/src/multivsr.rs`): VTP (3D CNN, linear-attention transformers,
  attention pooling), Transformer 12+12 с KV-кэшем, beam search как в Joey NMT, декодер
  токенов Whisper. Сверка с PyTorch (`tools/multivsr_ref.py`, `mvsr_check`): признаки
  max|Δ| 1.3e-5, энкодер 7e-6; greedy, beam 5 и beam 20 дают те же токены.
- Скорость (16 с видео, M1 Pro): PyTorch CPU 29 с (VTP 4.8 + beam20 24). Rust Metal 3.8 с
  целиком (VTP 3.1, beam5 0.6, beam20 1.0). Linear attention без перестановки голов
  (блочно-диагональный kᵀv) дал VTP 4.6 → 3.1 с: strided-копии candle на Metal медленные.
- Кроп лица по нашим landmarks вместо S3FD + syncnet. Рамка подогнана по 791 кадру
  (`tools/face_box_fit.py`), медиана по 13 кадрам, фрагмент 176 px при записи. WER на видео
  пользователя: 52% / 52% против 70% / 57% на кропах S3FD оригинала.
- В приложении язык выбирается в настройках, по умолчанию русский, если модель установлена.
  Русские промпты чистки, 60 русских тренировочных фраз, клипы в `clips/onboarding-ru`
  (лица T×96×96×3). Признаки VTP считаются во время записи (превью), после отпускания
  досчитываются ~8 кадров. Итог: клип 6.6 с → текст за 1.07 с вместе с чисткой (было 1.8 с).
  Кусочный подсчёт против целого: max|Δ| 9.5e-7.
- Дообучение для русского: `scripts/train_face_ru.py`. Модель описана заново на PyTorch
  с теми же именами тензоров: признаки совпадают точно, greedy-токены те же, токенизатор
  равен оригинальному. Обучение только на CPU, четверть клипов отложена, веса сохраняются,
  только если стало лучше. Rust накладывает `models/multivsr_face.safetensors` поверх базы.
  Смоук-тест на 10 клипах из двух видео прошёл (2 эпохи ≈ 1 мин). Оценки точности пока нет:
  нужны настоящие 24 русские фразы пользователя.
- Пока нет для русского: whisper-режим (AV-модель английская).

## Не сделано / не проверено

- Живая проверка с реальными разрешениями (хоткей Input Monitoring, вставка Accessibility,
  камера, микрофон) — из терминала агента разрешений нет; нужен запуск бандла пользователем.
- Windows-версия не портирована (здесь нет Windows для сборки и проверки).
