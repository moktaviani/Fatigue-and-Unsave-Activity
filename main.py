"""
python main.py --source webcam
python main.py --source path/ke/video.mp4
python main.py --source path/ke/gambar.jpg
--no-unsafe-alert   
--no-fatigue-alert 
"""

import argparse
import os
import time

import cv2
import numpy as np
import pygame
from ultralytics import YOLO

import config
from event_tracker import EventTracker
from utils import FPSCounter, ResourceMonitor, draw_resources

pygame.mixer.init()

sound_fatigue_1 = pygame.mixer.Sound(config.AUDIO_ALERT_1)
sound_fatigue_2 = pygame.mixer.Sound(config.AUDIO_ALERT_2)


def make_beep_sound(freq=1200, duration_ms=350, volume=0.5):
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave = np.sin(freq * t * 2 * np.pi)
    audio = (wave * 32767 * volume).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([audio, audio]))
    return pygame.sndarray.make_sound(stereo)


if config.UNSAFE_ALERT_SOUND_PATH:
    sound_unsafe = pygame.mixer.Sound(config.UNSAFE_ALERT_SOUND_PATH)
else:
    sound_unsafe = make_beep_sound(
        config.UNSAFE_ALERT_BEEP_FREQ_HZ, config.UNSAFE_ALERT_BEEP_DURATION_MS
    )


def play_fatigue_alert(occurrence_count):
    if occurrence_count % 3 == 0:
        sound_fatigue_2.play()
    else:
        sound_fatigue_1.play()


def play_unsafe_alert():
    sound_unsafe.play()


def get_best_detection_per_class(result, class_names, conf_threshold):
    detected = {name: False for name in class_names}
    best_conf = {name: 0.0 for name in class_names}

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_names[cls_id]
            if conf >= conf_threshold:
                detected[name] = True
                best_conf[name] = max(best_conf[name], conf)

    return detected, best_conf


def get_unsafe_detections(result, model_names, conf_threshold):
    names, confs = [], []
    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                names.append(model_names[int(box.cls[0])])
                confs.append(conf)
    return names, confs


def draw_fatigue_boxes(frame, result, class_names):
    if result.boxes is None:
        return frame
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = class_names[cls_id]
        if conf < config.CONF_THRESHOLD_FATIGUE:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = config.FATIGUE_BOX_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def draw_unsafe_boxes(frame, result, model_names):
    if result.boxes is None:
        return frame
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model_names[cls_id]
        if conf < config.CONF_THRESHOLD_UNSAFE:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = config.UNSAFE_BOX_COLORS.get(name, config.UNSAFE_BOX_COLOR_DEFAULT)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def draw_status_panel(frame, eye_tracker, mouth_tracker, now,
                       mouth_alert, eye_alert,
                       unsafe_alert_active, unsafe_names_active):
    h, w = frame.shape[:2]
    panel_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Mata tertutup: {eye_tracker.event_count}x "
                        f"(durasi saat ini: {eye_tracker.current_duration(now):.1f}s)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Menguap: {mouth_tracker.event_count}x "
                        f"(durasi saat ini: {mouth_tracker.current_duration(now):.1f}s)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    y = 85
    if mouth_alert:
        cv2.putText(frame, "PENGENDARA LELAH", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        y += 30

    if eye_alert:
        cv2.putText(frame, "PENGENDARA MENGANTUK,", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "HATI-HATI DAN BERISTIRAHATLAH!", (10, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 50

    if unsafe_alert_active:
        label = "AKTIVITAS BERBAHAYA: " + ", ".join(unsafe_names_active)
        cv2.putText(frame, label, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return frame


def run_stream(fatigue_model, unsafe_model, source, is_webcam, args):
    cap = cv2.VideoCapture(0 if is_webcam else source)
    if not cap.isOpened():
        print(f"Gagal membuka sumber video: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    source_name = "webcam.mp4" if is_webcam else os.path.basename(source)
    output_path = get_output_path(source_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Hasil video akan disimpan ke: {output_path}")

    eye_tracker = EventTracker(
        config.EYE_CLOSED_DURATION, config.EYE_CLOSED_EVENT_LIMIT, config.MISS_TOLERANCE)
    mouth_tracker = EventTracker(
        config.MOUTH_OPEN_DURATION, config.MOUTH_OPEN_EVENT_LIMIT, config.MISS_TOLERANCE)

    fps_counter = FPSCounter()
    resource_monitor = ResourceMonitor(interval=5)

    unsafe_alert_classes = set(config.UNSAFE_ALERT_CLASSES)
    last_unsafe_alert_time = 0.0
    unsafe_alert_active_until = 0.0
    unsafe_names_active = []

    print("Tekan 'q' untuk keluar, 'r' untuk reset counter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time() if is_webcam else (frame_idx / fps)

        fatigue_result = fatigue_model.predict(
            frame, conf=config.CONF_THRESHOLD_FATIGUE, verbose=False)[0]
        unsafe_result = unsafe_model.predict(
            frame, conf=config.CONF_THRESHOLD_UNSAFE, verbose=False)[0]

        detected_fatigue, _ = get_best_detection_per_class(
            fatigue_result, config.FATIGUE_CLASS_NAMES, config.CONF_THRESHOLD_FATIGUE)

        eye_tracker.update(detected_fatigue['closed_eye'], now)
        mouth_tracker.update(detected_fatigue['open_mouth'], now)

        eye_tracker.reset_if_idle(now, config.RESET_WINDOW)
        mouth_tracker.reset_if_idle(now, config.RESET_WINDOW)

        mouth_alert = mouth_tracker.is_alert(now, config.ALERT_DISPLAY_DURATION)
        if mouth_tracker.just_triggered and not args.no_fatigue_alert:
            play_fatigue_alert(mouth_tracker.alert_occurrence_count)

        eye_alert = eye_tracker.is_alert(now, config.ALERT_DISPLAY_DURATION)
        if eye_tracker.just_triggered and not args.no_fatigue_alert:
            play_fatigue_alert(eye_tracker.alert_occurrence_count)

        unsafe_names, unsafe_confs = get_unsafe_detections(
            unsafe_result, unsafe_model.names, config.CONF_THRESHOLD_UNSAFE)
        triggered_unsafe = [n for n in unsafe_names if n in unsafe_alert_classes]

        if args.debug and unsafe_names:
            print(f"[DEBUG unsafe] {list(zip(unsafe_names, [f'{c:.2f}' for c in unsafe_confs]))}")

        if (triggered_unsafe and not args.no_unsafe_alert
                and (now - last_unsafe_alert_time) >= config.UNSAFE_ALERT_COOLDOWN):
            print(f"[ALERT] Aktivitas berbahaya: {', '.join(sorted(set(triggered_unsafe)))}")
            play_unsafe_alert()
            last_unsafe_alert_time = now
            unsafe_alert_active_until = now + config.ALERT_DISPLAY_DURATION
            unsafe_names_active = sorted(set(triggered_unsafe))

        unsafe_alert_active = now < unsafe_alert_active_until

        fps_now = fps_counter.tick()
        resource_info = resource_monitor.sample(fps=fps_now)

        frame = draw_fatigue_boxes(frame, fatigue_result, config.FATIGUE_CLASS_NAMES)
        frame = draw_unsafe_boxes(frame, unsafe_result, unsafe_model.names)
        frame = draw_status_panel(frame, eye_tracker, mouth_tracker, now,
                                   mouth_alert, eye_alert,
                                   unsafe_alert_active, unsafe_names_active)
        frame = draw_resources(frame, resource_info)

        out_writer.write(frame)
        cv2.imshow("Deteksi Kantuk & Aktivitas Berbahaya Pengendara", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            eye_tracker.reset()
            mouth_tracker.reset()
            unsafe_alert_active_until = 0.0
            print("Counter di-reset.")

        frame_idx += 1

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"\nVideo tersimpan di: {output_path}")


def run_image(fatigue_model, unsafe_model, image_path, args):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    fatigue_result = fatigue_model.predict(
        frame, conf=config.CONF_THRESHOLD_FATIGUE, verbose=False)[0]
    unsafe_result = unsafe_model.predict(
        frame, conf=config.CONF_THRESHOLD_UNSAFE, verbose=False)[0]

    frame = draw_fatigue_boxes(frame, fatigue_result, config.FATIGUE_CLASS_NAMES)
    frame = draw_unsafe_boxes(frame, unsafe_result, unsafe_model.names)

    output_path = get_output_path(image_path)
    cv2.imwrite(output_path, frame)
    print(f"\nHasil gambar disimpan ke: {output_path}")

    detected_fatigue, best_conf_fatigue = get_best_detection_per_class(
        fatigue_result, config.FATIGUE_CLASS_NAMES, config.CONF_THRESHOLD_FATIGUE)
    print("\nHasil deteksi kantuk/lelah:")
    for name in config.FATIGUE_CLASS_NAMES:
        status = (f"terdeteksi (conf={best_conf_fatigue[name]:.2f})"
                  if detected_fatigue[name] else "tidak terdeteksi")
        print(f"  {name}: {status}")

    unsafe_names, unsafe_confs = get_unsafe_detections(
        unsafe_result, unsafe_model.names, config.CONF_THRESHOLD_UNSAFE)
    print("\nHasil deteksi aktivitas berbahaya:")
    if not unsafe_names:
        print("  tidak ada aktivitas berbahaya terdeteksi")
    else:
        for name, conf in zip(unsafe_names, unsafe_confs):
            print(f"  {name}: terdeteksi (conf={conf:.2f})")

    if not args.no_unsafe_alert and any(n in config.UNSAFE_ALERT_CLASSES for n in unsafe_names):
        play_unsafe_alert()

    cv2.imshow("Hasil Deteksi", frame)
    print("Tekan tombol apapun di jendela gambar untuk keluar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_output_path(input_path):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1] if input_path else ".mp4"
    return os.path.join(config.OUTPUT_DIR, f"hasil_{base_name}{ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Deteksi kantuk/lelah + aktivitas berbahaya (merokok/vape/HP) pengendara")
    parser.add_argument("--source", required=True,
                         help="'webcam', atau path ke file video/gambar")
    parser.add_argument("--fatigue-model", default=config.FATIGUE_MODEL_PATH,
                         help="path ke model .pt utk deteksi kantuk/lelah")
    parser.add_argument("--unsafe-model", default=config.UNSAFE_MODEL_PATH,
                         help="path ke model .pt utk deteksi aktivitas berbahaya")
    parser.add_argument("--no-fatigue-alert", action="store_true",
                         help="matikan suara alert kantuk/lelah")
    parser.add_argument("--no-unsafe-alert", action="store_true",
                         help="matikan suara alert aktivitas berbahaya")
    parser.add_argument("--debug", action="store_true",
                         help="print semua deteksi aktivitas berbahaya tiap frame")
    args = parser.parse_args()

    print(f"Load model kantuk/lelah   : {args.fatigue_model}")
    fatigue_model = YOLO(args.fatigue_model)
    print(f"Load model aktivitas bahaya: {args.unsafe_model}")
    unsafe_model = YOLO(args.unsafe_model)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    video_exts = (".mp4", ".avi", ".mov", ".mkv")

    if args.source.lower() == "webcam":
        run_stream(fatigue_model, unsafe_model, source=0, is_webcam=True, args=args)
    elif args.source.lower().endswith(image_exts):
        run_image(fatigue_model, unsafe_model, args.source, args)
    elif args.source.lower().endswith(video_exts):
        run_stream(fatigue_model, unsafe_model, source=args.source, is_webcam=False, args=args)
    else:
        print("Format sumber tidak dikenali. Gunakan 'webcam', file gambar "
              "(.jpg/.png), atau file video (.mp4/.avi/.mov).")


if __name__ == "__main__":
    main()
