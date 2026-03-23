# ============================================================
# SafeGuard AI — Main Processing Pipeline
# ============================================================
# Active signal stack:
#   1. Visual violence  — ResNet18 (predict_batch)
#   2. Audio toxicity   — Whisper + DistilBERT
#   3. Speech aggression— Wav2Vec2 binary classifier
#   4. Emotion          — Audio CNN (mel spectrogram)
#
# Face emotion REMOVED — unreliable on arbitrary videos
# ============================================================

import os
import numpy as np
from datetime import datetime

from utils.config import (
    VISUAL_WEIGHT, AUDIO_WEIGHT, FRAME_SAMPLE_RATE,
    VIOLENCE_THRESHOLD, TOXICITY_THRESHOLD, TEMPORAL_WINDOW
)
from preprocessing.video_processing  import extract_frames, temporal_smooth
from preprocessing.audio_processing  import extract_audio_from_video, record_microphone
from models.visual_model              import get_visual_model
from models.audio_model               import get_audio_model
from models.emotion_model             import get_emotion_model
from models.aggression_model          import get_aggression_model
from alerts.alert_manager             import compute_fused_score, determine_alert_level, get_alert_manager
from database.db_manager              import insert_incident


def analyze_video(video_path: str, source_label: str = "UPLOAD", wav_path_override: str | None = None) -> dict:
    result        = _empty_result(source_label)
    score_history = []

    # ── 1 & 2: Visual ────────────────────────────────────────
    frames = []
    try:
        frames = extract_frames(video_path, sample_fps=FRAME_SAMPLE_RATE)
        if frames:
            visual_model = get_visual_model()
            batch_scores = visual_model.predict_batch(frames, batch_size=16)
            mean_score   = float(np.mean(batch_scores)) if batch_scores else 0.0
            result["visual_score"] = round(mean_score, 4)
            result["frame_scores"] = [round(s, 4) for s in batch_scores]
            result["frame_count"]  = len(frames)
    except Exception as e:
        result["errors"].append(f"Visual processing: {e}")

    # ── 3, 4, 5: Audio + Transcript + Toxicity ───────────────
    wav_path           = None
    extracted_temp_wav = False
    try:
        if wav_path_override and os.path.isfile(wav_path_override):
            wav_path = wav_path_override
        else:
            wav_path = extract_audio_from_video(video_path)
            extracted_temp_wav = True
        audio_result = get_audio_model().analyze(wav_path)
        result["transcript"]     = audio_result["transcript"]
        result["toxicity_score"] = audio_result["toxicity_score"]
        result["toxicity_label"] = audio_result["toxicity_label"]
        result["toxic_phrases"]  = audio_result["toxic_phrases"]
    except Exception as e:
        result["errors"].append(f"Audio processing: {e}")

    # ── 5b: Aggression detection (Wav2Vec2) ───────────────────
    try:
        if wav_path and os.path.isfile(wav_path):
            ag = get_aggression_model().predict(wav_path)
            result["aggression_score"]     = ag["aggression_score"]
            result["is_speech_aggressive"] = ag["is_aggressive"]
    except Exception as e:
        result["errors"].append(f"Aggression detection: {e}")

    # ── 6: Emotion ────────────────────────────────────────────
    try:
        emotion_model = get_emotion_model()
        emo_audio     = None

        if wav_path and os.path.isfile(wav_path):
            emo_audio = emotion_model.predict_from_wav(wav_path)

        emo_transcript = emotion_model.predict_from_transcript(
            result["transcript"], result["toxicity_score"]
        )
        result["transcript_emotion"]      = emo_transcript.get("emotion", "neutral")
        result["transcript_emotion_conf"] = float(emo_transcript.get("confidence", 0.0) or 0.0)

        no_speech  = not (result.get("transcript") or "").strip()
        has_speech = not no_speech
        if no_speech and float(result.get("toxicity_score", 0.0) or 0.0) == 0.0:
            emo_audio = None

        chosen = None
        source = "transcript"

        if has_speech:
            chosen = emo_transcript
            source = "transcript"
        elif emo_audio is not None and not getattr(emotion_model, "demo_mode", True):
            if float(emo_audio.get("confidence", 0.0) or 0.0) >= 0.45:
                chosen = emo_audio
                source = "audio"

        if chosen is None:
            chosen = emo_transcript
            source = "transcript"

        result["emotion"]        = chosen.get("emotion", "neutral")
        result["emotion_conf"]   = float(chosen.get("confidence", 0.0) or 0.0)
        result["emotion_probs"]  = chosen.get("probabilities", {}) or {}
        result["is_aggressive"]  = bool(chosen.get("is_aggressive", False))
        result["emotion_source"] = source
    except Exception as e:
        result["errors"].append(f"Emotion detection: {e}")

    if extracted_temp_wav and wav_path and os.path.isfile(wav_path):
        try: os.remove(wav_path)
        except: pass

    return _finalize(result, source_label, score_history)


def analyze_webcam(duration: int = 5, cam_index: int = 0) -> dict:
    from preprocessing.video_processing import capture_webcam_frames
    result        = _empty_result("LIVE")
    score_history = []
    frames        = []
    wav_path      = None

    try:
        import threading
        mic_result = {}

        def capture_audio():
            try:
                path = record_microphone(duration)
                mic_result["path"] = path
            except Exception as e:
                mic_result["error"] = str(e)

        audio_thread = threading.Thread(target=capture_audio, daemon=True)
        audio_thread.start()
        frames = capture_webcam_frames(duration, cam_index, FRAME_SAMPLE_RATE)
        audio_thread.join(timeout=duration + 5)
        wav_path = mic_result.get("path")
    except Exception as e:
        result["errors"].append(f"Capture: {e}")

    if frames:
        try:
            visual_model = get_visual_model()
            batch_scores = visual_model.predict_batch(frames, batch_size=16)
            result["visual_score"] = round(float(np.mean(batch_scores)), 4)
            result["frame_scores"] = [round(s, 4) for s in batch_scores]
            result["frame_count"]  = len(frames)
        except Exception as e:
            result["errors"].append(f"Visual: {e}")

    if wav_path and os.path.isfile(wav_path):
        try:
            audio_result = get_audio_model().analyze(wav_path)
            result["transcript"]     = audio_result["transcript"]
            result["toxicity_score"] = audio_result["toxicity_score"]
            result["toxic_phrases"]  = audio_result["toxic_phrases"]
        except Exception as e:
            result["errors"].append(f"Audio: {e}")

    if wav_path and os.path.isfile(wav_path):
        try:
            ag = get_aggression_model().predict(wav_path)
            result["aggression_score"]     = ag["aggression_score"]
            result["is_speech_aggressive"] = ag["is_aggressive"]
        except Exception as e:
            result["errors"].append(f"Aggression: {e}")

    try:
        emotion_model = get_emotion_model()
        emo_result    = None
        if wav_path and os.path.isfile(wav_path):
            emo_result = emotion_model.predict_from_wav(wav_path)
        if emo_result is None:
            emo_result = emotion_model.predict_from_transcript(
                result["transcript"], result["toxicity_score"]
            )
        result["emotion"]       = emo_result["emotion"]
        result["emotion_conf"]  = emo_result["confidence"]
        result["emotion_probs"] = emo_result["probabilities"]
        result["is_aggressive"] = emo_result["is_aggressive"]
    except Exception as e:
        result["errors"].append(f"Emotion: {e}")

    if wav_path and os.path.isfile(wav_path):
        try: os.remove(wav_path)
        except: pass

    return _finalize(result, "LIVE", score_history)


def analyze_audio_only(wav_path: str, source_label: str = "UPLOAD") -> dict:
    result = _empty_result(source_label)

    try:
        audio_result = get_audio_model().analyze(wav_path)
        result["transcript"]     = audio_result["transcript"]
        result["toxicity_score"] = audio_result["toxicity_score"]
        result["toxicity_label"] = audio_result["toxicity_label"]
        result["toxic_phrases"]  = audio_result["toxic_phrases"]
    except Exception as e:
        result["errors"].append(str(e))

    try:
        ag = get_aggression_model().predict(wav_path)
        result["aggression_score"]     = ag["aggression_score"]
        result["is_speech_aggressive"] = ag["is_aggressive"]
    except Exception as e:
        result["errors"].append(f"Aggression: {e}")

    try:
        emotion_model = get_emotion_model()
        emo_result    = emotion_model.predict_from_wav(wav_path)
        if emo_result is None:
            emo_result = emotion_model.predict_from_transcript(
                result["transcript"], result["toxicity_score"]
            )
        result["emotion"]       = emo_result["emotion"]
        result["emotion_conf"]  = emo_result["confidence"]
        result["emotion_probs"] = emo_result["probabilities"]
        result["is_aggressive"] = emo_result["is_aggressive"]
    except Exception as e:
        result["errors"].append(str(e))

    return _finalize(result, source_label, score_history=[])


# ── Helpers ───────────────────────────────────────────────────

def _empty_result(source: str) -> dict:
    return {
        "source":                  source,
        "timestamp":               datetime.now().isoformat(),
        "visual_score":            0.0,
        "toxicity_score":          0.0,
        "toxicity_label":          "non-toxic",
        "fused_score":             0.0,
        "alert_level":             "SAFE",
        "emotion":                 "neutral",
        "emotion_conf":            0.0,
        "emotion_probs":           {},
        "emotion_source":          "transcript",
        "transcript_emotion":      "neutral",
        "transcript_emotion_conf": 0.0,
        "is_aggressive":           False,
        "transcript":              "",
        "toxic_phrases":           [],
        "frame_scores":            [],
        "frame_count":             0,
        "aggression_score":        0.0,
        "is_speech_aggressive":    False,
        "errors":                  [],
        "incident_id":             None,
    }


def _finalize(result: dict, source: str, score_history: list) -> dict:
    score_history.append(result["visual_score"])
    smoothed_visual = temporal_smooth(score_history, TEMPORAL_WINDOW)

    visual = smoothed_visual
    audio  = result["toxicity_score"]
    fused  = 0.0 if (visual == 0.0 and audio == 0.0) else \
             compute_fused_score(visual, audio, VISUAL_WEIGHT, AUDIO_WEIGHT)

    result["visual_score"] = round(smoothed_visual, 4)
    result["fused_score"]  = round(fused, 4)

    alert_level = determine_alert_level(
        smoothed_visual,
        result["toxicity_score"],
        result["emotion"],
        aggression_score=result.get("aggression_score", 0.0),
    )
    result["alert_level"] = alert_level

    try:
        iid = insert_incident(
            violence_score   = smoothed_visual,
            emotion_detected = result["emotion"],
            toxicity_score   = result["toxicity_score"],
            transcript       = result["transcript"],
            toxic_phrases    = result["toxic_phrases"],
            video_source     = source,
            alert_level      = alert_level,
        )
        result["incident_id"] = iid
    except Exception as e:
        result["errors"].append(f"DB insert: {e}")

    if alert_level != "SAFE":
        get_alert_manager().trigger(
            violence_score   = smoothed_visual,
            emotion          = result["emotion"],
            toxicity_score   = result["toxicity_score"],
            toxic_phrases    = result["toxic_phrases"],
            source           = source,
            transcript       = result["transcript"],
            aggression_score = result.get("aggression_score", 0.0),
        )

    return result


# ════════════════════════════════════════════════════════════
# RECORD + ANALYZE
# ════════════════════════════════════════════════════════════
def record_and_analyze(duration: int, save_path: str) -> dict:
    import threading, time as _time
    wav_path = save_path.replace(".mp4", ".wav")
    rh       = {}

    def _capture():
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam_fps = cap.get(cv2.CAP_PROP_FPS) or 20
            cam_fps = max(10.0, min(float(cam_fps), 30.0))
            fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
            writer  = cv2.VideoWriter(save_path, fourcc, cam_fps, (640, 480))

            has_audio = False
            audio_buf = []
            try:
                import sounddevice as sd
                sr = 16000
                def _cb(indata, frames, t, status):
                    audio_buf.extend(indata[:, 0].tolist())
                stream = sd.InputStream(samplerate=sr, channels=1, callback=_cb)
                stream.start()
                has_audio = True
            except Exception as ae:
                rh["audio_warn"] = str(ae)

            t0 = _time.time()
            while _time.time() - t0 < duration:
                ret, frame = cap.read()
                if ret:
                    writer.write(frame)

            cap.release()
            writer.release()

            if has_audio:
                stream.stop(); stream.close()
                import numpy as np_i, scipy.io.wavfile as wio
                a   = np_i.array(audio_buf, dtype=np_i.float32)
                a   = np_i.clip(a, -1.0, 1.0)
                a16 = (a * 32767.0).astype(np_i.int16)
                wio.write(wav_path, sr, a16)
                rh["wav"] = wav_path

            rh["ok"] = True
        except Exception as e:
            rh["error"] = str(e)

    t = threading.Thread(target=_capture, daemon=True)
    t.start()
    t.join(timeout=duration + 15)

    if "error" in rh:
        raise RuntimeError(f"Recording failed: {rh['error']}")
    if not rh.get("ok"):
        raise RuntimeError("Recording timed out.")

    return analyze_video(save_path, "LIVE_RECORD")


# ════════════════════════════════════════════════════════════
# SYSTEM STATUS
# ════════════════════════════════════════════════════════════
def get_system_status() -> dict:
    results = {}

    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok  = cap.isOpened(); cap.release()
        results["camera"] = {"status":"ok" if ok else "err",
            "message":"Connected" if ok else "Not detected","detail":"Webcam / IP Camera"}
    except ImportError:
        results["camera"] = {"status":"warn","message":"opencv-python not installed","detail":"pip install opencv-python"}

    try:
        import sounddevice as sd
        inp = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        results["audio"] = {"status":"ok","message":f"{len(inp)} input device(s) found","detail":"Microphone / Audio Input"}
    except ImportError:
        results["audio"] = {"status":"warn","message":"sounddevice not installed","detail":"pip install sounddevice"}

    try:
        vm = get_visual_model()
        loaded = getattr(vm, "weights_loaded", False)
        results["visual"] = {"status":"ok" if loaded else "warn",
            "message":"ResNet18 weights loaded" if loaded else "Demo mode — no weights",
            "detail":"Violence Detection Model"}
    except Exception as e:
        results["visual"] = {"status":"err","message":str(e)[:80],"detail":"Violence Detection Model"}

    try:
        am = get_audio_model()
        db = getattr(am, "distilbert_available", False)
        wh = getattr(am, "whisper_model", None) is not None
        results["audio_model"] = {"status":"ok" if wh else "warn",
            "message":"DistilBERT + Whisper ready" if (db and wh) else ("Whisper ready" if wh else "Models not loaded"),
            "detail":"ASR + Toxicity Model"}
    except Exception as e:
        results["audio_model"] = {"status":"err","message":str(e)[:80],"detail":"ASR + Toxicity Model"}

    try:
        em = get_emotion_model()
        hw = getattr(em, "weights_loaded", False)
        results["emotion"] = {"status":"ok" if hw else "warn",
            "message":"CNN weights loaded" if hw else "Acoustic heuristic mode",
            "detail":"Emotion Detection Model"}
    except Exception as e:
        results["emotion"] = {"status":"err","message":str(e)[:80],"detail":"Emotion Detection Model"}

    try:
        ag = get_aggression_model()
        loaded = getattr(ag, "weights_loaded", False)
        results["aggression"] = {"status":"ok" if loaded else "warn",
            "message":"Wav2Vec2 aggression model loaded" if loaded else "Demo mode — place aggression_model.pth in models/",
            "detail":"Speech Aggression Detector (Wav2Vec2)"}
    except Exception as e:
        results["aggression"] = {"status":"err","message":str(e)[:80],"detail":"Speech Aggression Detector"}

    try:
        from database.db_manager import get_stats
        s = get_stats()
        results["database"] = {"status":"ok","message":f"SQLite connected — {s['total']} records","detail":"Incident Database"}
    except Exception as e:
        results["database"] = {"status":"err","message":str(e)[:80],"detail":"SQLite Database"}

    try:
        import torch
        cuda = torch.cuda.is_available()
        results["torch"] = {"status":"ok",
            "message":f"PyTorch {torch.__version__}  ·  {'GPU: '+torch.cuda.get_device_name(0) if cuda else 'CPU only'}",
            "detail":"Deep Learning Runtime"}
    except ImportError:
        results["torch"] = {"status":"err","message":"PyTorch not installed","detail":"Deep Learning Runtime"}

    try:
        rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        files   = [f for f in os.listdir(rec_dir) if f.endswith(".mp4")] if os.path.isdir(rec_dir) else []
        size_mb = sum(os.path.getsize(os.path.join(rec_dir, f)) for f in files) // 1024 // 1024
        results["storage"] = {"status":"ok","message":f"{len(files)} recording(s)  ·  {size_mb} MB used","detail":f"Folder: {rec_dir}"}
    except Exception as e:
        results["storage"] = {"status":"warn","message":str(e),"detail":"Recordings Storage"}

    return results