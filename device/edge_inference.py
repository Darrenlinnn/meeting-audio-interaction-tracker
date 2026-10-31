# ============================================================
# Core2 UIFlow Inference + AWS IoT MQTT Upload
# Low-Latency 1-sec Temporal MLP
# Classes: silence / single_speaker / multiple_speakers
# Required:
#   /flash/mlp_model_export.json
#   /flash/certificate/device_cert.pem
#   /flash/certificate/private_key.pem
#   /flash/certificate/AmazonRootCA1.pem
# ============================================================

import M5
from M5 import *
import time, math, json, ustruct, gc
import network
import ssl
import ntptime
from umqtt.simple import MQTTClient

# =========================
# AWS / WIFI CONFIG
# =========================
WIFI_SSID     = "<your-wifi-ssid>"
WIFI_PASSWORD = "<your-wifi-password>"

AWS_ENDPOINT  = "<your-endpoint>-ats.iot.<region>.amazonaws.com"
CLIENT_ID     = "Core2Kit"
TOPIC         = "devices/" + CLIENT_ID + "/predictions"

CERT_PATH = "/flash/certificate/device_cert.pem"
KEY_PATH  = "/flash/certificate/private_key.pem"
CA_PATH   = "/flash/certificate/AmazonRootCA1.pem"

# =========================
# MODEL CONFIG
# =========================
SAMPLE_RATE = 8000
WIN_SEC = 1
WIN_SAMPLES = SAMPLE_RATE * WIN_SEC
BUF_BYTES = WIN_SAMPLES * 2

MODEL_PATH = "/flash/mlp_model_export.json"

# =========================
# UI INIT
# =========================
M5.begin()
Widgets.fillScreen(0x000000)

Speaker.begin()
Speaker.setVolumePercentage(0)
Speaker.end()

def show_status(title, line2="", line3="", color=0xFFFFFF):
    Widgets.fillScreen(0x000000)
    Widgets.Label(title, 20, 40, 1.0, color, 0x000000, Widgets.FONTS.DejaVu24)
    if line2:
        Widgets.Label(line2, 20, 105, 1.0, 0xFFFFFF, 0x000000, Widgets.FONTS.DejaVu18)
    if line3:
        Widgets.Label(line3, 20, 160, 1.0, 0xAAAAAA, 0x000000, Widgets.FONTS.DejaVu18)
    M5.update()

# =========================
# WIFI + AWS FUNCTIONS
# =========================
def connect_wifi():
    show_status("WiFi", "Connecting...", WIFI_SSID, 0x00CCFF)
    print("Connecting to WiFi...")

    wlan = network.WLAN(network.STA_IF)
    wlan.active(False)
    time.sleep(1)
    wlan.active(True)

    if not wlan.isconnected():
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)

        timeout = 20
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1
            print("Waiting WiFi... {} sec left".format(timeout))

    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print("WiFi connected:", ip)
        show_status("WiFi OK", ip, "", 0x00FF00)
        time.sleep(1)
    else:
        show_status("WiFi Failed", "Check SSID", "", 0xFF0000)
        raise RuntimeError("WiFi connection failed.")

def sync_time():
    show_status("NTP", "Syncing time...", "", 0x00CCFF)
    print("Syncing clock via NTP...")

    for attempt in range(5):
        try:
            ntptime.settime()
            print("Time synced:", time.localtime())
            show_status("Time OK", "NTP synced", "", 0x00FF00)
            time.sleep(1)
            return
        except Exception as e:
            print("NTP attempt failed:", e)
            time.sleep(2)

    print("WARNING: NTP sync failed, continuing anyway")
    show_status("NTP Warning", "Continue anyway", "", 0xFFFF00)
    time.sleep(1)

def connect_aws():
    print("Building SSL context...")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_mode = ssl.CERT_OPTIONAL
    context.load_cert_chain(CERT_PATH, KEY_PATH)

    print("Connecting to AWS IoT Core...")

    client = MQTTClient(
        client_id=CLIENT_ID,
        server=AWS_ENDPOINT,
        port=8883,
        keepalive=1200,
        ssl=context
    )

    client.connect()
    print("Connected to AWS IoT Core!")
    return client

def publish_prediction(client, pred_label, conf, probs):

    prob_dict = {}
    for i in range(len(probs)):
        prob_dict[LABEL_MAP[i]] = round(probs[i], 3)

    payload = json.dumps({
        "device_id": CLIENT_ID,
        "timestamp": time.time(),
        "predicted_class": pred_label,
        "confidence": round(conf, 3),
        "probabilities": prob_dict,
        "model_version": "mlp_1sec_v1"
    })

    client.publish(TOPIC, payload.encode())
    print("Published to AWS:", payload)

# =========================
# LOAD MODEL
# =========================
connect_wifi()
sync_time()
client = connect_aws()

show_status("Loading", "Model...", "", 0xFFFFFF)

with open(MODEL_PATH, "r") as f:
    exported = json.load(f)

SC_MEAN = exported["scaler"]["mean"]
SC_SCALE = exported["scaler"]["scale"]
WEIGHTS = exported["model"]["weights"]
BIASES = exported["model"]["biases"]
FEATURE_ORDER = exported["feature_order"]

LABEL_MAP = {}
for k in exported["label_map"]:
    LABEL_MAP[int(k)] = exported["label_map"][k]

print("Feature order:", FEATURE_ORDER)
print("Label map:", LABEL_MAP)

# =========================
# FEATURE EXTRACTION
# =========================
def read_i16(buf, i):
    return ustruct.unpack_from("<h", buf, i * 2)[0] / 32768.0

def extract_features_from_buffer(buf):
    n = WIN_SAMPLES

    abs_sum = 0.0
    sq_sum = 0.0
    mean_sum = 0.0
    peak = 0.0
    zc = 0

    prev = read_i16(buf, 0)

    for i in range(n):
        v = read_i16(buf, i)

        mean_sum += v
        av = abs(v)
        abs_sum += av
        sq_sum += v * v

        if av > peak:
            peak = av

        if i > 0:
            if (prev >= 0 and v < 0) or (prev < 0 and v >= 0):
                zc += 1
        prev = v

    mean = mean_sum / n
    mean_abs_amp = abs_sum / n
    rms_energy = math.sqrt(sq_sum / n)
    peak_amp = peak
    zero_crossing_rate = zc / (n - 1)

    var_sum = 0.0
    for i in range(n):
        v = read_i16(buf, i)
        d = v - mean
        var_sum += d * d
    variance = var_sum / n

    n_subsegments = 10
    seg_len = n // n_subsegments

    sub_rms = []

    for s in range(n_subsegments):
        start = s * seg_len
        end = start + seg_len

        seg_sq_sum = 0.0
        for i in range(start, end):
            v = read_i16(buf, i)
            seg_sq_sum += v * v

        sub_rms.append(math.sqrt(seg_sq_sum / seg_len))

    sub_rms_mean = sum(sub_rms) / len(sub_rms)

    sub_rms_var = 0.0
    for v in sub_rms:
        d = v - sub_rms_mean
        sub_rms_var += d * d
    sub_rms_std = math.sqrt(sub_rms_var / len(sub_rms))

    sub_rms_max = max(sub_rms)
    sub_rms_min = min(sub_rms)
    sub_rms_range = sub_rms_max - sub_rms_min

    active_threshold = max(0.002, 0.25 * sub_rms_mean)

    active_flags = []
    for v in sub_rms:
        active_flags.append(v > active_threshold)

    active_count = 0
    for flag in active_flags:
        if flag:
            active_count += 1
    active_ratio = active_count / len(active_flags)

    max_streak = 0
    current_streak = 0

    for flag in active_flags:
        if flag:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0

    max_active_streak = max_streak

    feature_dict = {
        "mean_abs_amp": mean_abs_amp,
        "rms_energy": rms_energy,
        "peak_amp": peak_amp,
        "variance": variance,
        "zero_crossing_rate": zero_crossing_rate,
        "sub_rms_std": sub_rms_std,
        "sub_rms_range": sub_rms_range,
        "active_ratio": active_ratio,
        "max_active_streak": max_active_streak,
    }

    return [feature_dict[name] for name in FEATURE_ORDER]

# =========================
# MLP INFERENCE
# =========================
def dense(x, W, b):
    out = []
    for j in range(len(W[0])):
        s = b[j]
        for i in range(len(x)):
            s += x[i] * W[i][j]
        out.append(s)
    return out

def relu(x):
    return [v if v > 0 else 0.0 for v in x]

def softmax(z):
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    total = sum(exps)
    return [v / total for v in exps]

def predict(raw_features):
    x = []

    for i in range(len(raw_features)):
        scale = SC_SCALE[i]
        if scale == 0:
            scale = 1.0
        x.append((raw_features[i] - SC_MEAN[i]) / scale)

    for layer_i in range(len(WEIGHTS)):
        x = dense(x, WEIGHTS[layer_i], BIASES[layer_i])
        if layer_i < len(WEIGHTS) - 1:
            x = relu(x)

    probs = softmax(x)
    pred_id = probs.index(max(probs))
    pred_label = LABEL_MAP[pred_id]
    conf = probs[pred_id]

    return pred_label, conf, probs

# =========================
# DISPLAY
# =========================
def show_prediction(label, conf, mqtt_ok=True):
    if label == "silence":
        title = "SILENCE"
        color = 0x00CCFF
    elif label == "single_speaker":
        title = "SINGLE"
        color = 0x33CC33
    else:
        title = "MULTIPLE"
        color = 0xFF00AA

    mqtt_text = "AWS: sent" if mqtt_ok else "AWS: failed"
    mqtt_color = 0x00FF00 if mqtt_ok else 0xFF0000

    Widgets.fillScreen(0x000000)
    Widgets.Label(title, 20, 35, 1.0, color, 0x000000, Widgets.FONTS.DejaVu24)
    Widgets.Label("Conf: {}%".format(int(conf * 100)), 20, 90, 1.0, 0xFFFFFF, 0x000000, Widgets.FONTS.DejaVu18)
    Widgets.Label("1-sec inference", 20, 135, 1.0, 0xAAAAAA, 0x000000, Widgets.FONTS.DejaVu18)
    Widgets.Label(mqtt_text, 20, 180, 1.0, mqtt_color, 0x000000, Widgets.FONTS.DejaVu18)
    M5.update()

# =========================
# MAIN LOOP
# =========================
show_status("Ready", "Starting mic...", "", 0x00FF00)

audio_buf = bytearray(BUF_BYTES)

Mic.begin()
time.sleep_ms(1000)

last_shown_label = None
last_mqtt_ok = True

while True:
    gc.collect()

    try:
        Mic.record(audio_buf, SAMPLE_RATE, False)

        while Mic.isRecording():
            M5.update()
            time.sleep_ms(30)

        feats = extract_features_from_buffer(audio_buf)
        pred_label, conf, probs = predict(feats)

        # Fast silence override
        rms_energy = feats[FEATURE_ORDER.index("rms_energy")]
        mean_abs_amp = feats[FEATURE_ORDER.index("mean_abs_amp")]

        if rms_energy < 0.006 and mean_abs_amp < 0.004:
            pred_label = "silence"
            conf = max(conf, 0.95)

        # Publish to AWS IoT Core
        mqtt_ok = True
        try:
            publish_prediction(client, pred_label, conf, probs)
        except Exception as e:
            mqtt_ok = False
            print("MQTT publish error:", e)

            try:
                print("Trying to reconnect AWS...")
                client = connect_aws()
                print("AWS reconnected.")
            except Exception as e2:
                print("AWS reconnect failed:", e2)

        # Refresh screen only when label or MQTT status changes
        if pred_label != last_shown_label or mqtt_ok != last_mqtt_ok:
            show_prediction(pred_label, conf, mqtt_ok)
            last_shown_label = pred_label
            last_mqtt_ok = mqtt_ok

        print("[pred]", pred_label, round(conf, 3), "mqtt_ok=", mqtt_ok)

        time.sleep_ms(50)

    except KeyboardInterrupt:
        print("Stopped by user.")
        try:
            client.disconnect()
        except:
            pass
        break

    except Exception as e:
        print("Main loop error:", e)
        show_status("Error", str(e), "Retrying...", 0xFF0000)
        time.sleep(3)