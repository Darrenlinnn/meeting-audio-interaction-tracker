# Records THREE 10-second audio clips — one per sound class:
#
#   1. "silence"            – quiet room / no speaking
#   2. "single_speaker"     – one person speaking continuously
#   3. "multiple_speakers"  – 2+ people speaking / overlapping discussion

import M5
from M5 import *
import time, struct, gc

SAMPLE_RATE      = 8000
BITS_PER_SAMPLE  = 16
NUM_CHANNELS     = 1
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8

CHUNK_SEC   = 5
NUM_CHUNKS  = 6
TOTAL_SEC   = CHUNK_SEC * NUM_CHUNKS

CHUNK_BYTES = SAMPLE_RATE * CHUNK_SEC * BYTES_PER_SAMPLE
TOTAL_BYTES = CHUNK_BYTES * NUM_CHUNKS

CLASSES = [
    (
        "SILENCE",
        "silence.wav",
        "Stay quiet.\nNo speaking.\nCapture room silence only.",
        "Stay quiet...",
        0x00CCFF
    ),
    (
        "SINGLE",
        "single_speaker.wav",
        "ONE person speaks\nfor 10 seconds.\nOthers stay silent.",
        "One person speaking...",
        0x33CC33
    ),
    (
        "MULTIPLE",
        "multiple_speakers.wav",
        "2+ people speak\nlike a discussion.\nInclude some overlap if possible.",
        "Multiple people speaking...",
        0xFF00AA
    ),
]

# ── WAV header ────────────────────────────────────────────────────────────────
def make_wav_header(sample_rate, bits_per_sample, num_channels, data_size):
    byte_rate   = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    h  = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    h += struct.pack(
        '<4sIHHIIHH',
        b'fmt ', 16, 1, num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample
    )
    h += struct.pack('<4sI', b'data', data_size)
    return h

# ── Core2 init ────────────────────────────────────────────────────────────────
M5.begin()
Widgets.fillScreen(0x000000)

# Speaker and Mic share I2S, so disable speaker first
Speaker.begin()
Speaker.setVolumePercentage(0)
Speaker.end()

# ── Display helpers ───────────────────────────────────────────────────────────
def show(line1, line2="", line3="", color=0xFFFFFF, bg=0x000000):
    Widgets.fillScreen(bg)
    if line1:
        Widgets.Label(line1, 20,  30, 1.0, color,    bg, Widgets.FONTS.DejaVu24)
    if line2:
        Widgets.Label(line2, 20,  95, 1.0, 0xFFFFFF, bg, Widgets.FONTS.DejaVu18)
    if line3:
        Widgets.Label(line3, 20, 170, 1.0, 0xAAAAAA, bg, Widgets.FONTS.DejaVu18)
    M5.update()

def update_line3(text, bg=0xFF0000):
    Widgets.Label("  " + text + "  ", 20, 170, 1.0, 0xFFFFFF, bg, Widgets.FONTS.DejaVu18)
    M5.update()

# ── Audio check helper ────────────────────────────────────────────────────────
def peak_amplitude(buf, n_samples=400):
    peak = 0
    limit = min(n_samples * 2, len(buf) - 1)
    for i in range(0, limit, 2):
        v = struct.unpack_from('<h', buf, i)[0]
        if v < 0:
            v = -v
        if v > peak:
            peak = v
    return peak

# ── Wait for BtnA press ───────────────────────────────────────────────────────
def wait_for_btnA():
    while True:
        M5.update()
        if BtnA.wasPressed():
            time.sleep_ms(250)
            return
        time.sleep_ms(50)

# ── Main recorder ─────────────────────────────────────────────────────────────
def record_and_save(filename, class_label, instruction, reminder, class_color):
    show("NEXT: " + class_label, instruction, "Press BtnA to start", class_color)
    wait_for_btnA()

    for i in range(3, 0, -1):
        show(class_label, "Starting in {}...".format(i), "", class_color)
        time.sleep(1)

    filepath = "/flash/" + filename

    with open(filepath, "wb") as f:
        f.write(make_wav_header(SAMPLE_RATE, BITS_PER_SAMPLE, NUM_CHANNELS, TOTAL_BYTES))

    gc.collect()
    buf = bytearray(CHUNK_BYTES)

    Mic.begin()
    time.sleep_ms(1000)

    # warm up mic and discard first chunk
    show("WARMING UP", class_label, "Please wait...", 0xAAAAAA)
    warm_buf = bytearray(CHUNK_BYTES)
    Mic.record(warm_buf, SAMPLE_RATE, False)
    while Mic.isRecording():
        M5.update()
        time.sleep_ms(100)

    gc.collect()

    t_total = time.ticks_ms()

    for chunk in range(NUM_CHUNKS):
        chunk_offset_sec = chunk * CHUNK_SEC

        show(
            "RECORDING",
            "{} ({}/{})".format(class_label, chunk + 1, NUM_CHUNKS),
            "{} {}s left".format(reminder, TOTAL_SEC - chunk_offset_sec),
            0xFF0000
        )

        t_chunk = time.ticks_ms()
        Mic.record(buf, SAMPLE_RATE, False)

        last_tick = -1
        while Mic.isRecording():
            M5.update()
            time.sleep_ms(200)
            elapsed = time.ticks_diff(time.ticks_ms(), t_chunk) // 1000
            if elapsed != last_tick:
                secs_left = max(0, TOTAL_SEC - chunk_offset_sec - elapsed)
                update_line3("{} {}s left".format(reminder, secs_left))
                last_tick = elapsed

        with open(filepath, "ab") as f:
            f.write(buf)

        gc.collect()

    Mic.end()

    actual_s = time.ticks_diff(time.ticks_ms(), t_total) / 1000.0
    total_file_bytes = 44 + TOTAL_BYTES

    show("Saved!", filename, "{} bytes ({:.1f}s)".format(total_file_bytes, actual_s), 0x00FF00)
    time.sleep(2)

    return total_file_bytes

# ── Intro ─────────────────────────────────────────────────────────────────────
show(
    "Project Recorder",
    "3 classes | 30s each",
    "silence / single / multiple",
    0xFFFFFF
)
time.sleep(3)

# ── Record all classes ────────────────────────────────────────────────────────
results = []

for label, fname, instr, reminder, color in CLASSES:
    nbytes = record_and_save(fname, label, instr, reminder, color)
    results.append((label, fname, nbytes))
    time.sleep(1)

# ── Final summary ─────────────────────────────────────────────────────────────
Widgets.fillScreen(0x000000)
Widgets.Label("ALL DONE!", 20, 10, 1.0, 0x00FF00, 0x000000, Widgets.FONTS.DejaVu24)

y_pos = 55
for label, fname, nbytes in results:
    Widgets.Label(
        "{}: {}B".format(fname, nbytes),
        20, y_pos, 1.0,
        0xFFFFFF, 0x000000, Widgets.FONTS.DejaVu18
    )
    y_pos += 32

Widgets.Label(
    "Files saved in /flash/",
    20, y_pos + 10, 1.0,
    0xFFFF00, 0x000000, Widgets.FONTS.DejaVu18
)
Widgets.Label(
    "Download via UIFlow File Manager",
    20, y_pos + 42, 1.0,
    0xFFFF00, 0x000000, Widgets.FONTS.DejaVu18
)

while True:
    M5.update()
    time.sleep(0.1)