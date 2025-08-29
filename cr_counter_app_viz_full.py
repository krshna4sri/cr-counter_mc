# cr_counter_app_viz_full.py — CR-Counter with full-page background + styles + species + multi-instrument + Raga modes
# Fixes in this version:
#  - Lead always starts on the actual tonic of the chosen Key/Mode/Raga (closest to middle register), not C
#  - Works for Western modes and all provided Ragas
#  - Hides/disables Mode whenever a Carnatic Raga is selected (engine already ignores Mode in that case)
# Run:  python -m streamlit run cr_counter_app_viz_full.py
from __future__ import annotations

import base64
import io
import math
import os
import random
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image

APP_NAME = "CR-Counter"

# -------------------- Optional deps --------------------
try:
    import music21 as m21
    _HAS_M21 = True
except Exception:
    _HAS_M21 = False

try:
    import mido
    _HAS_MIDO = True
except Exception:
    _HAS_MIDO = False

try:
    import numpy as _np
    _HAS_NP = True
except Exception:
    _HAS_NP = False


# -------------------- Visual helpers --------------------
def _encode_img(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def set_page_background(img: Image.Image):
    """Use the image as a TRUE full-page background."""
    b64 = _encode_img(img)
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{b64}") center center fixed no-repeat !important;
            background-size: cover !important;
        }}
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        .main .block-container {{
            background: rgba(255,255,255,0.86) !important;
            border-radius: 16px;
            padding: 12px 18px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.06);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_header_image() -> Image.Image | None:
    for c in [
        "header.jpg",
        "header.png",
        "header.jpg.jpg",
        "cover.jpg",
        "cover.png",
        "VS_Pop_Funky_1400x1400.jpg",
    ]:
        try:
            return Image.open(c).convert("RGB")
        except Exception:
            continue
    return None


def hero(title: str):
    st.markdown(
        f"<h1 style='text-align:center;margin:0.35rem 0 1.0rem 0'>{title}</h1>",
        unsafe_allow_html=True,
    )


# -------------------- Theory helpers --------------------
KEYS_CANON = [
    "C","G","D","A","E","B","F#","C#",
    "F","Bb","Eb","Ab","Db","Gb","Cb"
]

MAJOR_PC = {"C":0,"G":7,"D":2,"A":9,"E":4,"B":11,"F#":6,"C#":1,"F":5,"Bb":10,"Eb":3,"Ab":8,"Db":1,"Gb":6,"Cb":11}
FIFTHS   = {"C":0,"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7,"F":-1,"Bb":-2,"Eb":-3,"Ab":-4,"Db":-5,"Gb":-6,"Cb":-7}
STEP_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
STEP_FLAT  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
DIATONIC_MAJOR = [0,2,4,5,7,9,11]
DIATONIC_NAT_MINOR = [0,2,3,5,7,8,10]
DIATONIC_HARM_MINOR = [0,2,3,5,7,8,11]
FLAT_KEYS = {"F","Bb","Eb","Ab","Db","Gb","Cb"}

def prefers_flats(k: str) -> bool: return k in FLAT_KEYS

def normalize_key_mode(key: str, mode: str) -> Tuple[str, str]:
    """Pass-through for canonical dropdowns: prevents surprise fallbacks."""
    k = key if key in MAJOR_PC else "C"
    m = mode if mode in ("major", "natural_minor", "harmonic_minor") else "major"
    return k, m

def midi_to_pitch(midi: int, flats=False):
    names = STEP_FLAT if flats else STEP_SHARP
    pc = midi % 12; name = names[pc]; step = name[0]; alter = 0
    if len(name)==2: alter = 1 if name[1] == "#" else -1
    octave = (midi//12) - 1
    return step, alter, octave

def midi_to_name(midi: int, flats=False):
    names = STEP_FLAT if flats else STEP_SHARP
    pc = midi % 12; name = names[pc]; return f"{name}{(midi//12)-1}"

# -------- Raga definitions (semitone degrees from tonic) --------
# Correct Hindolam: S G2 M1 D1 N3 S -> [0,3,5,8,11]
RAGA_DEGREES = {
    "Shankarabharanam (Major)":       [0,2,4,5,7,9,11],
    "Kalyani (Lydian #4)":            [0,2,4,6,7,9,11],
    "Harikambhoji (Mixolydian)":      [0,2,4,5,7,9,10],
    "Kharaharapriya (Dorian)":        [0,2,3,5,7,9,10],
    "Natabhairavi (Natural Minor)":   [0,2,3,5,7,8,10],
    "Keeravani (Harmonic Minor)":     [0,2,3,5,7,8,11],
    "Charukesi":                      [0,2,4,5,7,8,11],
    "Sarasangi":                      [0,1,4,5,7,9,11],
    "Mararanjani":                    [0,2,4,6,7,8,10],

    # Pentatonic
    "Mohanam (Major Pentatonic)":     [0,2,4,7,9],
    "Hamsadhwani":                    [0,2,4,7,11],
    "Hindolam":                       [0,3,5,8,11],   # fixed
    "Suddha Saveri":                  [0,2,5,7,9],
    "Abhogi":                         [0,2,3,5,7],
    "Sreeranjani":                    [0,2,3,6,9],
    "Madhyamavati":                   [0,2,5,7,10],
    "Megh (Megh Malhar)":             [0,2,5,7,9],

    # Hexatonic / others
    "Durga":                          [0,2,5,7,9,10],
    "Devakriya (Sudha Dhanyasi)":     [0,2,3,7,9],
    "Revati":                         [0,1,5,7,11],
    "Amritavarshini":                 [0,4,6,7,11],
    "Vachaspati (Lydian b7)":         [0,2,4,6,7,9,10],
    "Hemavati":                       [0,2,3,6,7,9,11],
    "Shubhapantuvarali":              [0,1,4,5,7,8,11],
    "Todi (Hanumatodi)":              [0,1,3,6,7,8,11],
}

RAGA_LIST = ["None (use Western mode)"] + list(RAGA_DEGREES.keys())

def raga_scale_midis(key: str, raga: Optional[str], low=36, high=96) -> List[int]:
    tonic = MAJOR_PC[key]
    if not raga or raga == "None (use Western mode)":
        return list(range(low, high+1))  # neutral; caller filters with Western mode
    degrees = set(RAGA_DEGREES.get(raga, DIATONIC_MAJOR))
    allowed_pc = set((tonic + d) % 12 for d in degrees)
    return [m for m in range(low, high+1) if (m % 12) in allowed_pc]

def western_scale_midis(key: str, mode: str, low=36, high=96) -> List[int]:
    tonic = MAJOR_PC[key]
    if mode=="major": allowed = set((tonic + d) % 12 for d in DIATONIC_MAJOR)
    elif mode=="natural_minor": allowed = set((tonic + d) % 12 for d in DIATONIC_NAT_MINOR)
    else: allowed = set((tonic + d) % 12 for d in DIATONIC_HARM_MINOR)
    return [m for m in range(low, high+1) if (m % 12) in allowed]

def scale_midis(key: str, mode: str, raga: Optional[str], low=36, high=96) -> List[int]:
    if raga and raga != "None (use Western mode)":
        return raga_scale_midis(key, raga, low, high)
    return western_scale_midis(key, mode, low, high)

def consonant_with(a:int, b:int) -> bool:
    d = abs((a-b)%12); return d in (0,3,4,5,7,8,9)

def step_options(p:int, scale:List[int]): return [x for x in (p-2,p-1,p+1,p+2) if x in scale]

def tonic_near_middle(key: str, scale: List[int], middle: int = 60) -> int:
    """Find the tonic (by pitch class of key) *inside the given scale* closest to `middle` (C4 by default)."""
    tonic_pc = MAJOR_PC[key]
    candidates = [m for m in scale if (m % 12) == tonic_pc]
    if not candidates:  # should never happen, but be safe
        return min(scale, key=lambda m: abs(m - middle))
    return min(candidates, key=lambda m: abs(m - middle))


# -------------------- Styles & instruments --------------------
STYLE_PATTERNS = {
    "Pop":      {"lead":[[2,2,2,2]], "counter":[[1,1,2,1,1,2]], "bass":[[2,2,2,2]], "pad":[[8]]},
    "Rock":     {"lead":[[2,2,1,1,2]], "counter":[[1,1,1,1,2,2]], "bass":[[2,2,2,2]], "pad":[[4,4]]},
    "R&B":      {"lead":[[1,1,2,1,1,2]], "counter":[[1,1,1,1,1,1,2]], "bass":[[2,1,1,2,2]], "pad":[[8]]},
    "Dance":    {"lead":[[1,1,1,1,2,2]], "counter":[[1,1,1,1,1,1,2]], "bass":[[1,1,1,1,1,1,1,1]], "pad":[[4,4]]},
    "Country":  {"lead":[[2,2,2,2]], "counter":[[1,1,2,1,1,2]], "bass":[[2,2,2,2]], "pad":[[8]]},
    "Folk":     {"lead":[[2,2,2,2]], "counter":[[2,1,1,2]], "bass":[[2,2,2,2]], "pad":[[4,4]]},
    "World":    {"lead":[[1,1,2,1,1,2]], "counter":[[2,1,1,2]], "bass":[[2,2,1,1]], "pad":[[8]]},
    "Classical":{"lead":[[2,2,2,2]], "counter":[[1,1,2,1,1,2]], "bass":[[2,2,2,2]], "pad":[[8]]},
    "Jazz":     {"lead":[[1,1,1,1,2,2]], "counter":[[1,1,1,1,1,1,2]], "bass":[[2,2,2,2]], "pad":[[8]]},
    "Latin":    {"lead":[[1,1,2,1,1,2]], "counter":[[1,1,1,1,1,1,2]], "bass":[[2,2,1,1]], "pad":[[4,4]]},
}
INSTRUMENTS = {
    "Flute":         {"role":"lead",    "range":(72,96)},
    "Oboe":          {"role":"lead",    "range":(67,91)},
    "Clarinet":      {"role":"counter", "range":(60,88)},
    "Reed Section":  {"role":"counter", "range":(60,92)},
    "Violin":        {"role":"lead",    "range":(67,96)},
    "Viola":         {"role":"counter", "range":(55,79)},
    "Cello":         {"role":"bass",    "range":(48,67)},
    "Double Bass":   {"role":"bass",    "range":(40,64)},
    "Piano Pad":     {"role":"pad",     "range":(48,84)},
    "Strings Pad":   {"role":"pad",     "range":(55,91)},
}
def gen_rhythm(patterns, bars): return [d for b in range(bars) for d in patterns[b % len(patterns)]]


# -------------------- Lead generator --------------------
ALLOWED_INTERVALS = [-7,-5,-4,-3,-2,-1,1,2,3,4,5,7]
BASE_W = {i: (6-abs(i)) if abs(i)<=4 else (2 if abs(i) in (5,7) else 1) for i in ALLOWED_INTERVALS}

def sample_interval(prev_iv, weights):
    w = dict(weights)
    if abs(prev_iv)>=4:
        for i in (1,2):
            w[-i if prev_iv>0 else i] = w.get(-i if prev_iv>0 else i,1)*1.7
    tot = sum(w.values()); r = random.random()*tot; c=0.0
    for iv, wt in w.items():
        c += wt
        if r <= c: return iv
    import random as _r
    return _r.choice(ALLOWED_INTERVALS)

def choose_nearest_scale(target, scale): return min(scale, key=lambda m: abs(m-target))

def generate_lead_eighths(key, mode, raga, bars, register=(60,84)):
    scale = scale_midis(key, mode, raga, low=register[0], high=register[1])
    # START ON THE TONIC (closest to middle)
    start_note = tonic_near_middle(key, scale, middle=60)
    line = [start_note]
    weights = dict(BASE_W); prev_iv = 0
    total_eighths = bars*8
    for _ in range(1,total_eighths):
        iv = sample_interval(prev_iv, weights)
        cand = line[-1] + iv
        nearest = choose_nearest_scale(cand, scale)
        line.append(nearest); prev_iv = iv
    # Smooth landing on tonic
    line[-1] = tonic_near_middle(key, scale, middle=line[-1])
    return line


# -------------------- Counterpoint (species) --------------------
SPECIES_RHYTHM = {"1":[8], "2":[4,4], "3":[2,2,2,2], "4":[6,2], "5":[1,1,2,1,1,2], "Classical":[1,1,2,1,1,2]}

def generate_counter_species(cantus_slots, key, mode, raga, bars, species, register=(55,84)):
    scale = scale_midis(key, mode, raga, low=register[0], high=register[1])
    total = bars*8; out=[]; i=0
    while i < total:
        c = cantus_slots[i]
        if species == "1":
            cand = min(scale, key=lambda n: (0 if consonant_with(n,c) else 1, abs(n-c))); out += [cand]*8; i += 8
        elif species == "2":
            first = min(scale, key=lambda n: (0 if consonant_with(n,c) else 1, abs(n-c)))
            second = min(step_options(first, scale) or [first], key=lambda n: abs(n-first))
            out += [first]*4 + [second]*4; i += 8
        elif species == "3":
            b1 = min(scale, key=lambda n: (0 if consonant_with(n,c) else 1, abs(n-c)))
            s1 = min(step_options(b1, scale) or [b1], key=lambda n: abs(n-b1))
            b3 = min(scale, key=lambda n: (0 if consonant_with(n,c) else 1, abs(n-c)))
            s3 = min(step_options(b3, scale) or [b3], key=lambda n: abs(n-b3))
            out += [b1]*2 + [s1]*2 + [b3]*2 + [s3]*2; i += 8
        elif species == "4":
            held = min(scale, key=lambda n: (0 if consonant_with(n,c) else 1, abs(n-c)))
            res = [n for n in (held-1,held-2) if n in scale] or step_options(held, scale) or [held]
            out += [held]*6 + [res[0]]*2; i += 8
        else:  # "5" or "Classical"
            b1 = min(scale, key=lambda n: (0 if consonant_with(n,c) else 1, abs(n-c)))
            n1 = min(step_options(b1, scale) or [b1], key=lambda n: abs(n-b1))
            n2 = min(step_options(n1, scale) or [n1], key=lambda n: abs(n-n1))
            b3 = min(scale, key=lambda n: (0 if consonant_with(n,c) else 1, abs(n-c)))
            n3 = min(step_options(b3, scale) or [b3], key=lambda n: abs(n-b3))
            n4 = min(step_options(n3, scale) or [n3], key=lambda n: abs(n-n3))
            out += [b1, n1, n2, n2, b3, n3, n4, n4]; i += 8
    return out[:total]


# -------------------- MusicXML --------------------
def parts_to_musicxml(parts, key, mode, raga, tempo, bars):
    flats = prefers_flats(key)
    root = ET.Element("score-partwise", version="3.1")
    part_list = ET.SubElement(root, "part-list")
    for i,p in enumerate(parts, start=1):
        score_part = ET.SubElement(part_list, "score-part", id=f"P{i}")
        name = p["name"] + ("" if (not raga or raga=="None (use Western mode)") else f" ({raga})")
        ET.SubElement(score_part, "part-name").text = name
    for i,p in enumerate(parts, start=1):
        part = ET.SubElement(root, "part", id=f"P{i}")
        divisions = 2
        events = []; idx=0
        for dur in p["rhythm"]:
            pitch = p["slots"][idx]; events.append((pitch, dur)); idx += dur
        meas = ET.SubElement(part, "measure", number="1")
        attrs = ET.SubElement(meas, "attributes")
        ET.SubElement(attrs, "divisions").text = str(divisions)
        k = ET.SubElement(attrs, "key")
        ET.SubElement(k, "fifths").text = str(FIFTHS[key])
        ET.SubElement(k, "mode").text = "major" if mode=="major" else "minor"
        t = ET.SubElement(attrs, "time"); ET.SubElement(t, "beats").text="4"; ET.SubElement(t, "beat-type").text="4"
        clef = ET.SubElement(attrs, "clef"); ET.SubElement(clef, "sign").text="G"; ET.SubElement(clef, "line").text="2"
        d = ET.SubElement(meas, "direction", placement="above"); dt = ET.SubElement(d, "direction-type"); m = ET.SubElement(dt, "metronome")
        ET.SubElement(m, "beat-unit").text="quarter"; ET.SubElement(m, "per-minute").text=str(tempo); ET.SubElement(d, "sound", tempo=str(tempo))
        measure_num=1; used=0
        for pitch, dur8 in events:
            while used + dur8 > 8:
                split = 8 - used; _emit_note(meas, pitch, split, divisions, flats)
                measure_num += 1; meas = ET.SubElement(part, "measure", number=str(measure_num)); used=0; dur8 -= split
            _emit_note(meas, pitch, dur8, divisions, flats); used += dur8
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, method="xml")

def _emit_note(meas, midi, dur8, divisions, flats):
    duration = int(dur8 * (divisions/1))
    note = ET.SubElement(meas, "note")
    pitch = ET.SubElement(note, "pitch")
    step, alter, octave = midi_to_pitch(midi, flats=flats)
    ET.SubElement(pitch, "step").text = step
    if alter!=0: ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(octave)
    ET.SubElement(note, "duration").text = str(duration); ET.SubElement(note, "voice").text = "1"
    ET.SubElement(note, "type").text = {1:"eighth",2:"quarter",4:"half",8:"whole"}.get(dur8, "eighth")


# -------------------- MIDI export --------------------
PROGRAMS = {"lead":73, "counter":71, "bass":32, "pad":88}  # rough GM programs

def parts_to_midi_bytes(parts, tempo_bpm):
    if not _HAS_MIDO:
        raise RuntimeError("mido not installed")
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    us_per_beat = int(60_000_000 / max(1, int(tempo_bpm)))
    track.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))

    events = []  # (abs_tick, kind, a, b, role)
    for p in parts:
        role = p["role"]
        program = PROGRAMS.get(role, 73)
        events.append((0, "program_change", program, None, role))
        abs_tick = 0
        idx = 0
        for dur8 in p["rhythm"]:
            pitch = p["slots"][idx]
            dur_q = dur8 / 2.0
            dur_ticks = int(dur_q * 480)
            events.append((abs_tick, "on", pitch, 72, role))
            events.append((abs_tick + dur_ticks, "off", pitch, 64, role))
            abs_tick += dur_ticks
            idx += dur8

    def _sort_key(e):
        t, kind, *_ = e
        pri = 0 if kind == "program_change" else (1 if kind == "off" else 2)
        return (t, pri)
    events.sort(key=_sort_key)

    current = 0
    current_program = None
    for t, kind, a, b, _role in events:
        delta = t - current
        current = t
        if kind == "program_change":
            if a != current_program:
                track.append(mido.Message('program_change', program=int(a), time=delta))
                current_program = a
            else:
                track.append(mido.Message('program_change', program=int(a), time=delta))
        elif kind == "on":
            track.append(mido.Message('note_on', note=int(a), velocity=int(b), time=delta))
        elif kind == "off":
            track.append(mido.Message('note_off', note=int(a), velocity=int(b), time=delta))

    bio = io.BytesIO()
    mid.save(file=bio)
    return bio.getvalue()


# -------------------- Audio preview (first N bars) --------------------
_INSTR_GAIN = {"lead":0.9,"counter":0.7,"bass":0.7,"pad":0.5}
def _midi_to_hz(n): return 440.0 * (2.0 ** ((n - 69) / 12.0))

def parts_to_wav_preview(parts, tempo_bpm, bars=4, sr=22050):
    seconds_per_beat = 60.0 / max(1, tempo_bpm)
    beats_total = 4 * bars
    seconds_total = beats_total * seconds_per_beat

    n_samples = int(seconds_total * sr)
    if _HAS_NP:
        mix = _np.zeros(n_samples, dtype=_np.float32)
    else:
        mix = [0.0] * n_samples

    for p in parts:
        role = p["role"]; gain = _INSTR_GAIN.get(role, 0.6)
        idx = 0; t_cursor = 0.0
        for dur8 in p["rhythm"]:
            if t_cursor >= seconds_total: break
            pitch = p["slots"][idx]
            dur_q = dur8 / 2.0
            dur_sec = dur_q * seconds_per_beat
            f = _midi_to_hz(pitch)
            start = int(t_cursor * sr)
            end = min(n_samples, int((t_cursor + dur_sec) * sr))
            length = max(1, end - start)
            for i in range(length):
                s = math.sin(2*math.pi*f*(i/sr))
                a = min(1.0, i/(0.01*sr))
                r = min(1.0, (length-1-i)/(0.01*sr))
                env = min(a, r)
                val = gain * 0.3 * s * env
                if _HAS_NP:
                    mix[start+i] += val
                else:
                    mix[start+i] += val
            t_cursor += dur_sec
            idx += dur8

    if _HAS_NP:
        mx = max(1e-6, float(_np.max(_np.abs(mix))))
        mix = (mix / mx * 0.95)
        pcm16 = (mix * 32767.0).astype(_np.int16).tobytes()
    else:
        mx = max(1e-6, max(abs(x) for x in mix))
        out = io.BytesIO()
        for x in mix:
            v = int(32767 * 0.95 * (x / mx))
            out.write(struct.pack("<h", max(-32768, min(32767, v))))
        pcm16 = out.getvalue()

    buf = io.BytesIO()
    import wave
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm16)
    return buf.getvalue()


# -------------------- Score upload (fingerprint) --------------------
@dataclass
class ScoreFingerprint:
    contour: List[int]         # sequence of melodic intervals in scale degrees-ish
    rhythm_quavers: List[int]  # kept for future
    tonic_pc: int
    mode_guess: str

def _safe_parse_bytes_with_format(file_bytes: bytes, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".mid", ".midi"):
        return m21.converter.parseData(file_bytes, format="midi")
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = file_bytes
    return m21.converter.parseData(text, format="musicxml")

def analyze_upload(file_bytes: bytes, filename: str) -> Optional[ScoreFingerprint]:
    if not _HAS_M21:
        return None
    try:
        s = _safe_parse_bytes_with_format(file_bytes, filename)
    except Exception as e:
        st.error(f"Unable to parse uploaded score (in-memory): {e}")
        return None

    try:
        k = s.analyze("key")
        mode_guess = "major" if k.mode.lower().startswith("major") else "harmonic_minor"
        tonic_pc = MAJOR_PC.get(k.tonic.name.replace("-", "b"), 0)
    except Exception:
        mode_guess = "major"; tonic_pc = 0

    melody = s.parts[0].flat.notes if s.parts else s.flat.notes
    midi = [n.pitch.midi for n in melody if hasattr(n, "pitch")]
    if len(midi) < 4:
        st.warning("Uploaded score has too few melodic notes to fingerprint.")
        return None

    contour = []
    for i in range(1, len(midi)):
        d = midi[i] - midi[i-1]
        if d <= -3: contour.append(-2)
        elif d >= 3: contour.append(+2)
        elif d < 0: contour.append(-1)
        elif d > 0: contour.append(+1)
        else: contour.append(0)

    try:
        ql = s.flat.quarterLength
        n_quavers = int(round(ql * 2))
        quavers = [1] * max(1, n_quavers)
    except Exception:
        quavers = [1] * 32

    return ScoreFingerprint(contour=contour, rhythm_quavers=quavers, tonic_pc=tonic_pc, mode_guess=mode_guess)


def contour_to_slots(fp_contour: List[int], key: str, mode: str, raga: Optional[str], bars: int, register=(60,84)):
    """
    Build an eighth-note resolution melody driven by a contour, but DO NOT set the rhythm here.
    Rhythm is chosen by style (Pop/Latin/...).
    """
    scale = scale_midis(key, mode, raga, low=register[0], high=register[1])
    # START ON THE TONIC (closest to middle)
    start = tonic_near_middle(key, scale, middle=60)
    total_quavers = bars * 8

    out = [start]
    ci = 0
    while len(out) < total_quavers:
        step = 0
        if ci < len(fp_contour):
            c = fp_contour[ci]
            step = -2 if c <= -2 else (2 if c >= 2 else c)
        candidates = step_options(out[-1], scale) or [out[-1]]
        target = out[-1] + step
        nxt = min(candidates, key=lambda m: abs(m - target))
        out.append(nxt)
        ci += 1

    # land near tonic
    out[-1] = tonic_near_middle(key, scale, middle=out[-1])
    return out[:total_quavers]


# -------------------- UI --------------------
st.set_page_config(page_title=APP_NAME, page_icon="🎼", layout="centered")

base_img = load_header_image()
if base_img is not None:
    set_page_background(base_img)
hero(APP_NAME)

with st.sidebar:
    st.markdown("### Branding / Images")
    up_img = st.file_uploader("Upload a header/background (jpg/png)", type=["jpg","jpeg","png"])
    if up_img is not None:
        hdr = Image.open(up_img).convert("RGB")
        st.image(hdr, caption="Uploaded", use_container_width=True)
        set_page_background(hdr)
    st.caption("Tip: keep **header.jpg** next to the script for persistent branding.")

col1, col2 = st.columns(2)

# Key dropdown
key_input = col1.selectbox("Key", KEYS_CANON, index=KEYS_CANON.index("C"), key="ui_key")

# Carnatic Raga (optional)
raga_input = st.selectbox(
    "Carnatic Raga (optional)",
    ["None (use Western mode)"] + list(RAGA_DEGREES.keys()),
    index=0,
    help="Choose a raga to constrain pitch material. Rhythm still follows Arrangement Style."
)

# Mode: hide/disable when a raga is active (engine ignores it in that case anyway)
mode_slot = col2.empty()
MODES = ["major", "harmonic_minor", "natural_minor"]
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = "major"

if raga_input != "None (use Western mode)":
    mode_slot.markdown("**Mode**: _controlled by raga_")
    mode_input = st.session_state.ui_mode  # preserved but ignored
else:
    mode_input = mode_slot.selectbox("Mode", MODES, index=MODES.index(st.session_state.ui_mode), key="ui_mode")

style = st.selectbox("Arrangement Style", list(STYLE_PATTERNS.keys()), index=2)
species = st.selectbox("Counterpoint Species (counter part)", ["1","2","3","4","5","Classical"], index=2)

bars = st.slider("Bars (4/4 time)", 4, 128, 32, 1)
target_tokens = st.number_input("Optional: target length in eighth-notes (overrides Bars)", min_value=0, value=0, step=8)
bars_eff = int(math.ceil(target_tokens/8)) if target_tokens>0 else int(bars)
st.caption(f"Effective bars: {bars_eff} (total eighth-note tokens ≈ {bars_eff*8})")

tempo = st.slider("Tempo (BPM)", 60, 160, 96, 2)
seed = st.number_input("Random seed (optional)", value=0, step=1)

available = list(INSTRUMENTS.keys())
chosen = st.multiselect("Pick instruments (first lead carries melody)", available,
                        default=["Flute","Reed Section","Double Bass","Strings Pad","Piano Pad"])

st.markdown("#### Optional: upload a reference score (MusicXML or MIDI)")
upload = st.file_uploader(
    "We’ll learn the melodic contour; rhythm always follows your Arrangement Style.",
    type=["xml","musicxml","mxl","mid","midi"]
)

fp: Optional[ScoreFingerprint] = None
if upload is not None and _HAS_M21:
    try:
        fp = analyze_upload(upload.read(), upload.name)
        if fp:
            st.success("Score parsed. We'll guide the lead line using its contour (style keeps the rhythm).")
    except Exception as e:
        st.error(f"Unable to parse uploaded score: {e}")

if st.button("Generate Score"):
    # If a raga is active, Western mode is irrelevant for pitch selection — but we still
    # pass a neutral 'major' to MusicXML headers to avoid confusing changes.
    effective_mode = mode_input if raga_input == "None (use Western mode)" else "major"
    key, mode = normalize_key_mode(key_input, effective_mode)
    raga = raga_input

    if seed: random.seed(int(seed))
    patt = STYLE_PATTERNS[style]
    rhythms = {role: gen_rhythm(patt[role], bars_eff) for role in patt}
    insts = [{"name":n, **INSTRUMENTS[n]} for n in chosen if n in INSTRUMENTS]
    if not insts:
        st.error("Pick at least one instrument.")
        st.stop()
    if not any(i["role"]=="lead" for i in insts):
        insts[0]["role"]="lead"

    # Lead (contour-guided if fingerprint present) — rhythm from STYLE
    lead_inst = next(i for i in insts if i["role"]=="lead")
    if fp is not None:
        lead_slots = contour_to_slots(fp.contour, key, mode, raga, bars_eff, register=lead_inst["range"])
    else:
        lead_slots = generate_lead_eighths(key, mode, raga, bars_eff, register=lead_inst["range"])

    parts = []
    for inst in insts:
        role = inst["role"]
        if role == "lead":
            rhythm = rhythms["lead"]
            slots = lead_slots[:bars_eff*8]
        elif role == "counter":
            rhythm = rhythms["counter"]
            slots = generate_counter_species(lead_slots, key, mode, raga, bars_eff, species, register=inst["range"])
        elif role == "bass":
            rhythm = rhythms["bass"]
            base_scale = scale_midis(key, mode, raga, low=inst["range"][0], high=inst["range"][1])
            slots = []
            for i,p in enumerate(lead_slots[:bars_eff*8]):
                target = p-12 if (i%8) in (0,4) else p-7
                slots.append(min(base_scale, key=lambda m: abs(m-target)))
        else:  # pad
            rhythm = rhythms["pad"]
            base_scale = scale_midis(key, mode, raga, low=inst["range"][0], high=inst["range"][1])
            slots = []
            for i,p in enumerate(lead_slots[:bars_eff*8]):
                target = p if (i%8) in (0,4) else (slots[-1] if slots else p)
                slots.append(min(base_scale, key=lambda m: abs(m-target)))
        parts.append({"name": inst["name"], "role": role, "rhythm": rhythm, "slots": slots})

    # ---- Downloads & preview ----
    xml_bytes = parts_to_musicxml(parts, key, mode, raga, int(tempo), bars_eff)
    st.success("Generated! Download your score:")

    st.download_button(
        "Download MusicXML",
        data=xml_bytes,
        file_name="CR_Counter_Arrangement.xml",
        mime="application/vnd.recordare.musicxml+xml"
    )

    if _HAS_MIDO:
        try:
            midi_bytes = parts_to_midi_bytes(parts, int(tempo))
            st.download_button(
                "Download MIDI",
                data=midi_bytes,
                file_name="CR_Counter_Arrangement.mid",
                mime="audio/midi"
            )
        except Exception as e:
            st.warning(f"MIDI export failed: {e} (install/update `mido`)")

    with st.expander("Audio previews (first 4 bars, simple synth)"):
        try:
            wav_bytes = parts_to_wav_preview(parts, int(tempo), bars=4, sr=22050)
            st.audio(wav_bytes, format="audio/wav")
        except Exception as e:
            st.warning(f"Audio preview failed: {e}")

    flats = prefers_flats(key)
    st.markdown("**Preview (first 16 eighths per part)**")
    for p in parts:
        names = ", ".join(midi_to_name(m, flats) for m in p["slots"][:16])
        st.write(f"{p['name']} ({p['role']}): {names}")
