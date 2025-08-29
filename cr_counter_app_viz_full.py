# cr_counter_app_viz_full.py — CR-Counter with full-page background + styles + species + multi-instrument + fugue-inspired subjects
# Run:  python -m streamlit run cr_counter_app_viz_full.py
from __future__ import annotations
import streamlit as st
import xml.etree.ElementTree as ET
from PIL import Image
import base64, io, os, random, math
from typing import List, Tuple

APP_NAME = "CR-Counter"

# -------------------- Visual helpers --------------------
def _encode_img(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def set_page_background(img: Image.Image):
    """Use the image as a TRUE full-page background (current Streamlit selectors)."""
    b64 = _encode_img(img)
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{b64}") center center fixed no-repeat !important;
            background-size: cover !important;
        }}
        [data-testid="stHeader"] {{ background: transparent !important; }}
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
MAJOR_PC = {"C":0,"G":7,"D":2,"A":9,"E":4,"B":11,"F#":6,"C#":1,"F":5,"Bb":10,"Eb":3,"Ab":8,"Db":1,"Gb":6,"Cb":11}
FIFTHS   = {"C":0,"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7,"F":-1,"Bb":-2,"Eb":-3,"Ab":-4,"Db":-5,"Gb":-6,"Cb":-7}
STEP_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
STEP_FLAT  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
DIATONIC_MAJOR = [0,2,4,5,7,9,11]
DIATONIC_NAT_MINOR = [0,2,3,5,7,8,10]
DIATONIC_HARM_MINOR = [0,2,3,5,7,8,11]
FLAT_KEYS = {"F","Bb","Eb","Ab","Db","Gb","Cb"}

def normalize_key_mode(key: str, mode: str) -> Tuple[str,str]:
    k = key.strip().replace(" ", "")
    if not k: k = "C"
    base = k[0].upper(); acc = k[1:] if len(k)>1 else ""
    if acc in ("#", "♯"): k = base + "#"
    elif acc in ("b","♭","B"): k = base + "b"
    else: k = base + acc
    m = (mode or "major").strip().lower()
    if m == "minor": m = "harmonic_minor"
    if m not in ("major","natural_minor","harmonic_minor"): m = "major"
    return k, m

def prefers_flats(k: str) -> bool: return k in FLAT_KEYS

def midi_to_pitch(midi: int, flats=False):
    names = STEP_FLAT if flats else STEP_SHARP
    pc = midi % 12; name = names[pc]; step = name[0]; alter = 0
    if len(name)==2: alter = 1 if name[1] == "#" else -1
    octave = (midi//12) - 1
    return step, alter, octave

def midi_to_name(midi: int, flats=False):
    names = STEP_FLAT if flats else STEP_SHARP
    pc = midi % 12; name = names[pc]; return f"{name}{(midi//12)-1}"

def scale_midis(key: str, mode: str, low=36, high=96) -> List[int]:
    tonic = MAJOR_PC[key]
    if mode=="major": allowed = set((tonic + d) % 12 for d in DIATONIC_MAJOR)
    elif mode=="natural_minor": allowed = set((tonic + d) % 12 for d in DIATONIC_NAT_MINOR)
    else: allowed = set((tonic + d) % 12 for d in DIATONIC_HARM_MINOR)
    return [m for m in range(low, high+1) if (m % 12) in allowed]

def consonant_with(a:int, b:int) -> bool:
    d = abs((a-b)%12); return d in (0,3,4,5,7,8,9)

def step_options(p:int, scale:List[int]): return [x for x in (p-2,p-1,p+1,p+2) if x in scale]

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

# -------------------- Lead generator (Markov) --------------------
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

# -------------------- Fugue references (small creative library) --------------------
# Interval seeds (in semitones relative to subject start), rhythm in eighths
# These are stylized patterns inspired by public-domain fugues; not literal transcriptions.
FUGUE_SEEDS = {
    # Bach WTC-style C minor: subject with a minor 2nd tension + stepwise ascent + leap of 4th
    ("C","harmonic_minor"): {
        "name": "C minor (Bach-style)",
        "intervals": [0, +2, +3, +5, +7, +5, +3, +2, 0, -2, -3, -5, -7, -5, -3, -2],  # 16 eighths
        "rhythm":    [1]*16
    },
    # A second option (slightly different contour)
    ("C","natural_minor"): {
        "name": "C natural minor (Bach-style)",
        "intervals": [0, +2, +3, +5, +7, +8, +7, +5, +3, +2, 0, -2, -3, -5, -7, -8],
        "rhythm":    [1]*16
    },
    # Add a couple more minors so the toggle feels useful
    ("D","harmonic_minor"): {
        "name": "D minor (Bach-style)",
        "intervals": [0, +2, +3, +5, +7, +5, +3, +2, 0, -2, -3, -5, -7, -5, -3, -2],
        "rhythm":    [1]*16
    },
    ("G","harmonic_minor"): {
        "name": "G minor (Bach-style)",
        "intervals": [0, +2, +3, +5, +7, +5, +3, +2, 0, -2, -3, -5, -7, -5, -3, -2],
        "rhythm":    [1]*16
    },
}

def intervals_to_midis(start_midi:int, intervals:List[int], scale:List[int]) -> List[int]:
    """Map an interval plan to closest diatonic scale midis, starting on start_midi."""
    out = []
    for off in intervals:
        target = start_midi + off
        out.append(choose_nearest_scale(target, scale))
    return out

def generate_lead_eighths(key, mode, bars, register=(60,84), subject:List[int]|None=None):
    """If subject is given (sequence of midis), use that for the opening, then continue Markov."""
    scale = scale_midis(key, mode, low=register[0], high=register[1])
    tonic_pc = MAJOR_PC[key]; start_target = 60 + tonic_pc
    start_note = choose_nearest_scale(start_target, scale)

    line: List[int] = []
    if subject:
        # place subject first (trim/expand to available bars*8 as needed)
        for m in subject:
            line.append(choose_nearest_scale(m, scale))
    else:
        line.append(start_note)

    weights = dict(BASE_W); prev_iv = 0
    total_eighths = bars*8

    while len(line) < total_eighths:
        base = line[-1] if line else start_note
        iv = sample_interval(prev_iv, weights)
        cand = base + iv
        nearest = choose_nearest_scale(cand, scale)
        line.append(nearest); prev_iv = iv

    # cadence on tonic
    tonic_candidates = [m for m in scale if m%12==tonic_pc]
    line[-1] = min(tonic_candidates, key=lambda m: abs(m - line[-1]))
    return line[:total_eighths]

# -------------------- Counterpoint (species) --------------------
SPECIES_RHYTHM = {"1":[8], "2":[4,4], "3":[2,2,2,2], "4":[6,2], "5":[1,1,2,1,1,2], "Classical":[1,1,2,1,1,2]}

def generate_counter_species(cantus_slots, key, mode, bars, species, register=(55,84)):
    scale = scale_midis(key, mode, low=register[0], high=register[1])
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
def parts_to_musicxml(parts, key, mode, tempo, bars):
    flats = prefers_flats(key)
    root = ET.Element("score-partwise", version="3.1")
    part_list = ET.SubElement(root, "part-list")
    for i,p in enumerate(parts, start=1):
        score_part = ET.SubElement(part_list, "score-part", id=f"P{i}")
        ET.SubElement(score_part, "part-name").text = p["name"]
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

# -------------------- UI --------------------
st.set_page_config(page_title=APP_NAME, page_icon="🎼", layout="centered")

base_img = load_header_image()
if base_img is not None:
    set_page_background(base_img)
hero(APP_NAME)

with st.sidebar:
    st.markdown("### Branding / Images")
    up = st.file_uploader("Upload a header/background (jpg/png)", type=["jpg","jpeg","png"])
    if up is not None:
        hdr = Image.open(up).convert("RGB")
        st.image(hdr, caption="Uploaded", use_container_width=True)
        set_page_background(hdr)
    st.caption("Tip: keep **header.jpg** next to the script for persistent branding.")

col1, col2 = st.columns(2)
key_input = col1.text_input("Key", "C")
mode_input = col2.selectbox("Mode", ["major","harmonic_minor","natural_minor"], index=0)
style = st.selectbox("Arrangement Style", list(STYLE_PATTERNS.keys()), index=2)
species = st.selectbox("Counterpoint Species (counter part)", ["1","2","3","4","5","Classical"], index=2)

# NEW: fugue reference toggle
use_fugue = st.checkbox("Use fugue reference (if available)", value=True)
bars = st.slider("Bars (4/4 time)", 4, 128, 32, 1)
target_tokens = st.number_input("Optional: target length in eighth-notes (overrides Bars)", min_value=0, value=0, step=8)
bars_eff = int(math.ceil(target_tokens/8)) if target_tokens>0 else int(bars)
st.caption(f"Effective bars: {bars_eff} (total eighth-note tokens ≈ {bars_eff*8})")

tempo = st.slider("Tempo (BPM)", 60, 160, 96, 2)
seed = st.number_input("Random seed (optional)", value=0, step=1)

available = list(INSTRUMENTS.keys())
chosen = st.multiselect("Pick instruments (first lead carries melody)", available,
                        default=["Flute","Reed Section","Double Bass","Strings Pad","Piano Pad"])

if st.button("Generate Score"):
    key, mode = normalize_key_mode(key_input, mode_input)
    if seed: random.seed(int(seed))
    patt = STYLE_PATTERNS[style]
    rhythms = {role: gen_rhythm(patt[role], bars_eff) for role in patt}
    insts = [{"name":n, **INSTRUMENTS[n]} for n in chosen if n in INSTRUMENTS]
    if not any(i["role"]=="lead" for i in insts) and insts:
        insts[0]["role"]="lead"

    # Fugue subject (if available for this key/mode)
    subject_midis = None
    if use_fugue and (key, mode) in FUGUE_SEEDS:
        info = FUGUE_SEEDS[(key, mode)]
        lead_inst = next(i for i in insts if i["role"]=="lead")
        scale = scale_midis(key, mode, low=lead_inst["range"][0], high=lead_inst["range"][1])
        # start note near middle of register on tonic
        tonic_pc = MAJOR_PC[key]
        middle = (lead_inst["range"][0] + lead_inst["range"][1]) // 2
        start_on_tonic = min([m for m in scale if m%12==tonic_pc], key=lambda m: abs(m-middle))
        subject_midis = intervals_to_midis(start_on_tonic, info["intervals"], scale)
        st.info(f"Fugue subject: {info['name']} (seeded {len(subject_midis)} eighths)")

    # Lead
    lead_inst = next(i for i in insts if i["role"]=="lead")
    lead_slots = generate_lead_eighths(key, mode, bars_eff, register=lead_inst["range"], subject=subject_midis)

    parts = []
    for inst in insts:
        role = inst["role"]; rhythm = rhythms[role]
        if role=="lead":
            slots = lead_slots[:bars_eff*8]
        elif role=="counter":
            slots = generate_counter_species(lead_slots, key, mode, bars_eff, species, register=inst["range"])
        elif role=="bass":
            base_scale = scale_midis(key, mode, low=inst["range"][0], high=inst["range"][1])
            slots = []
            for i,p in enumerate(lead_slots[:bars_eff*8]):
                target = p-12 if (i%8) in (0,4) else p-7
                slots.append(min(base_scale, key=lambda m: abs(m-target)))
        else:  # pad
            base_scale = scale_midis(key, mode, low=inst["range"][0], high=inst["range"][1])
            slots = []
            for i,p in enumerate(lead_slots[:bars_eff*8]):
                target = p if (i%8) in (0,4) else (slots[-1] if slots else p)
                slots.append(min(base_scale, key=lambda m: abs(m-target)))
        parts.append({"name": inst["name"], "role": role, "rhythm": rhythm, "slots": slots})

    xml_bytes = parts_to_musicxml(parts, key, mode, int(tempo), bars_eff)
    st.success("Generated! Download the full multi-part score:")
    st.download_button("Download MusicXML",
                       data=xml_bytes,
                       file_name="CR_Counter_Arrangement.xml",
                       mime="application/vnd.recordare.musicxml+xml")

    flats = prefers_flats(key)
    st.markdown("**Preview (first 16 eighths per part)**")
    for p in parts:
        names = ", ".join(midi_to_name(m, flats) for m in p["slots"][:16])
        st.write(f"{p['name']} ({p['role']}): {names}")
