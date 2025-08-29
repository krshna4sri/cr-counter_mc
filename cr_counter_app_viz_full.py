# cr_counter_app_viz_full.py — CR-Counter with full-page background + styles + species + multi-instrument + MIDI + audio previews
# Run:  python -m streamlit run cr_counter_app_viz_full.py
from __future__ import annotations
import streamlit as st
import xml.etree.ElementTree as ET
from PIL import Image
import base64, io, os, random, math, struct, wave
from typing import List, Tuple

# NEW: midi export
try:
    from midiutil import MIDIFile
    _MIDI_AVAILABLE = True
except Exception:
    _MIDI_AVAILABLE = False

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
        /* 1) full page background */
        html, body, [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{b64}") center center fixed no-repeat !important;
            background-size: cover !important;
        }}
        /* 2) transparent header */
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        /* 3) readable 'glass' panel for content */
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
        "header.jpg.jpg",               # your uploaded filename case
        "cover.jpg",
        "cover.png",
        "VS_Pop_Funky_1400x1400.jpg",   # your shared image
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

# General MIDI program numbers (0-based)
GM_PROGRAM = {
    "Flute": 73,          # 74 in GM (1-based)
    "Oboe": 68,           # 69
    "Clarinet": 71,       # 72
    "Reed Section": 65,   # Alto Sax (proxy)
    "Violin": 40,         # 41
    "Viola": 41,          # 42
    "Cello": 42,          # 43
    "Double Bass": 43,    # 44 (Contrabass)
    "Piano Pad": 89,      # Pad 2 (Warm) proxy
    "Strings Pad": 50,    # SynthStrings 1
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

def generate_lead_eighths(key, mode, bars, register=(60,84)):
    scale = scale_midis(key, mode, low=register[0], high=register[1])
    tonic_pc = MAJOR_PC[key]; start_target = 60 + tonic_pc
    line = [choose_nearest_scale(start_target, scale)]
    weights = dict(BASE_W); prev_iv = 0
    total_eighths = bars*8
    for _ in range(1,total_eighths):
        iv = sample_interval(prev_iv, weights)
        cand = line[-1] + iv
        nearest = choose_nearest_scale(cand, scale)
        line.append(nearest); prev_iv = iv
    tonic_candidates = [m for m in scale if m%12==tonic_pc]
    line[-1] = min(tonic_candidates, key=lambda m: abs(m - line[-1]))
    return line

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
        divisions = 2  # eighth note = 1 unit
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
def _dur8_to_beats(dur8: int) -> float:
    # 1 eighth = 0.5 beats, 2 = 1 beat, 4 = 2 beats, 8 = 4 beats
    return dur8 / 2.0

def parts_to_midi(parts, tempo_bpm: int) -> bytes:
    """
    Build a Type-1 multi-track MIDI. One track per part, channels 0..15 (skip 9).
    """
    if not _MIDI_AVAILABLE:
        raise RuntimeError("midiutil not installed")

    mf = MIDIFile(len(parts))  # one track per part
    beat_time = 0.0
    for ti, p in enumerate(parts):
        track = ti
        channel = ti % 16
        if channel == 9:  # avoid percussion channel
            channel = (channel + 1) % 16
        mf.addTrackName(track, 0, p["name"])
        mf.addTempo(track, 0, int(tempo_bpm))
        program = GM_PROGRAM.get(p["name"], 48)  # default String Ensemble 1
        mf.addProgramChange(track, channel, 0, program)

        time_beats = 0.0
        idx = 0
        for dur8 in p["rhythm"]:
            pitch = p["slots"][idx]
            beats = _dur8_to_beats(dur8)
            mf.addNote(track, channel, pitch, time_beats, beats, 92 if p["role"]!="pad" else 80)
            time_beats += beats
            idx += dur8

    out = io.BytesIO()
    mf.writeFile(out)
    return out.getvalue()

# -------------------- Tiny audio previews (WAV) --------------------
def _midi_to_freq(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def _osc(value_t: float, freq: float, sr: int, wave_type: str) -> float:
    # simple oscillators
    t = value_t
    x = 2.0 * math.pi * freq * t
    if wave_type == "sine":
        return math.sin(x)
    elif wave_type == "triangle":
        # quick triangle approximate
        return 2.0 / math.pi * math.asin(math.sin(x))
    elif wave_type == "saw":
        # basic sawtooth approx
        return 2.0 * ((freq * t) - math.floor(0.5 + freq * t))
    else:
        return math.sin(x)

def _adsr(n: int, sr: int, dur_s: float, a=0.01, d=0.05, s=0.7, r=0.05) -> List[float]:
    total = n
    aN = int(a*sr); dN = int(d*sr); rN = int(r*sr)
    sN = max(0, total - aN - dN - rN)
    env = []
    # attack
    for i in range(max(aN,1)):
        env.append(i/max(aN,1))
    # decay
    for i in range(max(dN,1)):
        env.append(1 - (1-s)*(i/max(dN,1)))
    # sustain
    for _ in range(sN):
        env.append(s)
    # release
    last = env[-1] if env else s
    for i in range(max(rN,1)):
        env.append(last * (1 - i/max(rN,1)))
    # trim/pad
    if len(env) > total: env = env[:total]
    elif len(env) < total: env += [0.0]*(total-len(env))
    return env

def _instrument_waveform(name: str) -> str:
    # cheap timbre mapping
    if name in ("Flute","Oboe","Clarinet","Reed Section"):
        return "sine"
    if name in ("Violin","Viola","Strings Pad"):
        return "saw"
    if name in ("Cello","Double Bass"):
        return "triangle"
    if name in ("Piano Pad",):
        return "sine"
    return "sine"

def synth_preview_wav(part: dict, tempo_bpm: int, preview_bars=4, sr=44100) -> bytes:
    """
    Render a short monophonic WAV of 'preview_bars' for this part.
    Standard library only (wave/math); small and fast for demos.
    """
    seconds_per_beat = 60.0 / float(tempo_bpm)
    seconds_per_eighth = seconds_per_beat / 2.0

    # Gather events for the first N bars
    total_tokens = preview_bars * 8
    idx = 0
    events = []
    for dur8 in part["rhythm"]:
        if total_tokens <= 0:
            break
        use = min(dur8, total_tokens)
        pitch = part["slots"][idx]
        events.append((pitch, use))
        idx += dur8
        total_tokens -= use

    waveform = _instrument_waveform(part["name"])
    buf = io.BytesIO()
    wf = wave.open(buf, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)   # 16-bit
    wf.setframerate(sr)

    # render
    for pitch, dur8 in events:
        freq = _midi_to_freq(pitch)
        dur_s = seconds_per_eighth * dur8
        n = int(sr * dur_s)
        env = _adsr(n, sr, dur_s, a=0.01, d=0.03, s=0.75, r=0.06)
        for i in range(n):
            t = i / sr
            sample = 0.18 * _osc(t, freq, sr, waveform) * env[i]   # keep headroom
            wf.writeframes(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))
    wf.close()
    return buf.getvalue()

# -------------------- UI --------------------
st.set_page_config(page_title=APP_NAME, page_icon="🎼", layout="centered")

# Background from local file
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

    # Lead
    lead_inst = next(i for i in insts if i["role"]=="lead")
    lead_slots = generate_lead_eighths(key, mode, bars_eff, register=lead_inst["range"])

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
    st.success("Generated! Download your score:")
    st.download_button("Download MusicXML",
                       data=xml_bytes,
                       file_name="CR_Counter_Arrangement.xml",
                       mime="application/vnd.recordare.musicxml+xml")

    # NEW: MIDI download
    if _MIDI_AVAILABLE:
        midi_bytes = parts_to_midi(parts, int(tempo))
        st.download_button("Download MIDI",
                           data=midi_bytes,
                           file_name="CR_Counter_Arrangement.mid",
                           mime="audio/midi")
    else:
        st.info("Install `midiutil` to enable MIDI export:  `pip install midiutil`")

    flats = prefers_flats(key)
    st.markdown("**Preview (first 16 eighths per part)**")
    for p in parts:
        names = ", ".join(midi_to_name(m, flats) for m in p["slots"][:16])
        st.write(f"{p['name']} ({p['role']}): {names}")

    # NEW: inline audio previews (first 4 bars per part)
    with st.expander("Audio previews (first 4 bars, simple synth)"):
        for p in parts:
            wav = synth_preview_wav(p, int(tempo), preview_bars=4)
            st.write(p["name"])
            st.audio(wav, format="audio/wav")
            st.download_button(f"Download {p['name']} preview (WAV)",
                               data=wav,
                               file_name=f"{p['name'].replace(' ','_')}_preview.wav",
                               mime="audio/wav")
