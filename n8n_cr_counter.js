/**
 * n8n Function node: CR-Counter v2 (headless)
 * - Generates multi-part arrangement slots (Western or Raga), simplified counterpoint
 * - Exports MusicXML and MIDI as binary outputs
 * - No filesystem, no external libs, pure JS
 *
 * Output:
 *   returns one item:
 *   {
 *     json: { summary, params },
 *     binary: {
 *       musicxml: { data: <base64>, fileName: "CR_Counter_Arrangement.xml", mimeType: "application/vnd.recordare.musicxml+xml" },
 *       midi:     { data: <base64>, fileName: "CR_Counter_Arrangement.mid",  mimeType: "audio/midi" }
 *     }
 *   }
 */

// ---------- Helpers: seedable RNG (LCG) ----------
function lcg(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

// ---------- Theory / constants ----------
const KEYS_CANON = ["C","G","D","A","E","B","F#","C#","F","Bb","Eb","Ab","Db","Gb","Cb"];
const MAJOR_PC = {"C":0,"G":7,"D":2,"A":9,"E":4,"B":11,"F#":6,"C#":1,"F":5,"Bb":10,"Eb":3,"Ab":8,"Db":1,"Gb":6,"Cb":11};
const FIFTHS   = {"C":0,"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7,"F":-1,"Bb":-2,"Eb":-3,"Ab":-4,"Db":-5,"Gb":-6,"Cb":-7};
const STEP_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const STEP_FLAT  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"];
const DIATONIC_MAJOR = [0,2,4,5,7,9,11];
const DIATONIC_NAT_MINOR = [0,2,3,5,7,8,10];
const DIATONIC_HARM_MINOR = [0,2,3,5,7,8,11];
const FLAT_KEYS = new Set(["F","Bb","Eb","Ab","Db","Gb","Cb"]);
const prefersFlats = (k) => FLAT_KEYS.has(k);

const RAGA_DEGREES = {
  "Shankarabharanam (Major)": [0,2,4,5,7,9,11],
  "Kalyani (Lydian #4)": [0,2,4,6,7,9,11],
  "Harikambhoji (Mixolydian)": [0,2,4,5,7,9,10],
  "Kharaharapriya (Dorian)": [0,2,3,5,7,9,10],
  "Natabhairavi (Natural Minor)": [0,2,3,5,7,8,10],
  "Keeravani (Harmonic Minor)": [0,2,3,5,7,8,11],
  "Charukesi": [0,2,4,5,7,8,11],
  "Sarasangi": [0,1,4,5,7,9,11],
  "Mararanjani": [0,2,4,6,7,8,10],
  "Mohanam (Major Pentatonic)": [0,2,4,7,9],
  "Hamsadhwani": [0,2,4,7,11],
  "Hindolam": [0,3,5,8,11],
  "Suddha Saveri": [0,2,5,7,9],
  "Abhogi": [0,2,3,5,7],
  "Sreeranjani": [0,2,3,6,9],
  "Madhyamavati": [0,2,5,7,10],
  "Megh (Megh Malhar)": [0,2,5,7,9],
  "Durga": [0,2,5,7,9,10],
  "Devakriya (Sudha Dhanyasi)": [0,2,3,7,9],
  "Revati": [0,1,5,7,11],
  "Amritavarshini": [0,4,6,7,11],
  "Vachaspati (Lydian b7)": [0,2,4,6,7,9,10],
  "Hemavati": [0,2,3,6,7,9,11],
  "Shubhapantuvarali": [0,1,4,5,7,8,11],
  "Todi (Hanumatodi)": [0,1,3,6,7,8,11],
};
const RAGA_LIST = ["None (use Western mode)", ...Object.keys(RAGA_DEGREES)];

const STYLE_PATTERNS = {
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
};

const INSTRUMENTS = {
  "Flute":         {"role":"lead",    "range":[72,96]},
  "Oboe":          {"role":"lead",    "range":[67,91]},
  "Clarinet":      {"role":"counter", "range":[60,88]},
  "Reed Section":  {"role":"counter", "range":[60,92]},
  "Violin":        {"role":"lead",    "range":[67,96]},
  "Viola":         {"role":"counter", "range":[55,79]},
  "Cello":         {"role":"bass",    "range":[48,67]},
  "Double Bass":   {"role":"bass",    "range":[40,64]},
  "Piano Pad":     {"role":"pad",     "range":[48,84]},
  "Strings Pad":   {"role":"pad",     "range":[55,91]},
};

// ---------- Utility funcs ----------
function genRhythm(patterns, bars) { return Array.from({length: bars}, (_, i) => patterns[i % patterns.length]).flat(); }

function normalizeKeyMode(key, mode) {
  const k = MAJOR_PC.hasOwnProperty(key) ? key : "C";
  const m = (mode === "major" || mode === "natural_minor" || mode === "harmonic_minor") ? mode : "major";
  return [k, m];
}

function midiToPitch(midi, flats=false) {
  const names = flats ? STEP_FLAT : STEP_SHARP;
  const pc = midi % 12;
  const name = names[pc];
  const step = name[0];
  let alter = 0;
  if (name.length === 2) alter = name[1] === "#" ? 1 : -1;
  const octave = Math.floor(midi/12) - 1;
  return [step, alter, octave];
}
function midiToName(midi, flats=false) {
  const names = flats ? STEP_FLAT : STEP_SHARP;
  return `${names[midi % 12]}${Math.floor(midi/12)-1}`;
}

function ragaScaleMidis(key, raga, low=36, high=96) {
  const tonic = MAJOR_PC[key];
  if (!raga || raga === "None (use Western mode)") return Array.from({length: high-low+1}, (_,i)=>low+i);
  const degrees = new Set(RAGA_DEGREES[raga] || DIATONIC_MAJOR);
  const allowed = new Set([...degrees].map(d => (tonic + d) % 12));
  const out = [];
  for (let m=low; m<=high; m++) if (allowed.has(m%12)) out.push(m);
  return out;
}

function westernScaleMidis(key, mode, low=36, high=96) {
  const tonic = MAJOR_PC[key];
  let degs;
  if (mode==="major") degs = DIATONIC_MAJOR;
  else if (mode==="natural_minor") degs = DIATONIC_NAT_MINOR;
  else degs = DIATONIC_HARM_MINOR;
  const allowed = new Set(degs.map(d=> (tonic+d)%12));
  const out = [];
  for (let m=low; m<=high; m++) if (allowed.has(m%12)) out.push(m);
  return out;
}

function scaleMidis(key, mode, raga, low=36, high=96) {
  return (raga && raga!=="None (use Western mode)") ? ragaScaleMidis(key, raga, low, high) : westernScaleMidis(key, mode, low, high);
}
function tonicNearMiddle(key, scale, middle=60) {
  const tonicPc = MAJOR_PC[key];
  const cands = scale.filter(m => (m%12) === tonicPc);
  const pool = cands.length ? cands : scale;
  return pool.reduce((best, m)=> Math.abs(m-middle) < Math.abs(best-middle) ? m : best, pool[0]);
}
const consonantWith = (a,b)=> {
  const d = Math.abs((a-b)%12);
  return [0,3,4,5,7,8,9].includes(d);
};
const stepOptions = (p, scale) => [p-2,p-1,p+1,p+2].filter(x => scale.includes(x));

// ---------- Lead generator ----------
const ALLOWED_INTERVALS = [-7,-5,-4,-3,-2,-1,1,2,3,4,5,7];
const BASE_W = Object.fromEntries(ALLOWED_INTERVALS.map(i => [i, (Math.abs(i)<=4)?(6-Math.abs(i)):( [5,7].includes(Math.abs(i))?2:1 )]));

function sampleInterval(prevIv, weights, rnd) {
  const w = {...weights};
  if (Math.abs(prevIv) >= 4) {
    for (const i of [1,2]) {
      const key = (prevIv>0 ? -i : i);
      w[key] = (w[key] || 1) * 1.7;
    }
  }
  const tot = Object.values(w).reduce((a,b)=>a+b,0);
  let r = rnd() * tot, c=0;
  for (const [iv, wt] of Object.entries(w)) { c += wt; if (r <= c) return parseInt(iv,10); }
  // fallback
  return ALLOWED_INTERVALS[Math.floor(rnd()*ALLOWED_INTERVALS.length)];
}

function chooseNearestScale(target, scale) {
  return scale.reduce((best, m)=> Math.abs(m-target) < Math.abs(best-target) ? m : best, scale[0]);
}

function generateLeadEighths(key, mode, raga, bars, register=[60,84], rng) {
  const scale = scaleMidis(key, mode, raga, register[0], register[1]);
  let start = tonicNearMiddle(key, scale, 60);
  const line = [start];
  const weights = {...BASE_W};
  let prev = 0;
  const total8 = bars * 8;
  for (let i=1;i<total8;i++){
    const iv = sampleInterval(prev, weights, rng);
    const cand = line[line.length-1] + iv;
    const nearest = chooseNearestScale(cand, scale);
    line.push(nearest); prev = iv;
  }
  line[line.length-1] = tonicNearMiddle(key, scale, line[line.length-1]);
  return line;
}

// ---------- Counterpoint (simplified) ----------
const SPECIES_RHYTHM = {"1":[8], "2":[4,4], "3":[2,2,2,2], "4":[6,2], "5":[1,1,2,1,1,2], "Classical":[1,1,2,1,1,2]};

function generateCounterSpecies(cantusSlots, key, mode, raga, bars, species, register=[55,84]) {
  const scale = scaleMidis(key, mode, raga, register[0], register[1]);
  const total = bars * 8;
  const out = [];
  let i=0;
  function nearestConsonant(c) {
    return scale.reduce((best, n)=> {
      const score = (consonantWith(n,c)?0:1)*1000 + Math.abs(n-c);
      const bestScore = (consonantWith(best,c)?0:1)*1000 + Math.abs(best-c);
      return (score < bestScore) ? n : best;
    }, scale[0]);
  }
  while (i < total) {
    const c = cantusSlots[i];
    if (species === "1") {
      const cand = nearestConsonant(c);
      out.push(...Array(8).fill(cand)); i += 8;
    } else if (species === "2") {
      const first = nearestConsonant(c);
      const second = (stepOptions(first, scale)[0] ?? first);
      out.push(...Array(4).fill(first), ...Array(4).fill(second)); i += 8;
    } else if (species === "3") {
      const b1 = nearestConsonant(c), s1 = (stepOptions(b1, scale)[0] ?? b1);
      const b3 = nearestConsonant(c), s3 = (stepOptions(b3, scale)[0] ?? b3);
      out.push(b1,b1,s1,s1,b3,b3,s3,s3); i += 8;
    } else if (species === "4") {
      const held = nearestConsonant(c);
      const res = stepOptions(held, scale).filter(n=>n<held) // tendency down
      out.push(...Array(6).fill(held), (res[0] ?? held), (res[0] ?? held)); i += 8;
    } else { // "5"/"Classical"
      const b1 = nearestConsonant(c);
      const n1 = (stepOptions(b1, scale)[0] ?? b1);
      const n2 = (stepOptions(n1, scale)[0] ?? n1);
      const b3 = nearestConsonant(c);
      const n3 = (stepOptions(b3, scale)[0] ?? b3);
      const n4 = (stepOptions(n3, scale)[0] ?? n3);
      out.push(b1, n1, n2, n2, b3, n3, n4, n4); i += 8;
    }
  }
  return out.slice(0,total);
}
function avoidParallelsAndAccented(lead, other, scale) {
  const out = other.slice();
  const total = Math.min(lead.length, other.length);
  for (let i=0;i<total;i++){
    const beat = i % 8;
    const interval = Math.abs((lead[i] - out[i]) % 12);
    if (interval === 0 || interval === 7) {
      if (i>0 && (lead[i]-lead[i-1])*(out[i]-out[i-1]) > 0) {
        const opts = stepOptions(out[i], scale);
        if (opts.length) {
          out[i] = opts.reduce((best,x)=>{
            const cur = Math.abs((lead[i]-x)%12);
            const bestCur = Math.abs((lead[i]-best)%12);
            return Math.abs(cur-3) < Math.abs(bestCur-3) ? x : best;
          }, opts[0]);
        }
      }
    }
    if ([0,2,4,6].includes(beat)) {
      if (![0,3,4,5,7,8,9].includes(interval)) {
        const opts = stepOptions(out[i], scale);
        if (opts.length) {
          out[i] = opts.reduce((best,x)=>{
            const ok = [0,3,4,5,7,8,9].includes(Math.abs((lead[i]-x)%12)) ? 0 : 1;
            const bestOk = [0,3,4,5,7,8,9].includes(Math.abs((lead[i]-best)%12)) ? 0 : 1;
            return (ok<bestOk) ? x : best;
          }, opts[0]);
        }
      }
    }
  }
  return out;
}

// ---------- Raga-fugal (simplified to match headless constraints) ----------
function ragaPcs(key, raga) { const tonic = MAJOR_PC[key]; return RAGA_DEGREES[raga].map(d=> (tonic+d)%12); }
function ragaDegreeIndex(pc, pcs) {
  return pcs.reduce((bestIdx, _, i) => {
    const d = Math.min((pc - pcs[i])%12, (pcs[i] - pc)%12);
    const bestD = Math.min((pc - pcs[bestIdx])%12, (pcs[bestIdx] - pc)%12);
    return (d<bestD) ? i : bestIdx;
  }, 0);
}
function transposeDegreeInRaga(midi, key, raga, steps) {
  const pcs = ragaPcs(key, raga);
  const idx = ragaDegreeIndex(midi%12, pcs);
  const tgtPc = pcs[(idx + steps) % pcs.length];
  // search nearest same-pc note
  let best = midi;
  let bestDist = 1e9;
  for (let delta=-48; delta<=48; delta++) {
    const cand = midi + delta;
    if ((cand % 12) === tgtPc) {
      const d = Math.abs(delta);
      if (d < bestDist) { bestDist = d; best = cand; }
    }
  }
  return best;
}
function ragaAnswerSteps(key, raga) {
  const pcs = new Set(ragaPcs(key, raga));
  const tonic = MAJOR_PC[key];
  const pa = (tonic + 7) % 12;
  const ma = (tonic + 5) % 12;
  const degs = ragaPcs(key, raga);
  if (pcs.has(pa)) return degs.indexOf(pa);
  if (pcs.has(ma)) return degs.indexOf(ma);
  return 0;
}
function clipToRange(m, rng, scale) {
  let x = m;
  while (x < rng[0]) x += 12;
  while (x > rng[1]) x -= 12;
  return scale.reduce((best,c)=> (rng[0]<=c && c<=rng[1] && Math.abs(c-x)<Math.abs(best-x)) ? c : best, scale[0]);
}
function contraryStep(prev, target, scale) {
  const dir = target > prev ? -1 : 1;
  const cands = [prev + dir, prev + 2*dir].filter(c => scale.includes(c));
  return cands[0] ?? prev;
}
function buildRagaFugalParts(leadSlots, key, raga, bars, insts) {
  const total = bars*8;
  const parts = {};
  const scaleCache = {};
  insts.forEach(inst => { scaleCache[inst.name] = scaleMidis(key, "major", raga, inst.range[0], inst.range[1]); });

  // Lead
  const leadInst = insts.find(i=>i.role==="lead");
  parts[leadInst.name] = { slots: leadSlots.slice(0,total) };

  const degSteps = ragaAnswerSteps(key, raga);
  const roleOffsets = { counter: 8, bass: 4, pad: 16 };

  for (const inst of insts) {
    if (inst.role === "lead") continue;
    const rng = inst.range;
    const rscale = scaleCache[inst.name];
    const off = roleOffsets[inst.role] ?? 8;
    const raw = [];
    for (let i=0;i<total;i++){
      const srcIdx = Math.max(0, i - off);
      const src = leadSlots[srcIdx] ?? leadSlots[0];
      let im = transposeDegreeInRaga(src, key, raga, degSteps);
      if (i>0 && i%4===0) im = contraryStep(raw[raw.length-1] ?? im, im, rscale);
      im = clipToRange(im, rng, rscale);
      raw.push(im);
    }
    parts[inst.name] = { slots: raw.slice(0,total) };
  }

  // avoid unisons with lead
  const lead = parts[leadInst.name].slots;
  for (const inst of insts) {
    if (inst.role === "lead") continue;
    const rng = inst.range;
    const rscale = scaleCache[inst.name];
    const out = parts[inst.name].slots.slice();
    for (let i=0;i<total;i++){
      if (Math.abs((lead[i]-out[i])%12)===0) {
        const opts = [out[i]-1,out[i]+1,out[i]-2,out[i]+2].filter(x => rscale.includes(x) && rng[0]<=x && x<=rng[1]);
        if (opts.length) out[i] = opts.reduce((best,x)=> ( [0,7].includes(Math.abs((lead[i]-x)%12)) ? 1 : 0) < ( [0,7].includes(Math.abs((lead[i]-best)%12)) ? 1 : 0) ? x : best, opts[0]);
      }
    }
    parts[inst.name].slots = out;
  }
  return parts;
}

// ---------- MusicXML ----------
function partsToMusicXML(parts, key, mode, raga, tempo, bars) {
  const flats = prefersFlats(key);
  function emitNote(midi, dur8, divisions) {
    const [step, alter, octave] = midiToPitch(midi, flats);
    const typeMap = {1:"eighth",2:"quarter",4:"half",8:"whole"};
    let alterTag = (alter !== 0) ? `<alter>${alter}</alter>` : "";
    const duration = Math.floor(dur8 * (divisions/1));
    return `<note><pitch><step>${step}</step>${alterTag}<octave>${octave}</octave></pitch><duration>${duration}</duration><voice>1</voice><type>${typeMap[dur8]||"eighth"}</type></note>`;
  }
  const divisions = 2;
  // part-list
  let partList = parts.map((p, i) => `<score-part id="P${i+1}"><part-name>${p.name}${(raga && raga!=="None (use Western mode)")?` (${raga})`:""}</part-name></score-part>`).join("");
  let xml = `<?xml version="1.0" encoding="UTF-8"?>\n<score-partwise version="3.1"><part-list>${partList}</part-list>`;
  // parts
  for (let i=0;i<parts.length;i++){
    const p = parts[i];
    let events = [];
    let idx = 0;
    for (const dur of p.rhythm) {
      const pitch = p.slots[idx];
      events.push([pitch, dur]);
      idx += dur;
    }
    let measNum = 1, used = 0;
    let meas = [];
    const header = `<attributes><divisions>${divisions}</divisions><key><fifths>${FIFTHS[key]}</fifths><mode>${mode==="major"?"major":"minor"}</mode></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes><direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${tempo}</per-minute></metronome></direction-type><sound tempo="${tempo}"/></direction>`;
    let measures = [`<measure number="${measNum}">${header}`];
    for (let [pitch, dur8] of events) {
      while (used + dur8 > 8) {
        const split = 8 - used;
        measures[measures.length-1] += emitNote(pitch, split, divisions) + `</measure>`;
        measNum += 1; measures.push(`<measure number="${measNum}">`); used = 0; dur8 -= split;
      }
      measures[measures.length-1] += emitNote(pitch, dur8, divisions);
      used += dur8;
    }
    if (!measures[measures.length-1].endsWith("</measure>")) measures[measures.length-1] += `</measure>`;
    xml += `<part id="P${i+1}">` + measures.join("") + `</part>`;
  }
  xml += `</score-partwise>`;
  return xml;
}

// ---------- Minimal MIDI writer (single track, multiple channels) ----------
const PROGRAMS_GM = {"lead":73,"counter":71,"bass":32,"pad":88};
const PROGRAMS_PER_INST = {"Violin":40,"Viola":41,"Cello":42,"Double Bass":43,"Flute":73,"Oboe":68,"Clarinet":71,"Reed Section":65,"Piano Pad":0,"Strings Pad":48};

function vlq(n){ // variable-length quantity
  let buffer = n & 0x7F;
  const bytes = [];
  while ((n >>= 7)) { buffer <<= 8; buffer |= ((n & 0x7F) | 0x80); }
  while (true) { bytes.push(buffer & 0xFF); if (buffer & 0x80) buffer >>= 8; else break; }
  return bytes;
}
function midiDelta(t, current){ return vlq(Math.max(0, t-current)); }

function partsToMIDI(parts, tempoBpm) {
  const ticksPerBeat = 480;
  const usPerBeat = Math.floor(60000000 / Math.max(1, tempoBpm));
  const events = []; // {t, type, a, b, ch}
  let ch = 0;
  for (const p of parts) {
    const program = PROGRAMS_PER_INST[p.name] ?? PROGRAMS_GM[p.role] ?? 73;
    events.push({t:0, type:"pc", a:program, ch});
    let tAbs = 0, idx = 0;
    for (const dur8 of p.rhythm) {
      const pitch = p.slots[idx];
      const durQ = dur8/2.0;
      const durTicks = Math.floor(durQ * ticksPerBeat);
      events.push({t:tAbs, type:"on", a:pitch, b:92, ch});
      events.push({t:tAbs+durTicks, type:"off", a:pitch, b:64, ch});
      tAbs += durTicks;
      idx += dur8;
    }
    ch = (ch + 1) % 16;
  }
  // sort (program-change first at same tick, offs before ons)
  const pri = e => e.type==="pc"?0 : e.type==="off"?1 : 2;
  events.sort((A,B)=> (A.t-B.t) || (pri(A)-pri(B)));

  // Build track data
  const track = [];
  // tempo meta
  track.push(...vlq(0), 0xFF, 0x51, 0x03, (usPerBeat>>16)&0xFF, (usPerBeat>>8)&0xFF, usPerBeat&0xFF);
  let current = 0;
  for (const e of events) {
    track.push(...midiDelta(e.t, current));
    current = e.t;
    if (e.type==="pc") {
      track.push(0xC0 | (e.ch & 0x0F), e.a & 0x7F);
    } else if (e.type==="on") {
      track.push(0x90 | (e.ch & 0x0F), e.a & 0x7F, e.b & 0x7F);
    } else if (e.type==="off") {
      track.push(0x80 | (e.ch & 0x0F), e.a & 0x7F, e.b & 0x7F);
    }
  }
  // end of track
  track.push(...vlq(0), 0xFF, 0x2F, 0x00);

  // MIDI header
  function chunk(tag, bytes) {
    const len = bytes.length;
    return [
      ...Buffer.from(tag, 'ascii'),
      (len>>>24)&0xFF,(len>>>16)&0xFF,(len>>>8)&0xFF,len&0xFF,
      ...bytes
    ];
  }
  const hdr = [
    ...Buffer.from("MThd",'ascii'),
    0x00,0x00,0x00,0x06,
    0x00,0x00, // format 0
    0x00,0x01, // nTracks=1
    (ticksPerBeat>>8)&0xFF, ticksPerBeat&0xFF
  ];
  const trk = chunk("MTrk", track);
  const all = Buffer.from([...hdr, ...trk]);
  return all;
}

// ---------- MAIN ----------
const inJson = items[0]?.json || {};
const params = {
  key: inJson.key ?? "C",
  mode: inJson.mode ?? "major",
  raga: inJson.raga ?? "None (use Western mode)",
  style: inJson.style ?? "R&B",
  species: inJson.species ?? "Classical",
  bars: Math.max(4, Math.min(128, Number(inJson.bars ?? 32))),
  targetTokens: Number(inJson.targetTokens ?? 0),
  tempo: Math.max(30, Math.min(240, Number(inJson.tempo ?? 96))),
  seed: Number(inJson.seed ?? 0),
  instruments: Array.isArray(inJson.instruments) && inJson.instruments.length
    ? inJson.instruments : ["Flute","Reed Section","Double Bass","Strings Pad","Piano Pad"]
};

const rng = lcg(params.seed || 1);
const barsEff = params.targetTokens>0 ? Math.ceil(params.targetTokens/8) : params.bars;

// Build patterns
const patt = STYLE_PATTERNS[params.style] || STYLE_PATTERNS["R&B"];
const rhythms = Object.fromEntries(Object.entries(patt).map(([role, pat]) => [role, genRhythm(pat, barsEff)]));

// Assemble instruments
let insts = params.instruments.filter(n => INSTRUMENTS[n]).map(n => ({ name:n, ...INSTRUMENTS[n] }));
if (!insts.length) insts = [{ name:"Flute", ...INSTRUMENTS["Flute"] }];
if (!insts.some(i=>i.role==="lead")) insts[0].role = "lead";

// Choose effective mode (raga controls pitch if chosen)
const effectiveMode = (params.raga === "None (use Western mode)") ? params.mode : "major";
const [keyN, modeN] = normalizeKeyMode(params.key, effectiveMode);

// Lead
const leadInst = insts.find(i=>i.role==="lead");
const leadSlots = generateLeadEighths(keyN, modeN, params.raga, barsEff, leadInst.range, rng);

// Build parts
let partsArr = [];
if (params.raga !== "None (use Western mode)") {
  const rparts = buildRagaFugalParts(leadSlots, keyN, params.raga, barsEff, insts);
  for (const inst of insts) {
    partsArr.push({ name:inst.name, role:inst.role, rhythm:rhythms[inst.role], slots:rparts[inst.name].slots });
  }
} else {
  for (const inst of insts) {
    const role = inst.role;
    const rhythm = rhythms[role];
    let slots;
    if (role === "lead") {
      slots = leadSlots.slice(0, barsEff*8);
    } else if (role === "counter") {
      slots = generateCounterSpecies(leadSlots, keyN, modeN, null, barsEff, params.species, inst.range);
      const scl = scaleMidis(keyN, modeN, null, inst.range[0], inst.range[1]);
      slots = avoidParallelsAndAccented(leadSlots.slice(0,barsEff*8), slots, scl);
    } else if (role === "bass") {
      const baseScale = scaleMidis(keyN, modeN, null, inst.range[0], inst.range[1]);
      slots = [];
      for (let i=0;i<leadSlots.slice(0,barsEff*8).length;i++) {
        const pn = leadSlots[i];
        const target = ([0,4].includes(i%8)) ? pn-12 : pn-7;
        slots.push(baseScale.reduce((best,m)=> Math.abs(m-target)<Math.abs(best-target) ? m : best, baseScale[0]));
      }
    } else { // pad
      const baseScale = scaleMidis(keyN, modeN, null, inst.range[0], inst.range[1]);
      slots = [];
      const leadS = leadSlots.slice(0,barsEff*8);
      for (let i=0;i<leadS.length;i++) {
        const pn = leadS[i];
        const target = ([0,4].includes(i%8)) ? pn : (slots.length? slots[slots.length-1] : pn);
        slots.push(baseScale.reduce((best,m)=> Math.abs(m-target)<Math.abs(best-target) ? m : best, baseScale[0]));
      }
    }
    partsArr.push({ name:inst.name, role, rhythm, slots });
  }
}

// Exports
const musicxml = partsToMusicXML(partsArr, keyN, modeN, params.raga, params.tempo, barsEff);
const midiBuf = partsToMIDI(partsArr, params.tempo);

// Preview names
const flats = prefersFlats(keyN);
const preview = partsArr.map(p => ({
  part: `${p.name} (${p.role})`,
  first16: p.slots.slice(0,16).map(m => midiToName(m, flats))
}));

// Return one item with JSON + binary
return [
  {
    json: {
      summary: {
        app: "CR-Counter v2 (n8n headless)",
        effectiveBars: barsEff,
        tempo: params.tempo,
        key: keyN,
        mode: modeN,
        raga: params.raga,
        instruments: insts.map(i=>i.name),
        preview
      },
      params
    },
    binary: {
      musicxml: {
        data: Buffer.from(musicxml, 'utf8').toString('base64'),
        fileName: "CR_Counter_Arrangement.xml",
        mimeType: "application/vnd.recordare.musicxml+xml"
      },
      midi: {
        data: midiBuf.toString('base64'),
        fileName: "CR_Counter_Arrangement.mid",
        mimeType: "audio/midi"
      }
    }
  }
];
