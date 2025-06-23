# MIDIfren 🎵 Audio Stem & MIDI Processor 🧠  

**Convert audio stems to MIDI, extract beats, and play back with customizable settings!**  

---

### 💡 Features  
- **Stem Separation**: Extract vocals, melody, drums, or bass from audio 🎙️  
- **BPM Detection**: Auto-detect tempo or set manually ⏱️  
- **MIDI Conversion**: Convert stems or audio to MIDI with adjustable sensitivity and pitchbend 📊  
- **Playback**: Listen to exported MIDI files instantly 🎹  
- **Quantization**: Align notes to 16th-note grids (or custom ticks) 🔧  

---
### 📦 Installation  
```bash
# python version >= 3.8
git clone or download this repo and unzip
cd MIDIfren/
pip install -r requirements.txt
```  

### 📦 Usage  
```bash
python MIDIfren.py -i <input_audio> -t <sound_type> [options]
```  

**Options**:  
- `-t, --type` : Choose type: `vocals`, `melody`, `drums`, `bass`  
- `-b, --bpm` : Set BPM (e.g., `120`) or skip this and MIDIfren will autodetect
- `-m, --midi` : Convert audio to midi  
- `-s, --stem` : Extract stem based on type selected 
- `-q, --quantize` : Quantize generated midi  
- `-p, --pitchbend` : Detect pitchbend  
- `-o, --onset` : Adjust note trigger sensitivity (0-1, default = 1)  
- `-l, --listen` : Play generated MIDI file or exported stem immediately 🎶  

---


### 📌 Example  
```bash
python MIDIfren.py -i input.wav -t drums --listen
```  
**Output**:  
- `output/drums.wav` (stem)  
- `output/drums.mid` (MIDI)  
- Plays MIDI or stem at detected BPM 🎵  

---  
**Made by Omodaka9375 with ❤️ for audio enthusiasts & music devs!** 🎶