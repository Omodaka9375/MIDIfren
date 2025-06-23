# MIDIfren 🎵 Audio Stem & MIDI Processor 🧠  

**Convert audio to stems to MIDI! Extract melody, bass, drums to midi and play back with customizable settings!**  

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
### 📦 Usage with UI
```bash
python MIDIfren.py -u
```  
![webui](u.png "Gradio UI")

### 📦 Usage with terminal 
```bash
python MIDIfren.py -i <input_audio> -t <sound_type> [options]
```  
![terminal](h.png "Help Page")

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
# extract vocal stem from the audio and convert it to midi (with pitchbend) and listen to it
python MIDIfren.py -i input.wav -type vocals --pitchbend --listen --onset 1.0
```  
---
**Output**:  

Midi files, sonified midi and stems will be in the output folder.
- `output/drums.wav` (stem)  
- `output/drums.mid` (MIDI)  
- ...

---  
**Made by Omodaka9375 with ❤️ for audio enthusiasts & music devs!** 🎶
