# make by: https://github.com/Omodaka9375 2025
# repo: https://github.com/Omodaka9375/MIDIfren
# Melodic instruments conversion using Spotify's basic-pitch converter https://github.com/spotify/basic-pitch
# version 1.0
import argparse
import sys
import os
import time
import pretty_midi
import numpy as np
import torchaudio
import librosa
import mido
import torch
import pygame
import gradio as gr
import basic_pitch.note_creation as infer
from basic_pitch.inference import predict, predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
from demucs.pretrained import get_model
from demucs.apply import apply_model
from mido import Message, MidiTrack, MidiFile, MetaMessage
from typing import Optional, Tuple, List
from pathlib import Path

# Create a persistent directory for outputs
OUTPUT_DIR = Path("/tmp/audio_processor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(add_help=True, description="Convert audio to stems and/or midi file and listen to it", prog="MIDIfren",  epilog="Thank you for using MIDIfren <3 Omodaka9375 2025")
    parser.add_argument("-i","--input", help="Input audio file (wav, mp3 or flac)", type=str)
    parser.add_argument("-t", "--type", choices=['drums', 'bass', 'melody', 'vocals'])
    parser.add_argument("-m", "--midi", help="Convert to midi", action="store_true")
    parser.add_argument("-q", "--quantize", help="Quantize midi file", action="store_true")
    parser.add_argument("-s", "--stem", help="Extract stem", action="store_true")
    parser.add_argument("-p", "--pitchbend", help="Include pitchbend in midi", action="store_true")
    parser.add_argument("-b", "--bpm", type=float, help="Set specific BPM (beats per minute) for the MIDI file")
    parser.add_argument("-n", "--note", type=float, help="Ser minimal note length. Everything below will be ignored.")
    parser.add_argument("-o", "--onset", type=float, help="Set specific threshold for triggering notes. Range 0-1. Default 1. Bigger number less notes.")
    parser.add_argument("-l","--listen", help="Play given MIDI file", action="store_true")
    parser.add_argument("-g","--groove", help="Set time signature for the file", action="store_true")
    parser.add_argument("-w","--web", help="Launch localhost gradio webUI", action="store_true")
    args = parser.parse_args()
    
    _bpm = 120
    _groove = "4/4"
    _notelength = 127.70
    stem_type = None
    _quantize = True

    out_folder_path = Path("output")
    out_folder_path.mkdir(parents=True, exist_ok=True)

    if args.web:
        interface = create_interface()
        interface.launch(share=False,server_name="0.0.0.0", server_port=7860, auth=None, ssl_keyfile=None, ssl_certfile=None)
    else:
        if args.bpm:
            _bpm = args.bpm
        else:
            drumExtractor = DrumBeatExtractor()
            _bpm = drumExtractor.detect_tempo(args.input)   
        
        if args.groove:
            _groove = args.groove
        
        if args.note:
            _notelength = args.note

        if args.type:
            stem_type = args.type

        if args.stem:
            if args.type:
                stem_type = args.type
                processor = DemucsProcessor()
                sources, sample_rate = processor.separate_stems(args.input)
                print(f"Stem type requested: {stem_type}")
                stem_index = ['drums', 'bass', 'melody', 'vocals'].index(stem_type)
                selected_stem = sources[0, stem_index]
                # Save stem
                stem_path = out_folder_path / f"{stem_type}.wav"
                processor.save_stem(selected_stem, stem_type, out_folder_path, sample_rate)
                print(f"Saved stem to: {stem_path}")
                # Load the saved audio file
                audio_data, sr = librosa.load(stem_path, sr=None, mono=True)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
                # Convert to int16 format
                audio_data = (audio_data * 32767).astype(np.int16)
                if args.listen:
                    timer = librosa.get_duration(path=stem_path)
                    pygame.mixer.init()
                    pygame.mixer.music.set_volume(0.8)
                    time.sleep(1)
                    pygame.mixer.music.load(stem_path)
                    pygame.mixer.music.play()
                    time.sleep(timer)
            else:
                print("Error! You need to select type of stem to separate. Check manual.")
        else:
            # Load the saved audio file
            audio_data, sr = librosa.load(args.input, sr=None, mono=True)
            yt, index = librosa.effects.trim(audio_data)
            y_normalized = librosa.util.normalize(yt)
            if len(y_normalized.shape) > 1:
                y_normalized = y_normalized.mean(axis=1)  # Convert to mono if stereo
            # Convert to int16 format
            y_normalized = (y_normalized * 32767).astype(np.int16)
            stem_path = args.input
        
        if args.midi:
            if args.type:
                stem_type = args.type
                midi_path = out_folder_path / f"{stem_type}.mid"
                if stem_type == "drums":
                    _onset = 0.1 # bigger onset less notes
                    if args.onset:
                        _onset = float(args.onset)
                    drumExtractor = DrumBeatExtractor()
                    if args.quantize:
                        _quantize = args.quantize
                    drumExtractor.extract_midi(stem_path, _bpm, _onset, _groove,_quantize, False) # sensitivity
                    print(f"Saved MIDI to: {midi_path}")
                    if args.listen:
                        midiplayer = MidiDrumPlayer(midi_path, _bpm)
                        time.sleep(1)
                        midiplayer.playonce()
                else:
                    _sonify = False
                    _pitchbend = False
                    _onset = 1.0 # bigger onset less notes
                    if args.listen:
                        _sonify = True
                    if args.pitchbend:
                        _pitchbend = True
                    if args.onset:
                        _onset = float(args.onset)
                    converter = BasicPitchConverter()
                    converter.convert_to_midi(str(stem_path), str(midi_path), _bpm, _sonify, _pitchbend, _onset, _notelength)
                    print(f"Saved MIDI to: {midi_path}")

def process_single_audio(audio_path: str, stem_type: str, convert_midi: bool, separate_stems: bool, bpm: int, sensitivity: float, pitchbend: bool, notelength: float = 127.70, groove: str = "4/4", quantize: bool = True) -> Tuple[Tuple[int, np.ndarray], Optional[str]]:
    _bpm = bpm
    stem_path = None
    try:
        drumExtractor = DrumBeatExtractor()
        if _bpm <= -1:
            _bpm = drumExtractor.detect_tempo(audio_path)
    except:
        _bpm = 120

    try:
        process_dir = OUTPUT_DIR / str(hash(audio_path))
        process_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting processing of file: {audio_path}")
        print(f"Separating stems: {separate_stems}, bpm: {_bpm}")
        if separate_stems:
            # Create unique subdirectory for this processing
            processor = DemucsProcessor()
            # Process stems
            sources, sample_rate = processor.separate_stems(audio_path)
            print(f"Number of sources returned: {sources.shape}")
            print(f"Stem type requested: {stem_type}")
            # Get the requested stem
            stem_index = ['drums', 'bass', 'melody', 'vocals'].index(stem_type)
            selected_stem = sources[0, stem_index]
            # Save stem
            stem_path = process_dir / f"{stem_type}.wav"
            processor.save_stem(selected_stem, stem_type, str(process_dir), sample_rate)
            print(f"Saved stem to: {stem_path}")
            # Load the saved audio file for Gradio
            audio_data, sr = librosa.load(stem_path, sr=None, mono=True)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
            # Convert to int16 format
            audio_data = (audio_data * 32767).astype(np.int16)
            new_audio = audio_data
        else:
            # Load the saved audio file for Gradio
            audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
            yt, index = librosa.effects.trim(audio_data)
            y_normalized = librosa.util.normalize(yt)
            if len(y_normalized.shape) > 1:
                y_normalized = y_normalized.mean(axis=1)  # Convert to mono if stereo
            # Convert to int16 format
            y_normalized = (y_normalized * 32767).astype(np.int16)
            stem_path = audio_path
            new_audio = y_normalized

        # Convert to MIDI if requested
        midi_path = process_dir / f"{stem_type}.mid"
        
        if convert_midi:
            if stem_type == "drums":
                drumExtractor.extract_midi(stem_path, _bpm, sensitivity, groove, quantize, True)
                print(f"Saved MIDI to: {midi_path}")
            else:
                converter = BasicPitchConverter()
                converter.convert_to_midi(str(stem_path), str(midi_path), _bpm, False, pitchbend, sensitivity, notelength)
                print(f"Saved MIDI to: {midi_path}")
                
        return (sr, new_audio), str(midi_path) if midi_path else None
    except Exception as e:
        print(f"Error in process_single_audio: {str(e)}")
        raise

def create_interface():

    def process_audio(
        audio_files: List[str],
        stem_type: str,
        convert_midi: bool = True,
        separate_stems: bool = True,
        bpm: int = -1,
        sensitivity: float = 0.3,
        pitchbend: bool = True,
        notelength: float = 127.70,
        groove: str = "4/4",
        quantize: bool = True
    ) -> Tuple[Tuple[int, np.ndarray], Optional[str]]:
        try:
            print(f"Starting processing of {len(audio_files)} files")
            print(f"Selected stem type: {stem_type}")
            # Process single file for now
            if len(audio_files) > 0:
                audio_path = audio_files[0]  # Take first file
                print(f"Processing file: {audio_path}")
                return process_single_audio(audio_path, stem_type, convert_midi, separate_stems, bpm, sensitivity, pitchbend, notelength, groove, quantize)
            else:
                raise ValueError("No audio files provided")
        except Exception as e:
            print(f"Error in audio processing: {str(e)}")
            raise gr.Error(str(e))

    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.File(
                file_count="multiple",
                file_types=AudioValidator.SUPPORTED_FORMATS,
                label="Upload Audio File"
            ),
            gr.Dropdown(
                choices=['drums', 'bass', 'melody', 'vocals'],
                label="Select Stem to Extract",
                value="vocals"
            ),
            gr.Checkbox(label="Convert to MIDI", value=True),
            gr.Checkbox(label="Separate Stems", value=True),
            gr.Number(label="BPM (-1 autodetect):", value=-1),
            gr.Number(label="Sensitivity Threshold (0.0 - 1.0):", value=0.3),
            gr.Checkbox(label="Include MIDI pitchband:", value=True),
            gr.Number(label="Minimal note length in miliseconds:", value=127.70),
            gr.Text(label="Timesignature (groove)", value="4/4"),
            gr.Checkbox(label="Quantize MIDI", value=True),
            
        ],
        outputs=[
            gr.Audio(label="Separated Stem", type="numpy"),
            gr.File(label="MIDI File")
        ],
        title="MIDIfren 🎵 Audio Stem & MIDI Processor 🧠",
        description="\n\n",
        cache_examples=True,
        allow_flagging="never"
    )
    return interface    

def resource_path(relative_path):
	""" Get absolute path to resource, works for dev and for PyInstaller """
	try:
		# PyInstaller creates a temp folder and stores path in _MEIPASS
		base_path = sys._MEIPASS
	except Exception:
		base_path = os.path.abspath(".")

	return os.path.join(base_path, relative_path)	

def quantize_midi_file(input_path, output_path, resolution_ticks):
    mid = mido.MidiFile(input_path)
    new_mid = mido.MidiFile()

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        current_time = 0
        for msg in track:
            if not msg.is_meta:
                current_time += msg.time
                # Quantize note-on and note-off messages
                if msg.type in ['note_on', 'note_off']:
                    quantized_time = round(current_time / resolution_ticks) * resolution_ticks
                    msg.time = quantized_time - (current_time - msg.time) # Adjust msg.time for delta
                    current_time = quantized_time
            new_track.append(msg)
        new_mid.tracks.append(new_track)

    new_mid.save(output_path)

class DrumBeatExtractor:

    def __init__(self):
        # Analysis parameters
        self.y = None
        self.sr = None
        self.onset_frames = None
        self.drum_types = None
        
    def detect_tempo(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=None, mono=True)
            yt, index = librosa.effects.trim(y)
            tempo, _ = librosa.beat.beat_track(y=yt, sr=sr, units="time")
            print(f"Detected tempo: {int(tempo)} BPM")
            return int(tempo)
        except Exception as e:
            print("Error detecting tempo. Setting tempo to default 120.")
            return 120

    def extract_midi(self, audio_file, tempo, sensitivity, groove, quantize, web):
        times =  groove.split('/')
        nom = int(times[0])
        denom = int(times[1])
        try:
            process_dir = OUTPUT_DIR / str(hash(audio_file))
            process_dir.mkdir(parents=True, exist_ok=True)
            # Load as mono for HPSS and feature extraction
            out_folder_path = Path("output")
            
            self.y, self.sr = librosa.load(audio_file, sr=None, mono=True)
            yt, index = librosa.effects.trim(self.y)
            y_normalized = librosa.util.normalize(yt)
            # --- Perform HPSS --- # 
            y_percussive = librosa.effects.percussive(y_normalized)
            # We could also get y_harmonic = librosa.effects.harmonic(y_normalized) if needed later
            # Detect onsets (using the percussive component)
            onset_env = librosa.onset.onset_strength(
                y=y_percussive, # Use percussive component here
                sr=self.sr,
                hop_length=512,
                aggregate=np.median
            )
            # Adjust threshold with sensitivity slider
            # Lower threshold means more sensitivity (detects quieter onsets)
            # We invert the slider value (0.1 to 1.0) so higher slider means more sensitive
            wait_time = 0.04 # Corresponds to roughly 1/16th note at 120bpm, prevents double triggers
            delta_time = sensitivity # Adjust sensitivity range 0.9
            pre_avg_time = 0.1
            # post_avg_time = 0.0 # Use default = 1 frame
            pre_max_time = 0.03
            # post_max_time = 0.0 # Use default = 1 frame

            # Ensure frame counts are non-negative integers
            wait_frames = max(0, int(wait_time * self.sr / 512))
            pre_avg_frames = max(0, int(pre_avg_time * self.sr / 512))
            pre_max_frames = max(0, int(pre_max_time * self.sr / 512))

            self.onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=512,
                backtrack=True,
                units='frames',
                wait=wait_frames,
                delta=delta_time,
                pre_avg=pre_avg_frames,
                post_avg=1, # Default positive value
                pre_max=pre_max_frames,
                post_max=1 # Default positive value
            )

            onset_times = librosa.frames_to_time(self.onset_frames, sr=self.sr, hop_length=512)
            # Enhanced drum classification (using original audio segment for features)
            self.drum_types = []
            n_mfcc = 13 # Number of MFCCs to compute

            for i, frame in enumerate(self.onset_frames):
                start_sample = frame * 512
                # Use a slightly longer segment for better feature extraction
                segment_duration = 0.1 # 100ms segment
                end_sample = min(len(y_normalized), start_sample + int(segment_duration * self.sr))

                if start_sample >= end_sample:
                    continue

                segment = y_normalized[start_sample:end_sample]
                if len(segment) == 0:
                    continue

                # Calculate features
                spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sr)[0].mean()
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.sr)[0].mean()
                rms = np.sqrt(np.mean(segment**2))
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)[0].mean()
                mfccs = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=n_mfcc)
                mfcc1_mean = mfccs[0].mean()
                # mfcc2_mean = mfccs[1].mean() # Could use more MFCCs if needed

                # Refined rules with MFCCs (focus on kick)
                # Adjust these thresholds based on testing!
                kick_rms_thresh = 0.03
                kick_spec_cent_thresh = 1200 # Lowered threshold
                kick_spec_bw_thresh = 1800
                # MFCC[0] often relates to overall energy/loudness balance across spectrum
                # Kicks might have lower MFCC[0] than snares/hats? Experiment needed.
                kick_mfcc1_thresh = -150 # Example threshold, needs tuning

                snare_rms_thresh = 0.03
                snare_spec_bw_thresh = 2500
                snare_zcr_thresh = 0.08 # Slightly lower ZCR for snare vs hat

                hat_spec_cent_thresh = 2500
                hat_zcr_thresh = 0.15
                # --- Classification Logic --- #
                # Prioritize Kick detection if strong low-frequency energy detected
                if (rms > kick_rms_thresh and
                    spectral_centroid < kick_spec_cent_thresh and
                    spectral_bandwidth < kick_spec_bw_thresh and
                    mfcc1_mean > kick_mfcc1_thresh): # Added MFCC condition
                    drum_type = 36  # Kick (C1)
                # Then check for Hi-Hat (high ZCR and spectral centroid)
                elif (zero_crossing_rate > hat_zcr_thresh and
                      spectral_centroid > hat_spec_cent_thresh):
                    drum_type = 42  # Closed Hi-hat (F#1)
                # Then check for Snare (mid-range energy, higher bandwidth)
                elif (rms > snare_rms_thresh and
                      spectral_bandwidth > snare_spec_bw_thresh and 
                      zero_crossing_rate > snare_zcr_thresh):
                     drum_type = 38 # Snare (D1)
                # Fallback / Default
                else:
                    # Maybe default to kick if RMS is high but doesn't match others?
                    if rms > kick_rms_thresh * 1.5: # If it's loud but not clearly hat/snare
                         drum_type = 36 # Tentative Kick
                    else:
                         drum_type = 38 # Default to Snare if unsure

                self.drum_types.append(drum_type)

        except Exception as e:
             print(f"Error analyzing audio: {str(e)}")

        try:
            # Create a MIDI file
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            # Set tempo
            tempo_in_microseconds = mido.bpm2tempo(tempo)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo_in_microseconds))
            # Set time signature (assuming 4/4 by default) 
            track.append(mido.MetaMessage('time_signature', numerator=nom, denominator=denom))
            # Convert onset frames to ticks
            ticks_per_beat = mid.ticks_per_beat
            seconds_per_tick = 60.0 / (tempo * ticks_per_beat)
            onset_times = librosa.frames_to_time(self.onset_frames, sr=self.sr, hop_length=512)
            # Translate each onset to a MIDI note
            last_tick = 0
            for i, onset_time in enumerate(onset_times):
                tick = int(onset_time / seconds_per_tick)
                delta_time = tick - last_tick
                last_tick = tick
                
                if i < len(self.drum_types):
                    note = self.drum_types[i]
                    # Note on
                    track.append(mido.Message('note_on', note=note, velocity=100, time=delta_time))
                    # Note off (very short duration)
                    track.append(mido.Message('note_off', note=note, velocity=0, time=10))
            # Save the MIDI file
           
            if web:
                midi_path = process_dir / "drums.mid"
                mid.save(midi_path)
                if quantize:
                    quantize_midi_file(midi_path, midi_path, 120)
            else:
                midi_path = out_folder_path / "drums.mid"
                mid.save(midi_path)
                if quantize:
                    quantize_midi_file(midi_path, midi_path, 120)
              
        except Exception as e:
            print(f"Error exporting MIDI: {str(e)}")

class BasicPitchConverter:
    def __init__(self):
        self.process_options = {
            'frame_threshold': 0.3,
            'minimum_frequency': 32.7,  # C1
            'maximum_frequency': 2093,  # C7
            'melodia_trick': True
        }
        print("Basic Pitch converter initialized")  # Keep using print for consistency

    def convert_to_midi(self, audio_path: str, output_path: str, bpm: int, sonify: bool = False, pitchbend: bool = False, onset: float = 1.0, notelength: float = 127.70, progress: Optional[callable] = None) -> str:
        try:
            print(f"Converting to MIDI: {audio_path}")  # Keep debugging output
            if progress:
                progress(0.1, "Loading audio for MIDI conversion...")
            # Predict using Basic Pitch with correct parameters
            model_output, midi_data, note_events = predict(
                audio_path=audio_path,
                onset_threshold=onset,
                frame_threshold=self.process_options['frame_threshold'],
                minimum_note_length=notelength,
                minimum_frequency=self.process_options['minimum_frequency'],
                maximum_frequency=self.process_options['maximum_frequency'],
                melodia_trick=self.process_options['melodia_trick'],
                multiple_pitch_bends=pitchbend,
                midi_tempo=bpm,
            )
            if progress:
                progress(0.7, "Saving MIDI file...")
            
            print(f"Saving MIDI to: {output_path}")  # Keep debugging output
            # Save MIDI file with validation
            if isinstance(midi_data, pretty_midi.PrettyMIDI):
                midi_data.write(output_path)
                print(f"Successfully saved MIDI to {output_path}")  # Keep using print              
                if sonify:
                    timer = librosa.get_duration(path=audio_path)
                    midi_name = output_path+".wav"
                    infer.sonify_midi(midi_data, midi_name, 44100) 
                    pygame.mixer.init()
                    pygame.mixer.music.set_volume(0.8)
                    time.sleep(1)
                    pygame.mixer.music.load(midi_name)
                    pygame.mixer.music.play()
                    time.sleep(timer)
                    time.sleep(1)
                    os.remove(midi_name)
                return output_path
            else:
                raise ValueError("MIDI conversion failed: Invalid MIDI data")
            
        except Exception as e:
            print(f"Error in MIDI conversion: {str(e)}")  # Keep using print
            raise

class DemucsProcessor:
    def __init__(self, model_name="htdemucs"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            self.model = get_model(model_name)
            print(f"Model name: {model_name}")
            print(f"Model sources: {self.model.sources}")  # This will show available stems
            print(f"Model sample rate: {self.model.samplerate}")
            
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def separate_stems(self, audio_path: str, progress=None) -> Tuple[torch.Tensor, int]:
        try:
            if progress:
                progress(0.1, "Loading audio file...")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Audio loaded - Shape: {waveform.shape}")
            
            if progress:
                progress(0.3, "Processing stems...")
            
            # Input validation and logging: Check waveform dimensions
            if waveform.dim() not in (1, 2):
                raise ValueError(f"Invalid waveform dimensions: Expected 1D or 2D, got {waveform.dim()}")

            # Handle mono input by duplicating to stereo
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
                print("Converted mono to stereo by duplication")
            
            # Ensure 3D tensor for apply_model (batch, channels, time)
            waveform = waveform.unsqueeze(0)
            print(f"Waveform shape before apply_model: {waveform.shape}")
            # Process
            with torch.no_grad():
                sources = apply_model(self.model, waveform.to(self.device))
                print(f"Sources shape after processing: {sources.shape}")
                print(f"Available stems: {self.model.sources}")
            
            if progress:
                progress(0.8, "Finalizing separation...")
            
            return sources, sample_rate
            
        except Exception as e:
            print(f"Error in stem separation: {str(e)}")
            raise

    def save_stem(self, stem: torch.Tensor, stem_name: str, output_path: str, sample_rate: int):
        try:
            torchaudio.save(
                f"{output_path}/{stem_name}.wav",
                stem.cpu(),
                sample_rate
            )
        except Exception as e:
            print(f"Error saving stem: {str(e)}")
            raise

class AudioValidator:
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 30MB
    
    @staticmethod
    def validate_audio_file(file_path: str) -> Tuple[bool, str]:
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
                
            file_size = os.path.getsize(file_path)
            if file_size > AudioValidator.MAX_FILE_SIZE:
                return False, f"File too large. Maximum size: {AudioValidator.MAX_FILE_SIZE // 1024 // 1024}MB"
                
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in AudioValidator.SUPPORTED_FORMATS:
                return False, f"Unsupported format. Supported formats: {', '.join(AudioValidator.SUPPORTED_FORMATS)}"    
            # Validate audio file integrity
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                if sample_rate < 8000 or sample_rate > 48000:
                    return False, "Invalid sample rate"
            except Exception as e:
                return False, f"Invalid audio file: {str(e)}"
                
            return True, "Valid audio file"
        except Exception as e:
            print(f"Error validating audio file: {str(e)}")
            return False, str(e)

class MidiDrumPlayer():
    # Makes new midi file according to current tempo
    def __init__(self, midi_file, bpm):
        self.midi_file = midi_file
        self.bpm = bpm
        self.tempo_change()
        print(f'current playing at tempo {self.bpm}')
        self.mid = MidiFile(self.midi_file)
        self.freq = 44100    # audio CD quality
        self.bitsize = -16   # unsigned 16 bit
        self.channels = 1    # 1 is mono, 2 is stereo
        self.buffer = 1024    # number of samples

        pygame.mixer.init(self.freq, self.bitsize, self.channels, self.buffer)
        self.mixer = pygame.mixer
        self.mixer.music.set_volume(1)

        self.kick = self.mixer.Sound(resource_path(f"resources/drumkit/kick.wav"))
        self.snare = self.mixer.Sound(resource_path(f"resources/drumkit/snare.wav"))
        self.hh = self.mixer.Sound(resource_path(f"resources/drumkit/hihat.wav"))

        self.playstate = False
        self.loopstate = False

    def playonce(self):
        self.playstate = True

        for msg in self.mid.play():
            if self.playstate == False:
                break
            else:
                if msg.bytes()[0] == 144:
                    if msg.bytes()[1] == 36:
                        self.kick.play()
                    elif msg.bytes()[1] == 38: 
                        self.snare.play()
                    elif msg.bytes()[1] == 42: 
                        self.hh.play()
                    else:        
                        pass


    def playmidi(self):

        self.playstate = True
        while self.playstate:
            for msg in self.mid.play():
                current_note = msg.bytes()[1]
                if self.playstate == False:
                    break
                else:
                    if msg.bytes()[0] == 144:
                        if msg.bytes()[1] == 36:
                            self.kick.play()
                            print('kick')
                        elif msg.bytes()[1] == 38: 
                            self.snare.play()
                            print('snare')
                        elif msg.bytes()[1] == 42: 
                            self.hh.play()
                            print('hihat')
                        else:        
                            pass

    def tempo_change(self):
        mid = MidiFile(self.midi_file)
        mid_new = MidiFile()
        track = MidiTrack()
        mid_new.tracks.append(track)

        bpm = self.bpm
        tempo_new = mido.bpm2tempo(bpm)
        track.append(MetaMessage('set_tempo', tempo=tempo_new))

        for t in mid.tracks:
            for msg in t:
                if msg.type == 'note_on':
                    track.append(Message('note_on', note=msg.note, velocity=msg.note, time=msg.time))
                elif msg.type == 'note_off':
                    track.append(Message('note_off', note=msg.note, time=msg.time))

        mid_new.save(self.midi_file)

    def stopMidi(self):
        self.playstate = False

if __name__ == "__main__":
    main()