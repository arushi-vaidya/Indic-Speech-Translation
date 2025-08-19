import os, time, wave, threading, tkinter, json, shutil
from pathlib import Path

# Audio processing
import torch, whisper, pyaudio, soundfile, librosa, webrtcvad, numpy
from scipy.ndimage import uniform_filter1d
from pydub import AudioSegment

# Translation dependencies  
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor  # ← CRITICAL DEPENDENCY

# TTS dependencies
from TTS.utils.synthesizer import Synthesizer

import os
import time
import wave
import torch
import whisper
import pyaudio
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import librosa
import webrtcvad
from scipy.ndimage import uniform_filter1d
import queue
import shutil
from pathlib import Path
import json

# Force environment variables to prevent cache issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['FORCE_CPU'] = '1'
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache', 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.cache', 'transformers')

# Create cache directories
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

# Disable transformers warnings and cache warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Store original functions before patching
original_cuda_is_available = torch.cuda.is_available
original_cuda_device_count = torch.cuda.device_count

def fake_cuda_available():
    return False

def fake_cuda_device_count():
    return 0

# Apply minimal patches
torch.cuda.is_available = fake_cuda_available
torch.cuda.device_count = fake_cuda_device_count

# Patch tensor .cuda() and .to() methods
original_tensor_cuda = torch.Tensor.cuda
original_tensor_to = torch.Tensor.to

def patched_tensor_cuda(self, device=None, non_blocking=False):
    return self.cpu()

def patched_tensor_to(self, *args, **kwargs):
    if args:
        device = args[0]
        if isinstance(device, str) and 'cuda' in device.lower():
            return original_tensor_to(self, 'cpu', *args[1:], **kwargs)
        elif hasattr(device, 'type') and 'cuda' in str(device.type):
            return original_tensor_to(self, 'cpu', *args[1:], **kwargs)
    
    if 'device' in kwargs:
        device = kwargs['device']
        if isinstance(device, str) and 'cuda' in device.lower():
            kwargs['device'] = 'cpu'
        elif hasattr(device, 'type') and 'cuda' in str(device.type):
            kwargs['device'] = 'cpu'
    
    return original_tensor_to(self, *args, **kwargs)

# Apply tensor patches
torch.Tensor.cuda = patched_tensor_cuda
torch.Tensor.to = patched_tensor_to

print("✅ Cache and TTS patches applied - CPU-only mode enabled")


class FastKannadaTTS:
    """Integrated Fast Kannada TTS for the pipeline with enhanced error handling"""
    def __init__(self):
        self.setup_paths()
        self.synthesizer = None
        self.is_initialized = False
        
    def setup_paths(self):
        """Setup directory paths for models"""
        self.base_dir = Path.cwd()
        self.kn_dir = self.base_dir / 'kn'
        self.cpu_dir = self.base_dir / 'kn_cpu_patched'
        self.mac_opt_dir = self.base_dir / 'kannada_mac_optimized'
        
        # Priority order: optimized -> cpu patched -> original
        self.model_paths = self.find_best_models()
        
    def find_best_models(self):
        """Find the best available model set"""
        # Check for Mac optimized models first
        if self.mac_opt_dir.exists():
            fastpitch_model = self.mac_opt_dir / 'fastpitch_mac_optimized.pth'
            fastpitch_config = self.mac_opt_dir / 'fastpitch_config.json'
            hifigan_model = self.mac_opt_dir / 'hifigan_mac_optimized.pth'
            hifigan_config = self.mac_opt_dir / 'hifigan_config.json'
            
            if all(p.exists() for p in [fastpitch_model, fastpitch_config, hifigan_model, hifigan_config]):
                return {
                    'fastpitch_model': str(fastpitch_model),
                    'fastpitch_config': str(fastpitch_config),
                    'hifigan_model': str(hifigan_model),
                    'hifigan_config': str(hifigan_config),
                    'type': 'mac_optimized'
                }
        
        # Check for CPU patched models
        if self.cpu_dir.exists():
            fastpitch_model = self.cpu_dir / 'fastpitch' / 'best_model.pth'
            fastpitch_config = self.cpu_dir / 'fastpitch' / 'config.json'
            hifigan_model = self.cpu_dir / 'hifigan' / 'best_model.pth'
            hifigan_config = self.cpu_dir / 'hifigan' / 'config.json'
            
            if all(p.exists() for p in [fastpitch_model, fastpitch_config, hifigan_model, hifigan_config]):
                return {
                    'fastpitch_model': str(fastpitch_model),
                    'fastpitch_config': str(fastpitch_config),
                    'hifigan_model': str(hifigan_model),
                    'hifigan_config': str(hifigan_config),
                    'type': 'cpu_patched'
                }
        
        # Fall back to original models
        if self.kn_dir.exists():
            fastpitch_model = self.kn_dir / 'fastpitch' / 'best_model.pth'
            fastpitch_config = self.kn_dir / 'fastpitch' / 'config.json'
            hifigan_model = self.kn_dir / 'hifigan' / 'best_model.pth'
            hifigan_config = self.kn_dir / 'hifigan' / 'config.json'
            
            if all(p.exists() for p in [fastpitch_model, fastpitch_config, hifigan_model, hifigan_config]):
                return {
                    'fastpitch_model': str(fastpitch_model),
                    'fastpitch_config': str(fastpitch_config),
                    'hifigan_model': str(hifigan_model),
                    'hifigan_config': str(hifigan_config),
                    'type': 'original'
                }
        
        # Check models/v1/kn directory (from your project structure)
        models_dir = self.base_dir / 'models' / 'v1' / 'kn'
        if models_dir.exists():
            fastpitch_model = models_dir / 'fastpitch' / 'best_model.pth'
            fastpitch_config = models_dir / 'fastpitch' / 'config.json'
            hifigan_model = models_dir / 'hifigan' / 'best_model.pth'
            hifigan_config = models_dir / 'hifigan' / 'config.json'
            
            if all(p.exists() for p in [fastpitch_model, fastpitch_config, hifigan_model, hifigan_config]):
                return {
                    'fastpitch_model': str(fastpitch_model),
                    'fastpitch_config': str(fastpitch_config),
                    'hifigan_model': str(hifigan_model),
                    'hifigan_config': str(hifigan_config),
                    'type': 'models_v1'
                }
        
        return None
    
    def initialize(self):
        """Initialize the TTS synthesizer with enhanced error handling"""
        if self.is_initialized:
            return True
            
        if not self.model_paths:
            print("❌ No Kannada TTS models found!")
            return False
        
        try:
            print(f"🎯 Initializing Fast Kannada TTS using {self.model_paths['type']} models...")
            
            # Import TTS after patches are applied
            from TTS.utils.synthesizer import Synthesizer
            
            # CRITICAL FIX: Initialize with explicit CPU device and disabled cache
            self.synthesizer = Synthesizer(
                tts_checkpoint=self.model_paths['fastpitch_model'],
                tts_config_path=self.model_paths['fastpitch_config'],
                vocoder_checkpoint=self.model_paths['hifigan_model'],
                vocoder_config=self.model_paths['hifigan_config'],
                use_cuda=False
            )
            
            # Force all models to CPU and disable any CUDA references
            if hasattr(self.synthesizer, 'tts_model') and self.synthesizer.tts_model:
                self.synthesizer.tts_model = self.synthesizer.tts_model.cpu()
                self.synthesizer.tts_model.eval()
                
            if hasattr(self.synthesizer, 'vocoder_model') and self.synthesizer.vocoder_model:
                self.synthesizer.vocoder_model = self.synthesizer.vocoder_model.cpu()
                self.synthesizer.vocoder_model.eval()
            
            self.is_initialized = True
            print(f"✅ Fast Kannada TTS initialized successfully with {self.model_paths['type']} models!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Fast Kannada TTS: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return False
    
    def synthesize(self, text, output_path):
        """Synthesize Kannada text to speech with comprehensive error handling"""
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        try:
            print(f"🎵 Synthesizing: {text}")
            start_time = time.time()
            
            # Get speaker information from the synthesizer
            speakers = self.get_available_speakers()
            
            # CRITICAL FIX: Use multiple synthesis strategies with explicit cache control
            wav = None
            synthesis_attempts = [
                # Attempt 1: Basic synthesis without speaker (most likely to work)
                lambda: self.safe_synthesis(text, None),
                # Attempt 2: Use speaker_idx=0 if multi-speaker
                lambda: self.safe_synthesis(text, {'speaker_idx': 0}),
                # Attempt 3: Use first speaker name if available
                lambda: self.safe_synthesis(text, {'speaker_name': speakers[0]}) if speakers else None,
                # Attempt 4: Empty speaker_wav
                lambda: self.safe_synthesis(text, {'speaker_wav': ""}),
                # Attempt 5: Force single speaker mode
                lambda: self.force_single_speaker_synthesis(text)
            ]
            
            for i, attempt in enumerate(synthesis_attempts):
                try:
                    if attempt is None:
                        continue
                    wav = attempt()
                    if wav is not None and len(wav) > 100:  # Valid audio
                        print(f"✅ Synthesis successful with method {i+1}")
                        break
                except Exception as e:
                    print(f"⚠️ Synthesis method {i+1} failed: {e}")
                    continue
            
            if wav is None or len(wav) < 100:
                print("❌ All synthesis methods failed")
                return False
            
            # Save audio with error handling
            try:
                self.synthesizer.save_wav(wav, output_path)
            except Exception as e:
                print(f"⚠️ Save_wav failed, trying direct save: {e}")
                # Fallback: save directly using soundfile
                import soundfile as sf
                sample_rate = getattr(self.synthesizer, 'output_sample_rate', 22050)
                if hasattr(self.synthesizer, 'ap') and hasattr(self.synthesizer.ap, 'sample_rate'):
                    sample_rate = self.synthesizer.ap.sample_rate
                sf.write(output_path, wav, sample_rate)
            
            synthesis_time = time.time() - start_time
            
            # Verify output
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✅ Fast TTS synthesis completed in {synthesis_time:.2f} seconds!")
                return True
            else:
                print("❌ Output file was not created or is empty")
                return False
                
        except Exception as e:
            print(f"❌ Fast TTS synthesis failed: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return False
    
    def safe_synthesis(self, text, speaker_config):
        """Safe synthesis with cache and error handling"""
        try:
            # Disable any cache mechanisms that might cause issues
            with torch.no_grad():
                if speaker_config is None:
                    # Basic synthesis
                    wav = self.synthesizer.tts(text)
                elif 'speaker_idx' in speaker_config:
                    wav = self.synthesizer.tts(text, speaker_idx=speaker_config['speaker_idx'])
                elif 'speaker_name' in speaker_config:
                    wav = self.synthesizer.tts(text, speaker_name=speaker_config['speaker_name'])
                elif 'speaker_wav' in speaker_config:
                    wav = self.synthesizer.tts(text, speaker_wav=speaker_config['speaker_wav'])
                else:
                    wav = self.synthesizer.tts(text)
                
                return wav
                
        except Exception as e:
            print(f"Safe synthesis error: {e}")
            return None
    
    def force_single_speaker_synthesis(self, text):
        """Force single speaker mode to avoid cache issues"""
        try:
            # Try to force the model into single-speaker mode
            original_num_speakers = None
            if hasattr(self.synthesizer.tts_model, 'num_speakers'):
                original_num_speakers = self.synthesizer.tts_model.num_speakers
                self.synthesizer.tts_model.num_speakers = 0
            
            wav = self.synthesizer.tts(text)
            
            # Restore original setting
            if original_num_speakers is not None:
                self.synthesizer.tts_model.num_speakers = original_num_speakers
            
            return wav
            
        except Exception as e:
            print(f"Force single speaker error: {e}")
            return None

    def get_available_speakers(self):
        """Get available speakers from the model with better error handling"""
        try:
            if hasattr(self.synthesizer, 'tts_model') and self.synthesizer.tts_model:
                if hasattr(self.synthesizer.tts_model, 'speaker_manager') and self.synthesizer.tts_model.speaker_manager:
                    speakers = self.synthesizer.tts_model.speaker_manager.speaker_names
                    if speakers:
                        print(f"🎤 Available speakers: {speakers}")
                        return speakers
                    else:
                        print("🎤 Speaker manager exists but no speaker names found")
                
                # Try to get speaker count from config
                if hasattr(self.synthesizer, 'tts_config'):
                    if hasattr(self.synthesizer.tts_config, 'num_speakers'):
                        num_speakers = self.synthesizer.tts_config.num_speakers
                        if num_speakers and num_speakers > 1:
                            print(f"🎤 Model has {num_speakers} speakers (will use index 0)")
                            return list(range(num_speakers))
                    elif hasattr(self.synthesizer.tts_config, 'model_args'):
                        model_args = self.synthesizer.tts_config.model_args
                        if hasattr(model_args, 'num_speakers'):
                            num_speakers = model_args.num_speakers
                            if num_speakers and num_speakers > 1:
                                print(f"🎤 Model has {num_speakers} speakers (will use index 0)")
                                return list(range(num_speakers))
            
            print("🎤 No specific speaker information found, will try default methods")
            return None
            
        except Exception as e:
            print(f"🎤 Error getting speakers: {e}")
            return None


class AudioEnhancer:
    """Real-time audio enhancement for noise reduction"""
    def __init__(self, samplerate=16000, frame_length=2048, hop_length=512, 
                 alpha=0.98, noise_threshold=1.2, vad_aggressiveness=2):
        self.samplerate = samplerate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.alpha = alpha
        self.noise_threshold = noise_threshold
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.vad_frame_size = int(0.02 * samplerate)  # 20ms for VAD
        
        # Noise profile for enhancement
        self.noise_profile = np.zeros((frame_length // 2 + 1, 1))
        self.noise_frames_count = 0
        self.speech_frames_count = 0
        
    def spectral_subtraction(self, magnitude, noise_profile):
        """Apply spectral subtraction for noise reduction"""
        if np.any(noise_profile > 0):
            # Compute spectral subtraction mask
            mask = 1 - (self.noise_threshold * noise_profile) / (magnitude + 1e-8)
            mask = np.maximum(mask, 0.1)  # Minimum mask to prevent artifacts
            return mask
        return np.ones_like(magnitude)
    
    def enhance_audio_chunk(self, audio_chunk):
        """Enhance a chunk of audio using spectral subtraction"""
        if len(audio_chunk) < self.frame_length:
            return audio_chunk
            
        # STFT analysis
        stft = librosa.stft(audio_chunk, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude, phase = np.abs(stft), np.angle(stft)
        avg_magnitude = np.mean(magnitude, axis=1, keepdims=True)
        
        # VAD on first 20ms if available
        if len(audio_chunk) >= self.vad_frame_size:
            vad_frame = audio_chunk[:self.vad_frame_size]
            vad_frame_bytes = (vad_frame * 32767).astype(np.int16).tobytes()
            
            try:
                is_speech = self.vad.is_speech(vad_frame_bytes, self.samplerate)
            except:
                is_speech = True  # Default to speech if VAD fails
        else:
            is_speech = True
        
        # Update noise profile during non-speech segments
        if not is_speech:
            self.noise_profile = (self.alpha * self.noise_profile + 
                                (1 - self.alpha) * avg_magnitude)
            self.noise_frames_count += 1
        else:
            self.speech_frames_count += 1
        
        # Apply spectral subtraction
        mask = self.spectral_subtraction(magnitude, self.noise_profile)
        enhanced_magnitude = magnitude * mask
        
        # Smooth to reduce artifacts
        enhanced_magnitude = uniform_filter1d(enhanced_magnitude, size=3, axis=1)
        
        # ISTFT synthesis
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def enhance_full_audio(self, audio_data):
        """Enhance complete audio file"""
        if len(audio_data) == 0:
            return audio_data
            
        # Process in overlapping chunks for better quality
        chunk_size = self.frame_length * 4  # Larger chunks for file processing
        overlap = self.frame_length // 2
        enhanced_chunks = []
        
        for start in range(0, len(audio_data) - overlap, chunk_size - overlap):
            end = min(start + chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            
            if len(chunk) > self.frame_length // 2:  # Only process meaningful chunks
                enhanced_chunk = self.enhance_audio_chunk(chunk)
                enhanced_chunks.append(enhanced_chunk)
        
        if enhanced_chunks:
            # Overlap-add reconstruction
            enhanced_audio = enhanced_chunks[0]
            for i, chunk in enumerate(enhanced_chunks[1:], 1):
                # Simple overlap-add (can be improved with proper windowing)
                start_pos = (chunk_size - overlap) * i
                if start_pos < len(enhanced_audio):
                    # Overlap region
                    overlap_len = min(overlap, len(enhanced_audio) - start_pos, len(chunk))
                    if overlap_len > 0:
                        enhanced_audio[start_pos:start_pos + overlap_len] *= 0.5
                        enhanced_audio[start_pos:start_pos + overlap_len] += chunk[:overlap_len] * 0.5
                    
                    # Non-overlap region
                    if len(chunk) > overlap_len:
                        enhanced_audio = np.concatenate([
                            enhanced_audio,
                            chunk[overlap_len:]
                        ])
                else:
                    enhanced_audio = np.concatenate([enhanced_audio, chunk])
            
            return enhanced_audio
        
        return audio_data


class SpeechTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced English to Kannada Translator with Fast TTS - Fixed")
        self.root.geometry("650x700")  
        self.root.resizable(True, True)
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12))
        self.style.configure("TLabel", font=("Arial", 12))
        
        # Create app directory
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_dir = os.path.join(self.app_dir, "audio_files")
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Modified language mapping - prioritize Kannada with fast TTS
        self.language_mapping = {
            "Kannada (Fast TTS)": {
                "code": "kan_Knda",
                "speakers": ["Default", "Speaker 1", "Speaker 2"],
                "default_speaker": "Default",
                "use_fast_tts": True
            },
            "Telugu": {
                "code": "tel_Telu",
                "speakers": ["Prakash", "Lalitha", "Kiran"],
                "default_speaker": "Prakash",
                "use_fast_tts": False
            },
            "Hindi": {
                "code": "hin_Deva",
                "speakers": ["Ravi", "Priya", "Amit"],
                "default_speaker": "Ravi",
                "use_fast_tts": False
            },
            "Tamil": {
                "code": "tam_Taml",
                "speakers": ["Arun", "Meena"],
                "default_speaker": "Arun",
                "use_fast_tts": False
            },
            "Gujarati": {
                "code": "guj_Gujr",
                "speakers": ["Jignesh", "Kavita"],
                "default_speaker": "Jignesh",
                "use_fast_tts": False
            },
            "Bengali": {
                "code": "ben_Beng",
                "speakers": ["Rahul", "Mou"],
                "default_speaker": "Rahul",
                "use_fast_tts": False
            }
        }
        
        # Recording variables
        self.is_recording = False
        self.recorded_audio = []
        self.audio_thread = None
        self.temp_wav_file = os.path.join(self.audio_dir, "temp_recording.wav")
        self.enhanced_wav_file = os.path.join(self.audio_dir, "enhanced_recording.wav")
        self.output_file = os.path.join(self.audio_dir, "output.wav")  # Changed to .wav for fast TTS
        
        # Initialize audio enhancer
        self.audio_enhancer = AudioEnhancer(
            samplerate=16000,
            noise_threshold=1.5,
            vad_aggressiveness=2
        )
        
        # Initialize Fast Kannada TTS
        self.fast_kannada_tts = FastKannadaTTS()
        
        # Audio enhancement settings
        self.enhancement_enabled = tk.BooleanVar(value=True)
        self.noise_threshold_var = tk.DoubleVar(value=1.5)
        self.auto_play_enabled = tk.BooleanVar(value=True)
        
        # Initialize models
        self.load_models_thread = threading.Thread(target=self.load_models)
        self.load_models_thread.daemon = True
        self.load_models_thread.start()
        
        # Create UI
        self.create_widgets()
    
    def load_models(self):
        """Load all models in a background thread with enhanced error handling"""
        try:
            # Update status
            self.update_status("Loading models with cache fixes, please wait...")
            
            # Device configuration
            self.DEVICE = "cpu"  # Force CPU to avoid cache issues
            
            # Load Whisper model
            self.update_status("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            
            # Load translation model with explicit cache control
            self.update_status("Loading IndicTrans model with cache fixes...")
            model_name = "ai4bharat/indictrans2-en-indic-1B"
            
            # CRITICAL FIX: Load with explicit cache control and CPU forcing
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                local_files_only=False,
                force_download=False
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                torch_dtype=torch.float32,  # Use float32 for stability
                local_files_only=False,
                force_download=False
            ).to(self.DEVICE)
            
            # Force model to eval mode and CPU
            self.model.eval()
            
            self.ip = IndicProcessor(inference=True)
            
            # Initialize Fast Kannada TTS
            self.update_status("Initializing Fast Kannada TTS with fixes...")
            fast_tts_success = self.fast_kannada_tts.initialize()
            
            # Load fallback TTS model only if needed for other languages
            self.update_status("Loading fallback TTS model...")
            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                    "ai4bharat/indic-parler-tts",
                    cache_dir=os.environ['TRANSFORMERS_CACHE'],
                    torch_dtype=torch.float32
                ).to(self.DEVICE)
                self.tts_tokenizer = AutoTokenizer.from_pretrained(
                    "ai4bharat/indic-parler-tts",
                    cache_dir=os.environ['TRANSFORMERS_CACHE']
                )
                self.description_tokenizer = AutoTokenizer.from_pretrained(
                    self.tts_model.config.text_encoder._name_or_path,
                    cache_dir=os.environ['TRANSFORMERS_CACHE']
                )
                self.tts_model.eval()
                fallback_tts_loaded = True
            except Exception as e:
                print(f"⚠️ Fallback TTS failed to load: {e}")
                fallback_tts_loaded = False
            
            # Update status based on what loaded
            if fast_tts_success and fallback_tts_loaded:
                self.update_status("✅ All models loaded with cache fixes! Fast TTS ready for Kannada.")
            elif fast_tts_success:
                self.update_status("✅ Fast Kannada TTS loaded with fixes! Other languages may not be available.")
            elif fallback_tts_loaded:
                self.update_status("✅ Fallback TTS loaded. Fast TTS not available - check Kannada models.")
            else:
                self.update_status("⚠️ Limited functionality - some TTS models failed to load")
            
            self.record_button.config(state=tk.NORMAL)
            
        except Exception as e:
            error_msg = f"Error loading models: {str(e)}"
            print(f"Model loading error: {e}")
            import traceback
            traceback.print_exc()
            self.update_status(error_msg)
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Enhanced English to Kannada Translator - Cache Fixed", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        self.status_label = ttk.Label(main_frame, text="Loading models with cache fixes...", foreground="blue")
        self.status_label.pack(pady=10)
        
        # Fast TTS Status Frame
        tts_status_frame = ttk.LabelFrame(main_frame, text="TTS Status & Cache Info", padding=10)
        tts_status_frame.pack(fill=tk.X, pady=5)
        
        self.tts_status_label = ttk.Label(tts_status_frame, text="Checking Fast TTS availability...", foreground="orange")
        self.tts_status_label.pack(anchor=tk.W)
        
        self.cache_info_label = ttk.Label(tts_status_frame, text=f"Cache: {os.environ['TRANSFORMERS_CACHE']}", 
                                         foreground="gray", font=("Arial", 9))
        self.cache_info_label.pack(anchor=tk.W)
        
        # Update TTS status
        def update_tts_status():
            if self.fast_kannada_tts.model_paths:
                model_type = self.fast_kannada_tts.model_paths['type']
                self.tts_status_label.config(text=f"✅ Fast TTS Ready ({model_type} models found)", foreground="green")
            else:
                self.tts_status_label.config(text="⚠️ Fast TTS models not found - run setup first", foreground="red")
        
        self.root.after(2000, update_tts_status)  # Check status after 2 seconds
        
        # Audio Enhancement Settings Frame
        enhancement_frame = ttk.LabelFrame(main_frame, text="Audio Enhancement Settings", padding=10)
        enhancement_frame.pack(fill=tk.X, pady=5)
        
        enhancement_check = ttk.Checkbutton(enhancement_frame, text="Enable Audio Enhancement", 
                                          variable=self.enhancement_enabled)
        enhancement_check.pack(anchor=tk.W)
        
        threshold_frame = ttk.Frame(enhancement_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="Noise Reduction Level:").pack(side=tk.LEFT)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=3.0, 
                                  variable=self.noise_threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.threshold_label = ttk.Label(threshold_frame, text="1.5")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        threshold_scale.bind("<Motion>", self.update_threshold_label)
        
        auto_play_check = ttk.Checkbutton(enhancement_frame, text="Auto-play after translation", 
                                        variable=self.auto_play_enabled)
        auto_play_check.pack(anchor=tk.W, pady=5)
        
        # Language Selection Frame
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lang_frame, text="Target Language:").pack(side=tk.LEFT, padx=5)
        
        self.language_var = tk.StringVar(value="Kannada (Fast TTS)")
        languages = list(self.language_mapping.keys())
        language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, values=languages, width=20)
        language_combo.pack(side=tk.LEFT, padx=5)
        language_combo.bind("<<ComboboxSelected>>", self.on_language_change)
        
        # Recording Button
        self.record_button = ttk.Button(main_frame, text="Start Recording", 
                                       command=self.toggle_recording, state=tk.DISABLED)
        self.record_button.pack(pady=15)
        
        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)
        
        # Results Frame
        text_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(text_frame, text="English:").pack(anchor=tk.W)
        self.english_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.english_text.pack(fill=tk.X, pady=5)
        
        self.translated_label = ttk.Label(text_frame, text="Kannada:")
        self.translated_label.pack(anchor=tk.W)
        self.translated_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.translated_text.pack(fill=tk.X, pady=5)
        
        # Speaker Selection Frame
        speaker_frame = ttk.Frame(main_frame)
        speaker_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speaker_frame, text="Speaker:").pack(side=tk.LEFT, padx=5)
        self.speaker_var = tk.StringVar(value=self.language_mapping["Kannada (Fast TTS)"]["default_speaker"])
        self.speaker_combo = ttk.Combobox(speaker_frame, textvariable=self.speaker_var, 
                                         values=self.language_mapping["Kannada (Fast TTS)"]["speakers"], width=15)
        self.speaker_combo.pack(side=tk.LEFT, padx=5)
        
        # Play Button
        self.play_button = ttk.Button(main_frame, text="Play Translation", 
                                     command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(pady=10)
        
        # Audio Quality Info
        self.quality_label = ttk.Label(main_frame, text="", foreground="green", font=("Arial", 10))
        self.quality_label.pack(pady=5)
        
        # Performance Info
        self.performance_label = ttk.Label(main_frame, text="", foreground="blue", font=("Arial", 10))
        self.performance_label.pack(pady=2)
        
        # Error Info (new)
        self.error_label = ttk.Label(main_frame, text="", foreground="red", font=("Arial", 10))
        self.error_label.pack(pady=2)
    
    def update_threshold_label(self, event=None):
        """Update the threshold label with current value"""
        value = self.noise_threshold_var.get()
        self.threshold_label.config(text=f"{value:.1f}")
        self.audio_enhancer.noise_threshold = value
    
    def on_language_change(self, event=None):
        selected_language = self.language_var.get()
        if selected_language in self.language_mapping:
            speakers = self.language_mapping[selected_language]["speakers"]
            default_speaker = self.language_mapping[selected_language]["default_speaker"]
            
            self.speaker_combo['values'] = speakers
            self.speaker_var.set(default_speaker)
            
            # Update label to show language name without "(Fast TTS)"
            display_name = selected_language.replace(" (Fast TTS)", "")
            self.translated_label.config(text=f"{display_name}:")
    
    def update_status(self, message):
        def _update():
            self.status_label.config(text=message)
        
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)
    
    def update_quality_info(self, message):
        """Update audio quality information"""
        def _update():
            self.quality_label.config(text=message)
        
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)
    
    def update_performance_info(self, message):
        """Update performance information"""
        def _update():
            self.performance_label.config(text=message)
        
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)
    
    def update_error_info(self, message):
        """Update error information"""
        def _update():
            self.error_label.config(text=message)
        
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording with enhanced audio processing"""
        self.is_recording = True
        self.recorded_audio = []
        self.record_button.config(text="Stop Recording")
        self.update_status("Recording... Speak now")
        self.update_quality_info("")
        self.update_performance_info("")
        self.update_error_info("")
        
        # Reset audio enhancer state
        self.audio_enhancer.noise_profile = np.zeros((self.audio_enhancer.frame_length // 2 + 1, 1))
        self.audio_enhancer.noise_frames_count = 0
        self.audio_enhancer.speech_frames_count = 0
        
        self.english_text.delete(1.0, tk.END)
        self.translated_text.delete(1.0, tk.END)
        self.play_button.config(state=tk.DISABLED)
        
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            self.update_status("Processing audio...")
            
            # Wait for recording thread to finish
            if self.audio_thread:
                self.audio_thread.join()
            
            processing_thread = threading.Thread(target=self.process_recording)
            processing_thread.daemon = True
            processing_thread.start()
    
    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        try:
            p = pyaudio.PyAudio()
            info = f"Audio devices available:\n"
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                info += f"Device {i}: {dev_info['name']}, Inputs: {dev_info['maxInputChannels']}\n"
            print(info)
            
            default_input_device_index = p.get_default_input_device_info()['index']
            print(f"Using default input device: {default_input_device_index}")
            
            stream = p.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           input=True,
                           input_device_index=default_input_device_index,
                           frames_per_buffer=CHUNK)
            
            self.update_status("Recording... Speak now")
            
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.recorded_audio.append(data)
                except Exception as e:
                    print(f"Error during recording: {e}")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if len(self.recorded_audio) > 0:
                print(f"Saving {len(self.recorded_audio)} audio chunks to {self.temp_wav_file}")
                
                os.makedirs(os.path.dirname(self.temp_wav_file), exist_ok=True)
                
                wf = wave.open(self.temp_wav_file, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.recorded_audio))
                wf.close()
                
                if os.path.exists(self.temp_wav_file):
                    file_size = os.path.getsize(self.temp_wav_file)
                    print(f"Recording saved successfully. File size: {file_size} bytes")
                    if file_size < 100:
                        print("WARNING: Audio file is very small. Microphone might not be capturing audio.")
                else:
                    print(f"ERROR: Failed to create audio file at {self.temp_wav_file}")
            else:
                print("ERROR: No audio data recorded!")
                self.update_status("No audio recorded. Please check your microphone.")
        
        except Exception as e:
            print(f"Critical error in recording: {str(e)}")
            self.update_status(f"Recording error: {str(e)}")
    
    def enhance_recorded_audio(self):
        """Apply audio enhancement to the recorded file"""
        try:
            if not os.path.exists(self.temp_wav_file):
                return False
                
            # Load the recorded audio
            audio_data, sr = librosa.load(self.temp_wav_file, sr=16000)
            
            if len(audio_data) == 0:
                return False
            
            # Apply enhancement if enabled
            if self.enhancement_enabled.get():
                self.update_status("Enhancing audio quality...")
                enhanced_audio = self.audio_enhancer.enhance_full_audio(audio_data)
                
                # Normalize enhanced audio
                if np.max(np.abs(enhanced_audio)) > 0:
                    enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
                
                # Save enhanced audio
                sf.write(self.enhanced_wav_file, enhanced_audio, sr)
                
                # Update quality info
                speech_ratio = (self.audio_enhancer.speech_frames_count / 
                              max(1, self.audio_enhancer.speech_frames_count + self.audio_enhancer.noise_frames_count))
                quality_info = f"Enhancement applied: {speech_ratio:.1%} speech detected, noise reduced"
                self.update_quality_info(quality_info)
                
                return True
            else:
                # Copy original file if enhancement is disabled
                import shutil
                shutil.copy2(self.temp_wav_file, self.enhanced_wav_file)
                self.update_quality_info("Enhancement disabled - using original audio")
                return True
                
        except Exception as e:
            print(f"Audio enhancement error: {e}")
            self.update_quality_info(f"Enhancement failed: {str(e)}")
            # Copy original file as fallback
            try:
                import shutil
                shutil.copy2(self.temp_wav_file, self.enhanced_wav_file)
                return True
            except:
                return False
    
    def process_recording(self):
        """Process the recorded audio through the translation pipeline with enhanced error handling"""
        pipeline_start_time = time.time()
        
        try:
            # Check if recording file exists and has content
            if not os.path.exists(self.temp_wav_file) or os.path.getsize(self.temp_wav_file) < 100:
                self.update_status("Error: Recording file is empty or too small")
                self.update_error_info("No audio was recorded. Please check your microphone settings.")
                messagebox.showerror("Error", "No audio was recorded. Please check your microphone settings.")
                return
            
            # Start progress bar
            self.root.after(0, self.progress.start)
            
            # Enhance audio quality
            enhance_start = time.time()
            if not self.enhance_recorded_audio():
                self.update_status("Error: Failed to process audio")
                self.update_error_info("Audio enhancement failed")
                self.root.after(0, self.progress.stop)
                return
            enhance_time = time.time() - enhance_start
            
            # Use enhanced audio file for transcription
            audio_file_to_use = self.enhanced_wav_file if os.path.exists(self.enhanced_wav_file) else self.temp_wav_file
            
            # Get the selected language info
            selected_language = self.language_var.get()
            lang_info = self.language_mapping[selected_language]
            target_lang_code = lang_info["code"]
            use_fast_tts = lang_info.get("use_fast_tts", False)
            
            # Step 1: Transcribe using enhanced audio
            transcribe_start = time.time()
            self.update_status("Transcribing enhanced audio...")
            result = self.whisper_model.transcribe(audio_file_to_use)
            english_text = result["text"]
            transcribe_time = time.time() - transcribe_start
            
            if not english_text.strip():
                self.update_status("Error: Could not transcribe any text")
                self.update_error_info("No speech detected in the recording.")
                messagebox.showerror("Error", "No speech detected in the recording. Please try again.")
                self.root.after(0, self.progress.stop)
                return
            
            def update_english():
                self.english_text.delete(1.0, tk.END)
                self.english_text.insert(tk.END, english_text)
            self.root.after(0, update_english)
            
            # Step 2: Translate with cache fixes
            translate_start = time.time()
            display_name = selected_language.replace(" (Fast TTS)", "")
            self.update_status(f"Translating to {display_name}...")
            
            try:
                batch = self.ip.preprocess_batch([english_text], src_lang="eng_Latn", tgt_lang=target_lang_code)
                inputs = self.tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.DEVICE)
                
                # CRITICAL FIX: Use no_grad and disable cache explicitly
                with torch.no_grad():
                    # Clear any existing cache
                    if hasattr(self.model, 'generation_config'):
                        self.model.generation_config.use_cache = False
                    
                    output_tokens = self.model.generate(
                        **inputs,
                        use_cache=False,  # CRITICAL: Disable cache to prevent layer access error
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                with self.tokenizer.as_target_tokenizer():
                    decoded = self.tokenizer.batch_decode(
                        output_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                
                translated_text = self.ip.postprocess_batch(decoded, lang=target_lang_code)[0]
                translate_time = time.time() - translate_start
                
                def update_translated():
                    self.translated_text.delete(1.0, tk.END)
                    self.translated_text.insert(tk.END, translated_text)
                self.root.after(0, update_translated)
                
            except Exception as e:
                self.update_error_info(f"Translation failed: {str(e)}")
                print(f"Translation error: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, self.progress.stop)
                return
            
            # Step 3: Text to Speech (Fast TTS for Kannada, fallback for others)
            tts_start = time.time()
            if use_fast_tts and self.fast_kannada_tts.is_initialized:
                self.update_status(f"Generating {display_name} speech with Fast TTS...")
                
                # Use Fast Kannada TTS
                success = self.fast_kannada_tts.synthesize(translated_text, self.output_file)
                
                if not success:
                    self.update_status("Fast TTS failed, trying fallback...")
                    self.update_error_info("Fast TTS failed, attempting fallback")
                    success = self.fallback_tts(translated_text, selected_language)
            else:
                # Use fallback TTS for other languages
                self.update_status(f"Generating {display_name} speech...")
                success = self.fallback_tts(translated_text, selected_language)
            
            tts_time = time.time() - tts_start
            total_time = time.time() - pipeline_start_time
            
            # Update performance info
            perf_info = f"⚡ Times: Enhance={enhance_time:.1f}s, Transcribe={transcribe_time:.1f}s, Translate={translate_time:.1f}s, TTS={tts_time:.1f}s, Total={total_time:.1f}s"
            self.update_performance_info(perf_info)
            
            if success:
                # Enable play button
                def enable_play():
                    self.play_button.config(state=tk.NORMAL)
                    self.progress.stop()
                    if use_fast_tts:
                        self.update_status(f"✅ Fast TTS complete! Total time: {total_time:.1f}s")
                    else:
                        self.update_status(f"✅ Translation complete! Total time: {total_time:.1f}s")
                    
                    if self.auto_play_enabled.get():
                        self.root.after(100, self.play_audio)
                self.root.after(0, enable_play)
            else:
                self.root.after(0, self.progress.stop)
                self.update_status("❌ TTS generation failed")
                self.update_error_info("Text-to-speech synthesis failed")
            
        except Exception as e:
            self.root.after(0, self.progress.stop)
            error_msg = f"Processing failed: {str(e)}"
            self.update_status(f"❌ Error: {str(e)}")
            self.update_error_info(error_msg)
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
    
    def fallback_tts(self, text, language):
        """Fallback TTS using the original IndicParlerTTS with cache fixes"""
        try:
            if not hasattr(self, 'tts_model'):
                print("❌ Fallback TTS model not available")
                return False
            
            speaker = self.speaker_var.get()
            description = f"{speaker}'s voice is clear and natural."
            
            description_inputs = self.description_tokenizer(description, return_tensors="pt")
            prompt_inputs = self.tts_tokenizer(text, return_tensors="pt")
            
            description_input_ids = description_inputs.input_ids.to(self.DEVICE)
            description_attention_mask = description_inputs.attention_mask.to(self.DEVICE)
            prompt_input_ids = prompt_inputs.input_ids.to(self.DEVICE)
            prompt_attention_mask = prompt_inputs.attention_mask.to(self.DEVICE)
            
            # CRITICAL FIX: Use no_grad and disable cache
            with torch.no_grad():
                generation = self.tts_model.generate(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    use_cache=False,  # Disable cache to prevent layer access error
                    do_sample=False
                )
            
            audio_arr = generation.cpu().numpy().squeeze()
            sf.write(self.output_file, audio_arr, self.tts_model.config.sampling_rate)
            
            return os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 1000
            
        except Exception as e:
            print(f"❌ Fallback TTS failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def play_audio(self):
        try:
            if os.name == "nt":  # Windows
                os.system(f"start {self.output_file}")
            else:  # Linux/Mac
                os.system(f"open {self.output_file}")  # macOS
                # For Linux: os.system(f"mpg123 {self.output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not play audio: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()