import numpy as np
import sounddevice as sd
import urllib.request
import json
import sys
from scipy import signal
from rich.console import Console

SAMPLE_RATE = 44100
DURATION = 15.0 
console = Console()

def get_best_model():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req) as resp:
            models = [m["name"] for m in json.loads(resp.read()).get("models", [])]
            for m in models:
                if any(x in m.lower() for x in ['qwen', 'llama', 'phi']): return m
            return models[0] if models else None
    except:
        return None

def generate_ambient_audio(mood, model):
    console.print(f"[yellow]Consulting Edge AI for the acoustic matrix of '{mood}'...[/yellow]")
    
    prompt = (
        f"Generate exactly 5 parameters for a procedural audio engine to recreate the mood: '{mood}'.\n"
        "1. 'base_freq': (float 40.0 to 400.0) Main root tone frequency in Hz.\n"
        "2. 'chord_type': (string) ONLY return exactly 'major' (happy/peaceful) or 'minor' (sad/dark/tense).\n"
        "3. 'noise_level': (float 0.0 to 1.0) Volume of background texture. 1.0=heavy chaos/storm, 0.0=pure isolated bell.\n"
        "4. 'brightness': (float 0.1 to 1.0) LPF Cutoff. 0.1=muffled rumbling darkness, 1.0=bright airy wind.\n"
        "5. 'lfo_rate': (float 0.1 to 3.0) Speed of pulsing volume swells.\n"
        "Ensure keys are exactly as written above."
    )
    
    data = json.dumps({
        "model": model, 
        "prompt": prompt, 
        "stream": False,
        "format": "json"
    }).encode('utf-8')
    
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data)
    
    try:
        with urllib.request.urlopen(req) as resp:
            raw_res = json.loads(resp.read())['response']
            parsed = json.loads(raw_res)
            
            freq = max(20.0, min(float(parsed.get("base_freq", 110.0)), 1000.0))
            chord = str(parsed.get("chord_type", "minor")).lower()
            noise_lvl = max(0.0, min(float(parsed.get("noise_level", 0.5)), 1.0))
            brightness = max(0.1, min(float(parsed.get("brightness", 0.3)), 1.0))
            lfo = max(0.01, min(float(parsed.get("lfo_rate", 0.5)), 10.0))
            
    except Exception as e:
        console.print(f"[red]Generation parsing blocked: {e}. Yielding default ambient construct.[/red]")
        freq, chord, noise_lvl, brightness, lfo = 100.0, "minor", 0.5, 0.3, 0.5
        
    console.print(f"[cyan]Synthesizing {DURATION} seconds of High-Fidelity Audio...[/cyan]")
    console.print(f" ▸ Root Freq   : {freq:.1f} Hz [{chord.upper()} CHORD]")
    console.print(f" ▸ Texture Lvl : {noise_lvl*100:.1f} %")
    console.print(f" ▸ Brightness  : {brightness*100:.1f} % (LPF Factor)")
    console.print(f" ▸ Swell LFO   : {lfo:.2f} Hz")
    
    # Procedural Generation Core
    frames = int(SAMPLE_RATE * DURATION)
    t = np.arange(frames) / SAMPLE_RATE
    
    # 1. Harmonic Drone Generation
    ratios = [1.0, 1.25, 1.5] if chord == "major" else [1.0, 1.189, 1.5]
    drone = np.zeros(frames)
    for r in ratios:
        # Subtle detuning per voice for thick analog chorus effect
        detune = np.random.uniform(0.995, 1.005)
        drone += np.sin(2 * np.pi * (freq * r * detune) * t)
    drone /= len(ratios) # Normalize triad
    
    # Add heavy sub-octave rumble for cinematic depth
    drone += 0.8 * np.sin(2 * np.pi * (freq / 2.0) * t)
    
    # 2. Scipy Filtered Textural Noise
    raw_noise = np.random.normal(0, 1, frames)
    # Brightness maps aggressively to filter cutoff (400Hz to 6000Hz)
    cutoff = 400.0 + (brightness * 5600.0)
    b, a = signal.butter(1, cutoff / (SAMPLE_RATE / 2.0), btype='low')
    filtered_noise = signal.lfilter(b, a, raw_noise)
    
    # 3. Mixing Console
    mix = (drone * (1.0 - (noise_lvl * 0.7))) + (filtered_noise * noise_lvl * 0.5)
    
    # 4. Envelopes
    envelope = (np.sin(2 * np.pi * lfo * t) + 1.0) / 2.0
    
    fade_len = int(SAMPLE_RATE * 2.0) 
    master_env = np.ones(frames)
    master_env[:fade_len] = np.linspace(0, 1, fade_len)
    master_env[-fade_len:] = np.linspace(1, 0, fade_len)
    
    # Final mastering constraints
    final_audio = mix * (0.3 + 0.7 * envelope) * master_env * 0.15
    
    console.print("[bold green]▶ Playing Ambient Track (Locally Synthsized)[/bold green]")
    sd.play(final_audio, SAMPLE_RATE)
    sd.wait() 
    console.print("[dim]■ Playback finished.[/dim]\n")

def main():
    model = get_best_model()
    if not model:
        console.print("[bold red]Ollama daemon not running. Please start it.[/bold red]")
        sys.exit(1)
        
    console.print("[bold magenta]MoodSynth Cinematic Generator[/bold magenta]\n")
    
    while True:
        try:
            mood = input("Enter a Vibe (or 'quit'): ")
            if mood.lower() in ['exit', 'quit']: break
            if mood.strip():
                generate_ambient_audio(mood, model)
        except (KeyboardInterrupt, EOFError):
            break
            
    console.print("\n[dim]Disconnected.[/dim]")

if __name__ == "__main__":
    main()
