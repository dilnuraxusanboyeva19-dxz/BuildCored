import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import sounddevice as sd
    HAS_PLAYBACK = True
except ImportError:
    HAS_PLAYBACK = False

SAMPLE_RATE = 16000
def generate_synthetic_speech(duration=3.0, sample_rate=SAMPLE_RATE):
    """Generate speech-like audio using frequency-modulated sine waves."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = np.zeros_like(t)
    for burst_start in [0.2, 0.9, 1.6, 2.3]:
        burst_len = 0.5
        mask = (t >= burst_start) & (t < burst_start + burst_len)
        fundamental = np.random.uniform(100, 200)
        word = np.zeros_like(t)
        for harmonic in range(1, 5):
            word += np.sin(2 * np.pi * fundamental * harmonic * t) / harmonic
        envelope = np.zeros_like(t)
        envelope[mask] = np.sin(np.pi * (t[mask] - burst_start) / burst_len)
        signal += word * envelope
    signal = signal / np.max(np.abs(signal)) * 0.6
    return signal.astype(np.float32)

def add_synthetic_echo(signal, delay_ms=150, decay=0.5, sample_rate=SAMPLE_RATE):
    """Add a single echo to the signal."""
    delay_samples = int(sample_rate * delay_ms / 1000)
    echo = np.zeros_like(signal)
    if delay_samples < len(signal):
        echo[delay_samples:] = signal[:-delay_samples] * decay
    return signal + echo

def load_or_generate():
    """Try to load a .wav file from current dir. Otherwise generate one."""
    wav_files = [f for f in os.listdir(".") if f.lower().endswith(".wav")]
    if wav_files and HAS_SOUNDFILE:
        path = wav_files[0]
        print(f"📂 Loading: {path}")
        try:
            data, sr = sf.read(path)
            if data.ndim > 1: data = data[:, 0]
            if sr != SAMPLE_RATE:
                ratio = sr / SAMPLE_RATE
                indices = (np.arange(int(len(data) / ratio)) * ratio).astype(int)
                data = data[indices]
            return data.astype(np.float32), True
        except Exception as e:
            print(f"   Load failed: {e}")
    print("📡 No .wav file found — generating synthetic speech")
    return generate_synthetic_speech(), False

FILTER_ORDER = 2500  
LEARNING_RATE = 0.02 

def lms_filter(reference, mixed, filter_order, mu):
    """
    Normalized LMS adaptive filter.
    reference: clean signal (x)
    mixed: signal with echo (d)
    """
    N = len(mixed)
    w = np.zeros(filter_order, dtype=np.float32)
    error = np.zeros(N, dtype=np.float32)
    ref_buffer = np.zeros(filter_order, dtype=np.float32)

    for n in range(N):
        ref_buffer[1:] = ref_buffer[:-1]
        ref_buffer[0] = reference[n]

        predicted_echo = np.dot(w, ref_buffer)

        e = mixed[n] - predicted_echo
        error[n] = e

        norm = np.dot(ref_buffer, ref_buffer) + 1e-6
        w = w + (mu / norm) * e * ref_buffer

    return error, w


def plot_results(clean, echoed, cleaned, coefficients, sample_rate):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10))
    fig.suptitle("EchoKiller — Day 16", fontsize=14, fontweight='bold')
    t = np.arange(len(clean)) / sample_rate

    axes[0].plot(t, clean, color='#43a047', linewidth=0.8)
    axes[0].set_title("1. Clean Reference")
    axes[1].plot(t, echoed, color='#e53935', linewidth=0.8)
    axes[1].set_title("2. With Echo (Mic Input)")
    axes[2].plot(t, cleaned, color='#1e88e5', linewidth=0.8)
    axes[2].set_title("3. After EchoKiller (LMS Output)")
    

    tap_times = np.arange(len(coefficients)) / sample_rate * 1000 
    axes[3].stem(tap_times[::5], coefficients[::5], basefmt=" ", linefmt='#ff6f00', markerfmt='o')
    axes[3].set_title(f"4. Learned FIR Coefficients (Room Impulse Response)")
    axes[3].set_xlabel("Delay (ms)")
    
    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("\n" + "=" * 50)
    print("  🔇 EchoKiller — Adaptive FIR Echo Cancellation")
    print("=" * 50)

    clean, from_file = load_or_generate()
    echoed = add_synthetic_echo(clean, delay_ms=150, decay=0.5)

    print(f"\n⚙️ Running LMS: Order={FILTER_ORDER}, Mu={LEARNING_RATE}")
    start = time.time()
    cleaned, coefficients = lms_filter(clean, echoed, FILTER_ORDER, LEARNING_RATE)
    print(f"   Done in {time.time() - start:.1f}s")


    echo_energy = np.mean(echoed ** 2)
    cleaned_energy = np.mean(cleaned ** 2)
    reduction_db = 10 * np.log10(echo_energy / max(cleaned_energy, 1e-9))
    print(f"📊 Noise reduction: {reduction_db:.1f} dB")

    if HAS_PLAYBACK:
        try:
            print("\n🔊 Playing CLEANED audio...")
            sd.play(cleaned, SAMPLE_RATE); sd.wait()
        except KeyboardInterrupt:
            sd.stop()

    if HAS_SOUNDFILE:
        sf.write("cleaned_output.wav", cleaned, SAMPLE_RATE)
        print("\n💾 Saved: cleaned_output.wav")

    plot_results(clean, echoed, cleaned, coefficients, SAMPLE_RATE)

if __name__ == "__main__":
    main()
