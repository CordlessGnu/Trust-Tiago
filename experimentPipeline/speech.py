import os
import sys

import requests

# Setting up cuda
nvidia_dirs = []
site_packages_nvidia = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")

if os.path.exists(site_packages_nvidia):
    for root, dirs, files in os.walk(site_packages_nvidia):
        if any(f.endswith(".dll") for f in files):
            nvidia_dirs.append(root)

for path in nvidia_dirs:
    try:
        os.add_dll_directory(path)
    except Exception:
        pass
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Imports
import time
import keyboard
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openai import OpenAI
from pykokoro import KokoroPipeline, PipelineConfig
import re

pipeline = PipelineConfig()
device = "cuda"
compute_type = "float16"

# Whisper model setup
try:
    model = WhisperModel("small", device=device, compute_type=compute_type)
except Exception:
    device = "cpu"
    compute_type = "int8"
    model = WhisperModel("small", device=device, compute_type=compute_type)

# System prompt for VLM
sys_prompt = (
    "You are Tiago, an interactive cooking robot assistant with a built-in camera. "
    "You interact verbally with the user while delegating physical robotic actions to a Vision-Language-Action model (VLA).\n\n"

    "ROBOT CONTROL & INGREDIENT HANDLING:\n"
    "- YOU (the robot) physically move ingredients from the right side of the table to the left side for the user.\n"
    "- In your camera view, all items start on the right-hand side of the table.\n"
    "- Whenever a step requires an ingredient or tool from the right side, issue the <VLA> command IN THE SAME TURN as your spoken guidance.\n"
    "- NEVER ask the user to fetch items from the right side for you.\n"
    "- Always accompany every <VLA> command with spoken conversational text explaining what you are handing them (e.g., 'Here are the crushed tomatoes. <VLA>Pick up the can of crushed tomatoes from the right side of the table and place it on the left side</VLA>').\n"
    "- You will be automatically notified upon completion of a VLA action; do not repeat commands.\n\n"

    "CORE WORKFLOW PIPELINE:\n"
    "1. Introduction: Greet the user and introduce yourself as Tiago.\n"
    "2. Dietary Preferences: Ask for any dietary restrictions or preferences.\n"
    "3. Recipe Recommendation & Selection: Suggest specific recipes matching their preferences. "
    "Do NOT assume they have an external recipe. Provide full recipes step-by-step yourself.\n"
    "4. Confirmation: Once a meal is chosen, explicitly confirm: 'Are you ready to start cooking?'\n"
    "5. Step-by-Step Guidance & Execution: Present EXACTLY ONE concise step at a time. "
    "If the step requires an item from the right side, immediately include the <VLA> transfer tag in your turn and wait for user verbal confirmation before moving to the next step.\n\n"

    "CONSTRAINTS & GUARDRAILS:\n"
    "- One step at a time: Never output multiple steps, full ingredient lists, or pre-instructions at once.\n"
    "- Strict Privacy: Never infer cultural background, ethnicity, or history beyond explicit user input.\n\n"

    "TEXT-TO-SPEECH (TTS) FORMATTING RULES:\n"
    "- Spoken text (outside of <VLA></VLA> tags) must be pure conversational plain text.\n"
    "- ABSOLUTELY NO special characters, asterisks (*), dashes (-), bullet points, numbered lists, markdown, or LaTeX.\n"
    "- SPELL OUT all numbers, fractions, and units (e.g., write 'two cloves', 'one half cup', 'three tablespoons').\n"
    "- Keep every turn brief and under seventy-five words."
)

# Setting up LMstudio API information
model_name = "google/gemma-4-e2b"
client = OpenAI(
    base_url="",
    api_key="",
)

# Stores conversation history
convo_history = [
    {"role": "system", "content": sys_prompt}
]

pipe = KokoroPipeline(PipelineConfig(voice="af_sarah"))
# model = WhisperModel("small", device="cuda", compute_type="float16")

# Function to post the task to the server
def send_vla_task(task_text):
    if not task_text:
        return
    try:
        url = ""
        res = requests.post(url, params={"task": task_text.strip()}, timeout=1.0)
        if res.status_code == 200:
            print(f"[VLA Task Dispatched]: {task_text}")
        else:
            print(f"[VLA Error {res.status_code}]: {res.text}")
    except Exception as e:
        print(f"[VLA Connection Failed]: {e}")

# Cleaning the text for TTS and extracting the task
def speak_text(pipe, text):
    vla_match = re.search(r"<VLA>(.*?)</VLA>", text, flags=re.DOTALL)
    if vla_match:
        vla_task = vla_match.group(1).strip()
        send_vla_task(vla_task)

    clean_text = re.sub(r"<VLA>.*?</VLA>", "", text, flags=re.DOTALL).strip()
    if not clean_text:
        return

    print(f"[Speaking]: {clean_text}")
    res = pipe.run(clean_text)
    # Outputting the text audibly
    sd.play(res.audio, samplerate=24000)
    sd.wait()

# Cleaning the transcription
def transcribe_safe(model, audio):
    global device, compute_type
    try:
        segments, _ = model.transcribe(audio, beam_size=2)
        return " ".join(segment.text for segment in segments).strip()
    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cudnn" in str(e).lower():
            device = "cpu"
            compute_type = "int8"
            cpu_model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, _ = cpu_model.transcribe(audio, beam_size=2)
            return " ".join(segment.text for segment in segments).strip()
        else:
            raise e

# Recording the audio then transcribing
def record_and_transcribe(hotkey="v"):
    print("\n[Recording started - release and press 'v' again to stop]")
    chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        chunks.append(indata.copy())

    while keyboard.is_pressed(hotkey):
        time.sleep(0.05)

    with sd.InputStream(samplerate=16000, channels=1, callback=callback):
        keyboard.wait(hotkey)

    print("[Recording stopped]")

    while keyboard.is_pressed(hotkey):
        time.sleep(0.05)

    if not chunks:
        return ""

    audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
    return transcribe_safe(model, audio)

# Sending user response to the VLM
def get_llm_response(msg_content):
    convo_history.append({"role": "user", "content": msg_content})

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=convo_history,
        )

        completion_text = completion.choices[0].message.content
        convo_history.append({"role": "assistant", "content": completion_text})
        return completion_text
    except Exception as e:
        return f"API Error: {e}"

def main():
    # Prompting VLM
    starter_response = get_llm_response("Hi")
    print(f"starter response: {starter_response}\n")
    speak_text(pipe, starter_response)
    hotkey = "v"
    print(f"Press '{hotkey}' to start/stop recording.")
    print("Press 'Esc' to exit script.\n")

    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == hotkey:
                transcript = record_and_transcribe(hotkey=hotkey)
                if transcript:
                    print(f"[Transcription]: {transcript}")
                    print("[Requesting ingredients from API...]")
                    ingredients = get_llm_response(transcript)
                    print(f"[API Response]:\n{ingredients}\n")
                    speak_text(pipe, ingredients)
            elif event.name == "esc":
                print("Exiting...")
                break


if __name__ == "__main__":
    main()