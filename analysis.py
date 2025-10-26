import requests
import time
import json
import random
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import Google Generative AI SDK components
try:
    from google import genai
    from google.genai import types
except ImportError:
    # Set to None if the library is not installed
    genai = None

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
HF_TOKEN = os.getenv("HF_INFERENCE_TOKEN")

# Gemini LLM Configuration
GEMINI_MODEL = 'gemini-2.5-flash'
# Global client MUST be initialized to None at the module level
# We use this as a cache for the client instance
CLIENT_CACHE = None 


# --- CORE ANALYSIS CONSTANTS ---
FILLER_WORDS = {"um", "uh", "like", "you know", "so", "actually", "basically", "i mean", "right", "okay"}
VAGUE_PHRASES = ["thing", "stuff", "sort of", "kind of", "you know", "like"] 
TARGET_WPM_RANGE = (140, 160)
WPM_WINDOW_SEC = 10
TARGET_WPM_SIM = 150 
STRATEGIC_PAUSE_MIN_SEC = 1.5
HESITATION_MAX_SEC = 0.5


# --- UTILITY AND ANALYSIS FUNCTIONS ---

def _get_gemini_client():
    """Initializes and returns the cached Gemini client instance."""
    global CLIENT_CACHE
    
    if CLIENT_CACHE is not None:
        return CLIENT_CACHE
    
    if genai:
        try:
            # Attempt initialization if not yet done
            CLIENT_CACHE = genai.Client()
            return CLIENT_CACHE
        except Exception as e:
            # Print a warning but don't stop the module load
            print(f"Warning: Failed to initialize Gemini client. Check GEMINI_API_KEY. Error: {e}")
            return None
    return None


def call_whisper_api(audio_bytes: bytes) -> str:
    """Calls the Whisper API to get raw text transcription."""
    if not HF_TOKEN:
        raise ValueError("HF_INFERENCE_TOKEN is not set.")
        
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/wav" 
    }
    
    response = requests.post(API_URL, headers=headers, data=audio_bytes)
    response.raise_for_status()
    
    whisper_output = response.json()
    return whisper_output.get('text', '')


def _simulate_whisper_output(text: str) -> List[Dict[str, Any]]:
    """
    Simulates structured word-level data with realistic, variable timing.
    """
    AVG_SEC_PER_WORD = 60 / TARGET_WPM_SIM
    words = text.lower().split()
    simulated_data = []
    current_time = 0.0
    
    for word in words:
        duration_variation = random.uniform(AVG_SEC_PER_WORD * 0.70, AVG_SEC_PER_WORD * 1.30)
        
        start = current_time
        end = current_time + duration_variation
        
        simulated_data.append({
            "text": word,
            "start": round(start, 2),
            "end": round(end - start, 2),
            "duration": round(end - start, 2),
            "tags": []
        })
        current_time = end
        
    return simulated_data


def detect_fillers(word_list: List[Dict[str, Any]]):
    """Detects and tags filler words in the list."""
    filler_count = 0
    for word in word_list:
        text = word["text"].lower()
        if text in FILLER_WORDS:
            word["tags"].append("filler")
            word["tags"].append(text)
            filler_count += 1
    return word_list, filler_count


def analyze_pacing(word_list: List[Dict[str, Any]], target_wpm=TARGET_WPM_RANGE):
    """Calculates overall and local WPM, tagging fast/slow segments."""
    if not word_list:
        return word_list, 0.0, {}

    total_words = len(word_list)
    
    # Fix: Calculate total duration considering the duration of each word
    total_duration = 0.0
    for word in word_list:
        total_duration += word.get('duration', 0)
    
    # Convert to minutes and calculate WPM
    total_duration_minutes = total_duration / 60.0

    # Prevent division by zero and unrealistic WPM
    if total_duration_minutes <= 0:
        return word_list, 150.0, {}  # Return average speaking rate
        
    overall_wpm = total_words / total_duration_minutes
    
    # Cap WPM to realistic range
    overall_wpm = min(max(overall_wpm, 60), 300)
    
    # Calculate local pacing
    wpm_map = {}
    for i, word in enumerate(word_list):
        window_start = word["start"] - WPM_WINDOW_SEC / 2
        window_end = word["start"] + WPM_WINDOW_SEC / 2
        
        window_words = [w for w in word_list 
                       if w["start"] >= window_start and w["end"] <= window_end]
        
        if len(window_words) < 5: 
            continue
            
        # Calculate window duration using word durations
        window_duration = sum(w.get('duration', 0) for w in window_words)
        local_wpm = (len(window_words) / (window_duration / 60)) if window_duration > 0 else 0.0
        
        if local_wpm > target_wpm[1] * 1.1:
            word["tags"].append("fast_pacing")
        elif local_wpm < target_wpm[0] * 0.9:
            word["tags"].append("slow_pacing")
            
        wpm_map[word["start"]] = round(local_wpm)

    return word_list, overall_wpm, wpm_map


def analyze_pauses(word_list: List[Dict[str, Any]]):
    """
    Analyzes time gaps between words to identify Strategic Pauses vs. Hesitations.
    """
    strategic_pauses = 0
    hesitation_gaps = 0
    
    if len(word_list) < 2:
        return word_list, strategic_pauses, hesitation_gaps

    for i in range(1, len(word_list)):
        prev_word = word_list[i-1]
        current_word = word_list[i]
        
        gap_duration = current_word["start"] - prev_word["end"]
        
        if gap_duration > 0:
            is_followed_by_filler = "filler" in current_word["tags"]
            
            if gap_duration >= STRATEGIC_PAUSE_MIN_SEC and not is_followed_by_filler:
                strategic_pauses += 1
                prev_word["tags"].append("strategic_pause")
            
            elif gap_duration > HESITATION_MAX_SEC and is_followed_by_filler:
                hesitation_gaps += 1
                prev_word["tags"].append("hesitation_gap")
            
            elif gap_duration > HESITATION_MAX_SEC:
                prev_word["tags"].append("long_pause")
                
    return word_list, strategic_pauses, hesitation_gaps


def simulate_acoustic_metrics(wpm_map: Dict[float, int]) -> Dict[str, Any]:
    """
    Simulates acoustic metrics based on WPM variability (as a proxy).
    """
    metrics = {
        "avg_volume_status": "Normal",
        "pitch_monotony_score": 0.0
    }
    
    if not wpm_map:
        return metrics

    local_wpms = list(wpm_map.values())
    
    if len(local_wpms) > 1:
        mean_wpm = sum(local_wpms) / len(local_wpms)
        variance = sum([(x - mean_wpm) ** 2 for x in local_wpms]) / len(local_wpms)
        std_dev = variance ** 0.5
        
        monotony_score = max(0, min(100, 100 - round(std_dev * 5)))
        metrics["pitch_monotony_score"] = monotony_score
        
        if random.random() < 0.2:
             metrics["avg_volume_status"] = "Quiet"

    return metrics


def analyze_vague_language(raw_text: str) -> List[str]:
    """
    Uses the Gemini LLM to identify specific instances of vague or non-committal language.
    """
    client = _get_gemini_client()
    if client is None:
        return ["Vague language analysis skipped: Gemini API key not configured."]

    system_instruction = (
        "You are an expert editor. Your task is to analyze a transcription for any "
        "vague, non-committal, or diluting language (e.g., 'thing', 'stuff', 'sort of', "
        "'basically', 'I think'). Identify these phrases. "
        "Return ONLY a JSON list of the vague phrases you found. Do not include filler words like 'um' or 'uh'."
    )
    
    user_prompt = f"Transcription: '{raw_text}'"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={"type": "array", "items": {"type": "string"}},
                temperature=0.1
            )
        )
        
        vague_list = json.loads(response.text)
        if isinstance(vague_list, list):
            return vague_list
        return []

    except Exception as e:
        return ["Vague language analysis failed due to API error."]


def calculate_clarity_score(overall_wpm: float, filler_count: int, total_words: int, acoustic_metrics: Dict[str, Any]) -> int:
    """Calculates a score (0-100) incorporating new acoustic metrics."""
    score = 100
    MAX_PENALTY = 25

    # 1. Filler Penalty
    filler_rate_percent = (filler_count / total_words) * 100 if total_words > 0 else 0
    filler_penalty = min(filler_rate_percent * 6, MAX_PENALTY)
    score -= filler_penalty

    # 2. Pacing Penalty
    deviation = 0
    if overall_wpm > TARGET_WPM_RANGE[1]:
        deviation = overall_wpm - TARGET_WPM_RANGE[1]
    elif overall_wpm < TARGET_WPM_RANGE[0]:
        deviation = TARGET_WPM_RANGE[0] - overall_wpm
    
    pacing_penalty = min(deviation * 0.3, MAX_PENALTY)
    score -= pacing_penalty

    # 3. Monotony Penalty
    monotony = acoustic_metrics.get("pitch_monotony_score", 0)
    if monotony > 70:
        score -= min(15, (monotony - 70) * 0.5)

    # 4. Volume Penalty
    if acoustic_metrics.get("avg_volume_status") == "Quiet":
        score -= 10

    return max(0, min(100, round(score)))


def get_llm_rewrite(original_phrase: str) -> str:
    """Sends a problematic phrase to the Gemini LLM for a professional rewrite."""
    client = _get_gemini_client()
    if client is None:
        return "Suggestion failed: Gemini API key not configured."

    system_instruction = (
        "You are an expert presentation coach. Your sole task is to take a given phrase "
        "and rewrite it to be more concise, professional, and clear. You MUST completely "
        "remove all filler words, hesitations, and unnecessary conversational language. "
        "Return ONLY the polished, improved sentence."
    )
    
    user_prompt = f"Original Phrase to rewrite: '{original_phrase}'"
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=100
            )
        )
        llm_output = response.text.strip()
        return llm_output if llm_output else "The phrasing could be more direct."

    except Exception as e:
        return f"Suggestion failed due to API error: {e}"


def analyze_topic_relevance(raw_text: str, topic: str):
    """
    Uses Gemini to analyze the relevance of the transcription content to the topic.
    Returns a score and content suggestions.
    """
    client = _get_gemini_client()
    if client is None:
        return None, ["Relevance analysis skipped: Gemini API not configured."]

    system_instruction = (
        "You are an expert content analyzer for presentations. Your task is to critique "
        "the content's relevance to the stated topic and provide specific, actionable suggestions. "
        "The presentation content is: "
    )
    
    user_prompt = (
        f"The user's presentation topic is: '{topic}'\n\n"
        f"The user's raw transcription is: '{raw_text}'\n\n"
        "Critique the transcription based ONLY on its relevance to the topic. "
        "First, assign a **Relevance Score from 1 (low) to 10 (high)**. "
        "Second, provide a **short, three-bullet list of relevant points** the user "
        "should ADD to their presentation to improve coverage or depth. "
        "Format your entire response as a JSON object with two keys: 'score' (integer) and 'suggestions' (list of strings)."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={"type": "object", "properties": {
                    "score": {"type": "integer"},
                    "suggestions": {"type": "array", "items": {"type": "string"}}
                }},
                temperature=0.3
            )
        )
        
        json_output = json.loads(response.text)
        score = json_output.get("score")
        suggestions = json_output.get("suggestions")
        
        return score, suggestions

    except Exception as e:
        print(f"Warning: Gemini Content Analysis failed: {e}")
        return None, ["Content analysis failed due to API error."]


def get_llm_presentation_script(topic: str, time_limit_minutes: int) -> str:
    """
    Uses the Gemini LLM to generate a full presentation script with proper
    greetings, transitions, and a thank you, suitable for a voiceover.
    """
    client = _get_gemini_client()
    if client is None:
        return "Script generation failed: Gemini API service is unavailable."

    # Approximate word count based on 150 WPM
    estimated_word_count = time_limit_minutes * 150

    system_instruction = (
        "You are an expert, professional, and engaging presenter. Your task is to generate "
        "a complete, ready-to-read presentation script for an AI voiceover. "
        "The script MUST flow naturally and include: "
        "1. A polite greeting (e.g., 'Good morning/afternoon, everyone...'). "
        "2. A clear introduction to the topic. "
        "3. Main content (concise and well-structured). "
        "4. A concluding summary. "
        "5. A professional thank you (e.g., 'Thank you for your time and attention.'). "
        "Return ONLY the plain text of the full script, without any markdown headings or extra commentary."
    )
    
    user_prompt = (
        f"Generate a professional presentation script on the following topic: '{topic}'. "
        f"The script should be approximately {time_limit_minutes} minutes long, "
        f"corresponding to about {estimated_word_count} words. "
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        return response.text.strip()

    except Exception as e:
        print(f"Warning: Gemini Voiceover Script Generation failed: {e}")
        return "Script generation failed due to API error."


# --- NEW FUNCTION FOR PRESENTATION CONTENT ---

def get_llm_presentation_content(topic: str, time_limit_minutes: int) -> Dict[str, Any]:
    """
    Uses the Gemini LLM to generate a structured presentation outline, content, and 
    suggested visuals/charts. Returns a dict with 'template_suggestion' and 'slides'.
    """
    client = _get_gemini_client()
    if client is None:
        return {"error": "Presentation content generation failed: Gemini API service is unavailable."}

    # Approximate word count based on 150 WPM
    estimated_word_count = time_limit_minutes * 150

    system_instruction = (
        "You are an expert presentation designer. Your task is to create a complete, "
        "structured presentation outline for a given topic and time limit. "
        "The output MUST be a JSON object with two keys: "
        "'template_suggestion' (a single short paragraph describing visual style, color palette, and focal imagery), "
        "and 'slides' (an array of slide objects). Each slide object must contain: "
        "1. 'slide_title': A concise title. "
        "2. 'main_points': A list of 3-5 key bullet points for the slide. "
        "3. 'visual_suggestion': A concise suggestion (max 7 words) for a visual element (Image, Chart, or Graph). "
        "The presentation MUST include a Title Slide, an Introduction, 2-4 Content Slides, and a Conclusion Slide. "
        "Return JSON only."
    )
    
    user_prompt = (
        f"Generate a professional presentation outline on the topic: '{topic}'. "
        f"The content should be designed for a presentation of approximately {time_limit_minutes} minutes, "
        f"or about {estimated_word_count} words. Ensure a logical, professional flow."
    )

    try:
        # Primary request: expect a JSON object with template_suggestion + slides
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "template_suggestion": {"type": "string"},
                        "slides": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "slide_title": {"type": "string"},
                                    "main_points": {"type": "array", "items": {"type": "string"}},
                                    "visual_suggestion": {"type": "string"}
                                },
                                "required": ["slide_title", "main_points", "visual_suggestion"]
                            }
                        }
                    },
                    "required": ["template_suggestion", "slides"]
                },
                temperature=0.6,
                max_output_tokens=2048
            )
        )

        parsed = json.loads(response.text)

        # If the model returned a plain array (older behavior), handle fallback
        if isinstance(parsed, list):
            slides = parsed
            # Attempt to generate a concise template suggestion using slide summaries
            try:
                slide_summary = "\n".join([f"- {s.get('slide_title','')}: {s.get('visual_suggestion','')}" for s in slides])
                tmpl_prompt = (
                    f"Provide a single concise template suggestion (visual style, color palette, and focal imagery) "
                    f"for a presentation about '{topic}'. Use the following slides to inform the design:\n\n{slide_summary}\n\n"
                    f"Return only the template suggestion as one short paragraph."
                )
                tmpl_resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[tmpl_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction="You are an expert presentation designer. Return one short polished template suggestion.",
                        temperature=0.5,
                        max_output_tokens=120
                    )
                )
                template_suggestion = tmpl_resp.text.strip() or ""
            except Exception:
                template_suggestion = ""
            return {"template_suggestion": template_suggestion, "slides": slides}

        # If parsed is already an object with both keys, return directly
        if isinstance(parsed, dict):
            template = parsed.get("template_suggestion", "")
            slides = parsed.get("slides", [])
            return {"template_suggestion": template, "slides": slides}

        # Unexpected format
        return {"error": f"Presentation content generation returned unexpected format: {type(parsed)}"}

    except Exception as e:
        print(f"Warning: Gemini Presentation Content Generation failed: {e}")
        return {"error": f"Presentation content generation failed due to API error: {e}"}

def analyze_script_redundancy(script: str, slide_bullets: List[str]) -> Dict[str, Any]:
    """
    Uses the LLM to compare a full script against slide bullet points
    to identify redundancy (i.e., reading the slides verbatim).
    Returns a redundancy score and specific redundant phrases.
    """
    client = _get_gemini_client()
    if client is None:
        return {"error": "Redundancy analysis skipped: Gemini API not configured."}

    # Format the slide bullets into a single, structured block for the prompt
    formatted_slides = "\n".join([f"- {bullet}" for bullet in slide_bullets])

    system_instruction = (
        "You are an expert presentation coach. Your task is to analyze a speaker script "
        "and compare it to the slide bullet points to check for redundancy. "
        "High redundancy means the speaker is reading the slides verbatim, which is poor practice. "
        "1. Assign a **Redundancy Score from 0 (Perfect/No overlap) to 100 (High/Total overlap)**. "
        "2. Identify up to five **exact or near-exact phrases** from the script that are redundant with the slides. "
        "Return ONLY a JSON object with two keys: 'redundancy_score' (integer) and 'redundant_phrases' (list of strings)."
    )
    
    user_prompt = (
        "Compare the following speaker script to the slide content.\n\n"
        f"--- SLIDE CONTENT ---\n{formatted_slides}\n\n"
        f"--- SPEAKER SCRIPT ---\n{script}\n"
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={"type": "object", "properties": {
                    "redundancy_score": {"type": "integer"},
                    "redundant_phrases": {"type": "array", "items": {"type": "string"}}
                }},
                temperature=0.1
            )
        )
        
        return json.loads(response.text)

    except Exception as e:
        return {"error": f"Redundancy analysis failed due to API error: {e}"}
