import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from io import BytesIO
import pyttsx3 
from enum import Enum
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

# IMPORT FIX: Explicitly import the analysis module to allow access to its globals (analysis.client)
import analysis
from analysis import (
    call_whisper_api, _simulate_whisper_output, detect_fillers, 
    analyze_pacing, calculate_clarity_score, get_llm_rewrite,
    analyze_vague_language, analyze_topic_relevance, analyze_pauses, 
    simulate_acoustic_metrics, TARGET_WPM_RANGE,_get_gemini_client,
    get_llm_presentation_script, get_llm_presentation_content, analyze_script_redundancy # <--- NEW IMPORT
)

# Load environment variables
load_dotenv()

# Configure Cloudinary with environment variables
cloudinary.config( 
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key = os.getenv('CLOUDINARY_API_KEY'),
    api_secret = os.getenv('CLOUDINARY_API_SECRET')
)

# --- Pydantic Response Models ---

class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    tags: List[str]

class AcousticMetrics(BaseModel):
    avg_volume_status: str
    pitch_monotony_score: float

class FullEvaluationResponse(BaseModel):
    # Core Delivery Metrics
    clarity_score: int
    overall_wpm: float
    filler_count: int
    filler_words_used: List[str] = Field(default_factory=list, description="The actual filler words found in the speech")  # Add this line
    strategic_pauses: int
    hesitation_gaps: int
    acoustic_metrics: AcousticMetrics
    
    # Content & Vague Language Metrics
    relevance_score: Optional[int] = None # Optional/None if topic is not provided
    suggested_content: List[str] = []
    vague_phrases_found: List[str]
    
    # Output
    raw_transcription: str
    feedback: List[str]
    transcript: List[TranscriptSegment]

class ContentAnalysisResponse(BaseModel):
    relevance_score: Optional[int]
    suggested_content: List[str]

# MODEL 1: For the speech generation endpoint
class SpeechDraftResponse(BaseModel):
    topic: str
    time_limit_minutes: float
    estimated_word_count: int
    generated_speech_draft: List[str]  # Changed from List[Dict[str, Any]] to List[str]
    # New: tone used for generation
    tone: str
    
# MODEL 2: For the AI voiceover script endpoint
class AIVoiceoverResponse(BaseModel):
    topic: str
    time_limit_minutes: float
    estimated_word_count: int
    full_voiceover_script: str
    audio_file_url: str  # Changed from audio_file_saved_as to audio_file_url
    
# NEW MODEL 3: For the presentation content endpoint
class PresentationSlide(BaseModel):
    slide_title: str
    main_points: List[str]
    visual_suggestion: str

class PresentationContentResponse(BaseModel):
    template_suggestion: str
    topic: str
    time_limit_minutes: float  # Changed from int to float
    estimated_word_count: int
    slides: List[PresentationSlide]

class RedundancyAnalysisResponse(BaseModel):
    redundancy_score: int = Field(..., description="Redundancy score from 0 (Low) to 100 (High).")
    analysis_tip: str = Field(..., description="Actionable tip based on the score.")
    redundant_phrases_found: List[str] = Field(..., description="Specific phrases from the script that overlap with slide content.")
    slide_content_provided: List[str]
    
# Add these models after your existing models
class AnalysisFeedback(BaseModel):
    clarity_score: int
    overall_wpm: float
    filler_count: int
    filler_words_used: List[str]
    feedback: List[str]
    vague_phrases_found: List[str]

class OverallFeedbackResponse(BaseModel):
    total_sessions: int
    performance_summary: str
    improvement_areas: List[str]
    strengths: List[str]
    action_items: List[str]
    filler_word_analysis: Dict[str, Any]

# --- FastAPI Initialization (CRITICAL: Must be defined as 'app') ---
app = FastAPI(
    title="Presentation Coach API",
    description="Backend API for speech analysis and LLM-powered content critique.",
    version="1.3.1" # Updated version number
)

class TimeLimit(str, Enum):  # Change to str,Enum for better serialization
    thirty_seconds = "0.5"
    one_min = "1.0"
    two_min = "2.0"
    three_min = "3.0" 
    four_min = "4.0"
    five_min = "5.0"

ALLOWED_TIME_LIMITS = {float(t.value) for t in TimeLimit}

def generate_core_feedback(overall_wpm, filler_count, total_words, tagged_list, acoustic_metrics, pause_metrics, vague_phrases, relevance_score, suggested_content):
    """Generates actionable feedback based on ALL metrics (Delivery & Content)."""
    
    feedback = []
    target_range = analysis.TARGET_WPM_RANGE
    
    # FIX: Extract pause metrics from the dictionary
    strategic_pauses = pause_metrics['strategic']
    hesitation_gaps = pause_metrics['hesitation']
    # END FIX

    # 1. Delivery Feedback (Pacing, Fillers, Pauses)
    if overall_wpm > target_range[1]:
        feedback.append(f"⚠️ Pacing Alert: You spoke too fast ({round(overall_wpm)} WPM). Slow down.")
    elif overall_wpm < target_range[0]:
        feedback.append(f"⚠️ Pacing Alert: You spoke too slowly ({round(overall_wpm)} WPM). Increase energy.")
    else:
        feedback.append(f"✅ Pacing: Your overall speed ({round(overall_wpm)} WPM) is within range.")

    filler_rate = (filler_count / total_words) * 100 if total_words > 0 else 0
    if filler_rate > 2.0:
        feedback.append(f"🛑 Filler Warning: Used {filler_count} filler words ({round(filler_rate, 1)}%). Replace with pauses.")
    
    # 2. Pause Analysis Feedback (FIX APPLIED HERE)
    if strategic_pauses > 0:
        feedback.append(f"⭐ Pause Success: Found {strategic_pauses} strategic pauses. Use these more!")
    if hesitation_gaps > 0:
        feedback.append(f"🚨 Hesitation Warning: Detected {hesitation_gaps} hesitation gaps. Practice smooth transitions.")

    monotony = acoustic_metrics.pitch_monotony_score
    if monotony > 75:
        feedback.append(f"🛑 Monotony Alert: Your delivery is flat (Score {monotony}/100). Vary your tone and pitch.")
    if acoustic_metrics.avg_volume_status == "Quiet":
        feedback.append(f"📢 Volume Alert: Your volume was too quiet in sections. Project more.")
        
    if vague_phrases and isinstance(vague_phrases, list) and vague_phrases[0] != "Vague language analysis failed due to API error.":
         feedback.append(f"📉 Vague Language: Found {len(vague_phrases)} instances of vague words (e.g., '{vague_phrases[0]}'). Be precise.")

    # 3. LLM Rewriting Logic (Same as before)
    llm_rewrites = []
    filler_word_indices = [i for i, word in enumerate(tagged_list) if "filler" in word["tags"]]
    analyzed_indices = set()
    
    for i in filler_word_indices:
        if i not in analyzed_indices and len(llm_rewrites) < 3:
            start_index = max(0, i - 2)
            end_index = min(len(tagged_list), i + 3)
            segment_words = tagged_list[start_index:end_index]
            original_phrase = " ".join([w["text"] for w in segment_words])
            
            for j in range(start_index, end_index): analyzed_indices.add(j)
                
            rewrite = get_llm_rewrite(original_phrase)
            
            llm_rewrites.append(f"Original: \"{original_phrase.capitalize()}...\" -> Improvement: \"{rewrite.capitalize()}\"")

    if llm_rewrites:
        feedback.append("🧠 LLM Phrase Suggestions:")
        feedback.extend(llm_rewrites)
        
    # 4. Content Relevance Feedback (NEW MERGED LOGIC)
    if relevance_score is not None:
        feedback.append(f"\n🧠 **Content Relevance Score**: {relevance_score}/10")
        if relevance_score < 7:
            feedback.append("🛑 **Content Warning**: Your presentation may lack focus on the core topic. Review your structure.")
            
    if suggested_content and isinstance(suggested_content, list) and len(suggested_content) > 0 and suggested_content[0] != "Content analysis failed due to API error.":
        feedback.append("\n📚 **Recommended Content to Add**")
        for item in suggested_content:
            feedback.append(f"   - {item}")
            
    return feedback


# --- ENDPOINT 1 (MERGED): /analyze (Full Delivery and Content Analysis) ---
@app.post("/analyze", response_model=FullEvaluationResponse)
async def analyze_presentation(
    audio_file: Optional[UploadFile] = File(None),  # Changed to File(None)
    audio_url: Optional[str] = Form(None, description="URL to audio file (WAV format)"),
    topic: Optional[str] = Form(None, description="Optional topic for content relevance check")
):
    audio_bytes = None

    # Input validation
    if not audio_file and not audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either an audio file upload or a valid audio URL"
        )
    
    if audio_file and audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Please provide either an audio file OR an audio URL, not both"
        )

    try:
        if audio_url:
            # Handle URL input
            try:
                response = requests.get(audio_url, timeout=10)
                response.raise_for_status()
                audio_bytes = response.content
            except requests.exceptions.RequestException as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to download audio from URL: {str(e)}"
                )
        elif audio_file:
            # Handle file upload
            audio_bytes = await audio_file.read()
        else:
            raise HTTPException(
                status_code=400,
                detail="No audio source provided"
            )

        # 1. Transcription
        # Use the audio_bytes regardless of whether it came from a file upload or a URL download
        raw_text = analysis.call_whisper_api(audio_bytes) 
        if not raw_text:
            raise HTTPException(status_code=400, detail="Transcription failed to produce text.")
        
        # 2. Core Analysis Pipeline (Remains the same)
        word_list_data = analysis._simulate_whisper_output(raw_text)
        total_words = len(word_list_data)

        word_list_data, filler_count = analysis.detect_fillers(word_list_data)
        
        # After word_list_data, filler_count = analysis.detect_fillers(word_list_data)
        # Add this code to extract actual filler words:
        actual_fillers = []
        for word in word_list_data:
            if "filler" in word.get("tags", []):
                actual_fillers.append(word["text"])
        
        word_list_data, overall_wpm, wpm_map = analysis.analyze_pacing(word_list_data)
        word_list_data, strategic_pauses, hesitation_gaps = analysis.analyze_pauses(word_list_data)
        vague_phrases = analysis.analyze_vague_language(raw_text)
        
        acoustic_metrics_dict = analysis.simulate_acoustic_metrics(wpm_map)
        acoustic_metrics = AcousticMetrics(**acoustic_metrics_dict)
        
        clarity_score = analysis.calculate_clarity_score(overall_wpm, filler_count, total_words, acoustic_metrics_dict)
        
        # 3. Content Analysis (CONDITIONAL)
        relevance_score = None
        suggested_content = []
        
        if topic and topic.strip():
            relevance_score, suggested_content = analysis.analyze_topic_relevance(raw_text, topic)

        # 4. Feedback Generation
        pause_metrics = {'strategic': strategic_pauses, 'hesitation': hesitation_gaps}
        feedback_list = generate_core_feedback(
            overall_wpm, filler_count, total_words, word_list_data, 
            acoustic_metrics, pause_metrics, vague_phrases,
            relevance_score, suggested_content
        )

        # 5. Return Results
        return FullEvaluationResponse(
            clarity_score=clarity_score,
            overall_wpm=round(overall_wpm, 1),
            filler_count=filler_count,
            filler_words_used=actual_fillers,  # Add this line
            strategic_pauses=strategic_pauses,
            hesitation_gaps=hesitation_gaps,
            vague_phrases_found=vague_phrases,
            acoustic_metrics=acoustic_metrics,
            relevance_score=relevance_score,
            suggested_content=suggested_content,
            raw_transcription=raw_text,
            feedback=feedback_list,
            transcript=word_list_data
        )

    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Whisper API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error during analysis: {e}")

# --- ENDPOINT 2: /speech_draft (Draft Generation) ---
@app.post("/speech_draft", response_model=SpeechDraftResponse, summary="Generates a professional speech draft based on topic and time limit (Max 5 mins).")
async def speech_draft(
    topic: str = Form(..., description="The topic for the generated speech."),
    time_limit_minutes: TimeLimit = Form(..., description="Desired length of the speech in minutes. Choose from the dropdown."),
    tone: str = Form("professional", description="Tone for the speech: e.g., professional, casual, student")
):
    # Validate selected time limit against allowed options (dropdown)
    if float(time_limit_minutes.value) not in ALLOWED_TIME_LIMITS:
        raise HTTPException(status_code=400, detail=f"Invalid time limit. Allowed values: {sorted(ALLOWED_TIME_LIMITS)}")
    
    allowed_tones = ["professional", "casual", "student", "formal", "friendly"]
    tone = tone.strip().lower()
    if tone not in allowed_tones:
        raise HTTPException(status_code=400, detail=f"Invalid tone. Allowed tones: {', '.join(allowed_tones)}")
 
    # Approximate word count based on 150 WPM
    minutes_value = float(time_limit_minutes.value)
    estimated_word_count = int(round(minutes_value * 150))

    client = analysis._get_gemini_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini API service is unavailable. Check configuration.")

    system_instruction = (
        "You are an expert speechwriter and presentation coach. Your task is to generate a professional, "
        "compelling, and well-structured speech draft. The speech MUST be formatted with clear "
        "Markdown headings for easy reading and practice. Ensure the content logically flows "
        "through an Introduction, main Body points, and a powerful Conclusion."
    )
    
    user_prompt = (
        f"Generate a presentation draft on the following topic: '{topic}'. Use a {tone} tone for the voice and style. "
        f"The speech must be structured and approximately {minutes_value} minutes long, "
        f"which corresponds to about {estimated_word_count} words (assuming a moderate pace of 150 WPM). "
        f"Format the output using Markdown with sections labeled: # Title, ## Introduction, ### Main Points (with bullet lists), and ## Conclusion."
    )

    try:
        response = client.models.generate_content(
            model=analysis.GEMINI_MODEL,
            contents=[user_prompt],
            config=analysis.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=2048
            )
        )

        # Get the raw markdown text
        raw_speech = response.text.strip()
        
        # Parse the markdown into a flat array
        speech_parts = []
        lines = raw_speech.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):  # Title
                speech_parts.append(line[2:].strip())
            
            elif line.startswith('## '):  # Main sections
                speech_parts.append(line[3:].strip())
            
            elif line.startswith('### '):  # Sub-sections
                speech_parts.append(line[4:].strip())
            
            elif line.startswith('* ') or line.startswith('- '):  # Bullet points
                point = line[2:].strip()
                if point.startswith('**'):  # Handle bold text
                    point = point[2:-2].strip()  # Remove ** markers
                speech_parts.append(point)
            
            elif line:  # Regular text (non-empty)
                speech_parts.append(line)

        return SpeechDraftResponse(
            topic=topic,
            time_limit_minutes=minutes_value,
            estimated_word_count=estimated_word_count,
            generated_speech_draft=speech_parts,  # Now it's a simple array of strings
            tone=tone
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM content generation failed: {e}")


# --- ENDPOINT 3: /generate_ai_voiceover (Script & Audio Generation) ---
@app.post("/generate_ai_voiceover", response_model=AIVoiceoverResponse)
async def generate_ai_voiceover(
    topic: str = Form(..., description="The topic for the voiceover presentation."),
    time_limit_minutes: TimeLimit = Form(..., description="Desired length of the script in minutes.")
):
    try:
        minutes_value = float(time_limit_minutes.value)
        estimated_word_count = int(round(minutes_value * 150))

        # 1. Generate the script
        full_script = analysis.get_llm_presentation_script(topic, minutes_value)

        if full_script.startswith("Script generation failed"):
            raise HTTPException(status_code=500, detail=full_script)
            
        try:
            # 2. Generate temporary WAV file with robust error handling
            length_label = "30sec" if minutes_value == 0.5 else f"{int(minutes_value)}min"
            temp_filename = os.path.join(
                os.getenv('TEMP_AUDIO_PATH', '/app/temp_audio'),
                f"temp_voiceover_{topic.replace(' ', '_').lower()}_{length_label}.wav"
            )
            
            # Initialize engine with specific driver
            try:
                engine = pyttsx3.init(driverName='espeak')
            except:
                # Fallback to default initialization
                engine = pyttsx3.init()
            
            # Configure basic properties first
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Careful voice selection
            try:
                voices = engine.getProperty('voices')
                if voices:
                    # Try each voice until one works
                    for voice in voices:
                        try:
                            engine.setProperty('voice', voice.id)
                            # Test if voice works
                            engine.say("Test")
                            engine.runAndWait()
                            break
                        except:
                            continue
            except Exception as voice_err:
                print(f"Voice selection failed: {str(voice_err)}")
                # Continue with default voice
            
            # Save to file with error checking
            try:
                engine.save_to_file(full_script, temp_filename)
                engine.runAndWait()
                
                if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
                    raise Exception("Failed to generate audio file")
                
            except Exception as save_err:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Audio file generation failed: {str(save_err)}"
                )

            # 3. Upload to Cloudinary with error handling
            try:
                if not os.getenv('CLOUDINARY_CLOUD_NAME'):
                    raise HTTPException(status_code=500, detail="Cloudinary configuration missing")

                upload_result = cloudinary.uploader.upload(
                    temp_filename,
                    resource_type="raw",
                    folder="presentation_audio",
                    public_id=f"voiceover_{topic.replace(' ', '_').lower()}_{length_label}",
                    overwrite=True
                )
                
                audio_url = upload_result.get('secure_url')
                if not audio_url:
                    raise HTTPException(status_code=500, detail="Failed to get URL from Cloudinary upload")

            except Exception as cloud_err:
                raise HTTPException(status_code=500, detail=f"Cloudinary upload failed: {str(cloud_err)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

            # 4. Return response with Cloudinary URL
            return AIVoiceoverResponse(
                topic=topic,
                time_limit_minutes=minutes_value,
                estimated_word_count=estimated_word_count,
                full_voiceover_script=full_script,
                audio_file_url=audio_url
            )

        except Exception as e:
            # Clean up temp file in case of any error
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise HTTPException(status_code=500, detail=f"Voice generation failed: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voiceover generation failed: {str(e)}")


# --- NEW ENDPOINT 4: /presentation_content (Content, Outline, & Images) ---
@app.post("/presentation_content", response_model=PresentationContentResponse)
async def presentation_content(
    topic: str = Form(..., description="The topic for the presentation."),
    time_limit_minutes: TimeLimit = Form(..., description="Desired length of the presentation in minutes (0.5=30sec, 1-5 min)")
):
    try:
        # Convert string enum value to float for calculations
        minutes_value = float(time_limit_minutes.value)
        estimated_word_count = int(round(minutes_value * 150))

        # Check if Gemini client is available
        client = analysis._get_gemini_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Gemini API service is unavailable")

        # 1. Generate the structured content
        slide_data = analysis.get_llm_presentation_content(topic, minutes_value)
        
        if slide_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate presentation content")

        if isinstance(slide_data, dict) and 'error' in slide_data:
            raise HTTPException(status_code=500, detail=slide_data['error'])
        
        # Handle response format
        if isinstance(slide_data, dict) and 'slides' in slide_data:
            template_suggestion = slide_data.get('template_suggestion', "")
            slides_raw = slide_data['slides']
        elif isinstance(slide_data, list):
            template_suggestion = ""
            slides_raw = slide_data
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Unexpected presentation content format. Got type: {type(slide_data)}"
            )
        
        # Validate slide structure
        try:
            validated_slides = [PresentationSlide(**slide) for slide in slides_raw]
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Invalid slide structure: {str(e)}"
            )

        return PresentationContentResponse(
            template_suggestion=template_suggestion,
            topic=topic,
            time_limit_minutes=minutes_value,
            estimated_word_count=estimated_word_count,
            slides=validated_slides
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Presentation content generation failed: {str(e)}"
        )

# --- ENDPOINT 8: /check_redundancy (Script-Slide Overlap) ---
@app.post("/check_redundancy", response_model=RedundancyAnalysisResponse, summary="Checks the speaker script against slide content for excessive redundancy.")
async def check_redundancy(
    script: str = Form(..., description="The full, intended speaker script/notes."),
    slide_bullets: List[str] = Form(..., description="The list of bullet points or main texts intended for the slides.")
):
    
    # 1. Generate redundancy analysis using the new analysis function
    redundancy_data = analysis.analyze_script_redundancy(script, slide_bullets)

    if 'error' in redundancy_data:
        raise HTTPException(status_code=500, detail=redundancy_data['error'])
    
    score = redundancy_data.get("redundancy_score", 50)
    
    # 2. Generate actionable tip based on the score
    if score < 20:
        tip = "✅ Excellent separation! Your script elaborates well on your slides. Keep your slide text minimal."
    elif score < 50:
        tip = "⚠️ Moderate Redundancy. Review the flagged phrases and use different language in your script than what is visible on the slide."
    else:
        tip = "🛑 High Redundancy. You are likely reading directly from your slides. Your script should be complementary, not identical, to the visual content. Restructure your notes."

    # 3. Return the structured content
    return RedundancyAnalysisResponse(
        redundancy_score=score,
        analysis_tip=tip,
        redundant_phrases_found=redundancy_data.get("redundant_phrases", []),
        slide_content_provided=slide_bullets
    )

# First, update the models
class FeedbackInput(BaseModel):
    clarity_score: int
    overall_wpm: float
    filler_count: int
    filler_words_used: List[str]
    feedback: List[str]
    vague_phrases_found: List[str]

class OverallFeedbackRequest(BaseModel):
    feedbacks: List[FeedbackInput]

@app.post("/overall_feedback", response_model=OverallFeedbackResponse)
async def overall_feedback(
    feedbacks: List[FeedbackInput] = Body(..., embed=True)  # Changed this line
):
    try:
        if len(feedbacks) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 sessions for meaningful analysis. Received {len(feedbacks)}"
            )
        
        # Prepare data for Gemini
        feedback_summary = {
            "metrics": {
                "avg_clarity": sum(f.clarity_score for f in feedbacks) / len(feedbacks),
                "avg_wpm": sum(f.overall_wpm for f in feedbacks) / len(feedbacks),
                "total_fillers": sum(f.filler_count for f in feedbacks),
            },
            "all_feedback": [item for f in feedbacks for item in f.feedback],
            "all_fillers": [word for f in feedbacks for word in f.filler_words_used],
            "all_vague_phrases": [phrase for f in feedbacks for phrase in f.vague_phrases_found]
        }

        client = analysis._get_gemini_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Gemini API service is unavailable")

        system_instruction = """
        You are an expert speech coach analyzing multiple presentation sessions.
        Provide a comprehensive analysis focusing on:
        1. Overall performance trends
        2. Specific areas needing improvement
        3. Notable strengths
        4. Actionable recommendations
        Format the response as JSON with these exact keys:
        {
            "performance_summary": "overall analysis",
            "improvement_areas": ["area1", "area2"...],
            "strengths": ["strength1", "strength2"...],
            "action_items": ["action1", "action2"...]
        }
        """

        prompt = f"""
        Analyze these presentation sessions:
        - Average Clarity Score: {feedback_summary['metrics']['avg_clarity']}/100
        - Average WPM: {feedback_summary['metrics']['avg_wpm']}
        - Total Filler Words: {feedback_summary['metrics']['total_fillers']}
        
        Common Feedback Points:
        {feedback_summary['all_feedback']}
        
        Frequently Used Filler Words:
        {feedback_summary['all_fillers']}
        
        Vague Phrases Used:
        {feedback_summary['all_vague_phrases']}
        
        Provide a structured analysis as JSON focusing on patterns, improvements needed, and specific recommendations.
        """

        response = client.models.generate_content(
            model=analysis.GEMINI_MODEL,
            contents=[prompt],
            config=analysis.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                response_format="json"
            )
        )

        analysis_result = response.json()
        
        # Calculate filler word frequencies
        from collections import Counter
        filler_analysis = {
            "most_common": Counter(feedback_summary['all_fillers']).most_common(5),
            "total_count": len(feedback_summary['all_fillers']),
            "unique_count": len(set(feedback_summary['all_fillers']))
        }

        return OverallFeedbackResponse(
            total_sessions=len(feedbacks),
            performance_summary=analysis_result["performance_summary"],
            improvement_areas=analysis_result["improvement_areas"],
            strengths=analysis_result["strengths"],
            action_items=analysis_result["action_items"],
            filler_word_analysis=filler_analysis
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate overall feedback: {str(e)}"
        )

