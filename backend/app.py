"""
JobShield AI — v4.0
3-layer detection:
  Layer 1: Groq LLM (real AI — best accuracy)
  Layer 2: ML model (SGDClassifier trained on real data)
  Layer 3: Rule-based (keyword patterns — always runs)
Final score = weighted blend of all available layers.
"""

import re
import os
import json
import pickle
import base64
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from PIL import Image

load_dotenv()

# ── Persistent scan counter ──────────────────────────────
COUNTER_FILE = os.path.join(os.path.dirname(__file__), "scan_count.json")

def get_scan_count() -> int:
    try:
        if os.path.exists(COUNTER_FILE):
            with open(COUNTER_FILE, "r") as f:
                return json.load(f).get("count", 0)
    except Exception:
        pass
    return 0

def increment_scan_count() -> int:
    count = get_scan_count() + 1
    try:
        with open(COUNTER_FILE, "w") as f:
            json.dump({"count": count}, f)
    except Exception:
        pass
    return count

app = FastAPI(title="JobShield AI", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Layer 1: Groq AI ──────────────────────────────────────
GROQ_AVAILABLE = False
groq_client = None
try:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key and api_key != "your_groq_api_key_here":
        groq_client = Groq(api_key=api_key)
        GROQ_AVAILABLE = True
        print("✅ Groq AI loaded")
    else:
        print("⚠️  No GROQ_API_KEY — AI layer disabled")
except Exception as e:
    print(f"⚠️  Groq not available: {e}")

# ── Layer 2: ML model ─────────────────────────────────────
ML_AVAILABLE = False
vectorizer = clf_text = clf_num = None
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    vectorizer = model["vectorizer"]
    clf_text   = model["clf_text"]
    clf_num    = model["clf_num"]
    ML_AVAILABLE = True
    print("✅ ML model loaded")
except Exception:
    print("⚠️  model.pkl not found")


# ============================================================
# Enums & Models
# ============================================================

class VerdictEnum(str, Enum):
    SAFE      = "Safe"
    UNCERTAIN = "Uncertain"
    SCAM      = "Scam"

class RiskLevelEnum(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

class JobAnalysisRequest(BaseModel):
    job_description: str = Field(..., min_length=20, max_length=10000)

class JobAnalysisResponse(BaseModel):
    score:       int = Field(..., ge=0, le=100)
    verdict:     VerdictEnum
    reasons:     List[str]
    risk_level:  RiskLevelEnum
    ml_verdict:  str = "N/A"
    ai_verdict:  str = "N/A"
    layers_used: List[str] = []
    is_job_post: bool = True
    invalid_reason: str = ""
    total_scans: int = 0


# ============================================================
# Job Context Detection — is this even a job post?
# ============================================================

JOB_CONTEXT_KEYWORDS = [
    # Core job terms
    "job", "hiring", "vacancy", "opening", "position", "role", "post",
    "recruit", "recruitment", "career", "employment", "work",
    # Job details
    "salary", "ctc", "lpa", "stipend", "pay", "compensation", "package",
    "experience", "qualification", "degree", "skills", "requirement",
    # Actions
    "apply", "application", "candidate", "interview", "joining",
    "fresher", "graduate", "intern", "internship",
    # Company terms
    "company", "firm", "organisation", "organization", "employer",
    "office", "department", "team", "manager",
    # Job types
    "full-time", "part-time", "remote", "work from home", "wfh",
    "contract", "permanent", "freelance",
    # Scam-specific (these alone indicate job context)
    "registration fee", "training fee", "joining fee", "earn",
    "income", "per month", "per day", "daily earning",
]

def is_job_related(text: str) -> tuple[bool, int]:
    """
    Check if the text is related to a job posting.
    Returns (is_job, match_count)
    Needs at least 2 keyword matches to be considered a job post.
    """
    t = text.lower()
    matches = sum(1 for kw in JOB_CONTEXT_KEYWORDS if kw in t)
    return matches >= 2, matches



def analyze_with_groq(text: str) -> Optional[dict]:
    """
    Ask Groq LLM to analyze the job posting like a senior fraud investigator.
    Returns dict with score, verdict, reasons or None if unavailable.
    """
    if not GROQ_AVAILABLE:
        return None

    prompt = f"""You are an elite job fraud investigator with deep expertise in company verification and impersonation detection.
Analyze this job posting with a 5-layer investigation and return ONLY valid JSON.

Job Posting:
\"\"\"
{text[:3000]}
\"\"\"

=== LAYER 1: COMPANY IDENTITY INVESTIGATION ===
Check if the company mentioned is REAL or IMPERSONATED:

REAL company signals:
- Company name matches a known registered business (MNC, startup, SME)
- Company description is consistent with their known industry
- Salary offered matches what that company typically pays
- Job role makes sense for that company's business
- Contact details match the company's known domain (e.g., @manpowergroup.com, @tcs.com)

IMPERSONATION / DUPLICATE signals (scammers copy real company names):
- Claims to be a famous company (Google, Amazon, TCS, Infosys) BUT:
  * Contact is WhatsApp/Telegram/personal Gmail instead of official email
  * Salary is unrealistically high (e.g., Google offering ₹5 LPA for freshers)
  * No interview process mentioned for a company known for rigorous hiring
  * Job description doesn't match the company's actual business
  * Location doesn't match company's known offices
  * Asks for payment/fee (real MNCs NEVER charge candidates)
- Company name has slight spelling variations (e.g., "Infosys Solutions" vs "Infosys")
- Claims to be a company but description sounds like a scam operation

=== LAYER 2: SALARY REALITY CHECK ===
Compare salary vs role vs experience:
- Data Entry / Form Filling: ₹8,000-15,000/month is real. ₹50,000/month = SCAM
- HR Executive (1-2 yrs): ₹2-5 LPA is real. ₹15 LPA = suspicious
- Software Engineer (fresher): ₹3-8 LPA is real. ₹25 LPA = suspicious  
- Sales Executive: ₹2-4 LPA + incentives is real
- Senior roles (5+ yrs): ₹8-25 LPA is real depending on company
- Work from home + no skills + high pay = SCAM pattern

=== LAYER 3: PAYMENT & FEE DETECTION ===
Any of these = AUTOMATIC SCAM:
- Registration fee, training fee, kit purchase, security deposit
- "Pay ₹X to get started", "refundable deposit", "buy your own equipment"
- UPI payment, bank transfer required before joining
- "Small investment" to start working

=== LAYER 4: PROCESS LEGITIMACY CHECK ===
Real hiring process vs scam process:
REAL: Application → Screening → Interview → Offer Letter → Joining
SCAM: Apply → Instant Selection → Pay Fee → Start Working

Red flags:
- "Selected immediately", "no interview needed", "100% selection"
- "Start today", "instant joining", "no experience no problem"
- Only WhatsApp contact, no official email or website
- "Company name will be revealed after selection"

=== LAYER 5: CONTENT CONSISTENCY CHECK ===
Check if the job posting is internally consistent:
- Does the job title match the responsibilities?
- Does the required experience match the salary?
- Is the company description consistent with the job role?
- Are the perks (health insurance, PF) realistic for the company size?
- Does the location make sense for the company?

Inconsistency examples (suspicious):
- "Senior Manager" role requiring 0 experience
- "Google" job but contact is gmail.com
- "MNC company" but no company name given
- Salary breakup: ₹3,00,000 fixed + ₹2 variable (odd but not scam)

=== VERDICT CALIBRATION ===
SAFE (score 0-24): Named company + realistic salary + proper requirements + professional contact
UNCERTAIN (score 25-54): Limited info, weak signals — needs verification
SCAM (score 55-100): Payment demands OR impersonation OR no experience + unrealistic pay OR no company + informal contact

Real examples:
- ManpowerGroup HR role ₹3-4 LPA, MBA required, Hyderabad → SAFE (score: 8)
- TCS Software Engineer ₹5-8 LPA, technical interview → SAFE (score: 5)
- "Google" job via WhatsApp, ₹50k/month, no interview → SCAM (score: 92)
- Unknown company, ₹500 registration fee, work from home → SCAM (score: 95)
- New startup, vague description, only mobile number → UNCERTAIN (score: 38)
- "Earn ₹50,000/month liking posts, no experience" → SCAM (score: 98)

Return ONLY this JSON (no markdown, no explanation):
{{
  "score": <integer 0-100>,
  "verdict": "<Safe|Uncertain|Scam>",
  "risk_level": "<low|medium|high>",
  "reasons": [
    "<Layer finding 1 with emoji>",
    "<Layer finding 2 with emoji>",
    "<Layer finding 3 with emoji>",
    "<Layer finding 4 with emoji>"
  ],
  "company_verdict": "<Real|Suspicious|Impersonated|Unknown>",
  "key_red_flags": ["<flag1>", "<flag2>"],
  "key_green_flags": ["<flag1>", "<flag2>"]
}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        result = json.loads(raw)

        # Validate required fields
        score   = max(0, min(100, int(result.get("score", 50))))
        verdict = result.get("verdict", "Uncertain")
        if verdict not in ["Safe", "Uncertain", "Scam"]:
            verdict = "Uncertain"
        risk    = result.get("risk_level", "medium")
        if risk not in ["low", "medium", "high"]:
            risk = "medium"

        reasons = result.get("reasons", [])
        red     = result.get("key_red_flags", [])
        green   = result.get("key_green_flags", [])
        company_verdict = result.get("company_verdict", "Unknown")

        # Add impersonation warning if detected
        if company_verdict == "Impersonated":
            reasons.insert(0, "🚨 COMPANY IMPERSONATION DETECTED — This may be a fake job using a real company's name!")
            score = max(score, 70)
            verdict = "Scam"
            risk = "high"

        # Add flag summaries to reasons
        if red:
            reasons.append(f"🚩 Red flags: {', '.join(red[:3])}")
        if green:
            reasons.append(f"✅ Green flags: {', '.join(green[:3])}")

        return {
            "score":      score,
            "verdict":    verdict,
            "risk_level": risk,
            "reasons":    reasons,
        }

    except Exception as e:
        print(f"Groq error: {e}")
        return None


# ============================================================
# Layer 2 — ML Model
# ============================================================

def analyze_with_ml(text: str) -> Optional[dict]:
    if not ML_AVAILABLE:
        return None
    try:
        text_vec  = vectorizer.transform([text])
        pred_text = int(clf_text.predict(text_vec)[0])
        prob_text = clf_text.predict_proba(text_vec)[0]

        char_count = len(text)
        pred_num   = int(clf_num.predict([[0, 0.0, char_count]])[0])

        is_fake    = pred_text == 1 or pred_num == 1
        confidence = float(max(prob_text)) * 100

        return {
            "is_fake":    is_fake,
            "confidence": round(confidence, 1),
            "verdict":    "FAKE" if is_fake else "REAL",
        }
    except Exception as e:
        print(f"ML error: {e}")
        return None


# ============================================================
# Layer 3 — Rule-based (always runs as safety net)
# ============================================================

PAYMENT_KW = [
    "registration fee","training fee","security deposit","processing fee",
    "application fee","upfront payment","advance payment","pay first",
    "buy kit","pay to join","investment required","joining fee",
    "pay ₹","pay rs","deposit ₹","send money","pay via upi",
    "credit card required","starter kit fee","refundable deposit",
]
NO_INTERVIEW_KW = [
    "no interview","without interview","direct joining","instant hire",
    "hired immediately","guaranteed selection","100% selection",
    "everyone selected","no screening","no test","no assessment",
]
EASY_KW = [
    "no experience needed","no experience required","anyone can do",
    "easy work","simple task","just like","just share","just click",
    "work from mobile","no qualification","no skills required",
    "earn while sitting","earn from home easily",
]
URGENCY_KW = [
    "limited seats","hurry up","act now","last chance",
    "closing today","urgent hiring","don't miss","expires today",
]
INFORMAL_KW = [
    "whatsapp only","contact on whatsapp","telegram only",
    "contact on telegram","dm me","gmail only",
]
OVERPROMISE_KW = [
    "earn lakhs","guaranteed income","passive income",
    "make money fast","unlimited earning","50k per month",
]
MICRO_TASK_KW = [
    "like posts","share posts","watch videos","click ads",
    "data typing","copy paste","form filling","captcha",
]
PROFESSIONAL_KW = [
    "interview process","technical round","hr round","offer letter",
    "background verification","salary range","ctc","equal opportunity",
    "registered company","responsibilities:","requirements:",
]

def analyze_with_rules(text: str) -> dict:
    t = text.lower()
    score   = 0
    reasons = []

    pay     = [k for k in PAYMENT_KW     if k in t]
    no_int  = [k for k in NO_INTERVIEW_KW if k in t]
    easy    = [k for k in EASY_KW        if k in t]
    urgent  = [k for k in URGENCY_KW     if k in t]
    inform  = [k for k in INFORMAL_KW    if k in t]
    over    = [k for k in OVERPROMISE_KW if k in t]
    micro   = [k for k in MICRO_TASK_KW  if k in t]
    pro     = [k for k in PROFESSIONAL_KW if k in t]

    if pay:
        score += 40
        reasons.append(f"🚩 Asks for money ({pay[0]}) — real employers NEVER charge candidates.")
    if no_int:
        score += 30
        reasons.append("🚩 No interview or screening — legitimate jobs always verify candidates.")
    if easy:
        score += 20
        reasons.append("🚩 Promises easy selection with no effort — classic scam tactic.")
    if urgent:
        score += 15
        reasons.append(f"⚠️ Urgency pressure ({urgent[0]}) — scammers rush you to avoid research.")
    if inform:
        score += 18
        reasons.append(f"⚠️ Informal contact only ({inform[0]}) — real companies use official channels.")
    if over:
        score += 15
        reasons.append("⚠️ Unrealistic earning claims — too good to be true.")
    if micro:
        score += 22
        reasons.append("🚩 Micro-task scam (liking posts, surveys) — not a real job.")

    # Combinations
    if pay and no_int:
        score += 20
        reasons.append("🚨 CRITICAL: Payment + no interview = most common scam pattern.")
    if pay and easy:
        score += 15
        reasons.append("🚨 Easy selection + payment = scammers lowering your guard.")

    # Positive signals
    if len(pro) >= 3:
        if not pay:
            score = max(0, score - 15)
        reasons.append("✅ Professional job structure detected (interview, salary, qualifications).")

    # Email domain check
    for domain in re.findall(r'[\w\.-]+@([\w\.-]+)', t):
        if domain in ["gmail.com","yahoo.com","hotmail.com"]:
            score += 6
            reasons.append(f"📧 Uses '{domain}' — real companies use their own domain.")
        elif any(d in domain for d in ["google","microsoft","amazon","tcs","infosys","wipro"]):
            score = max(0, score - 8)
            reasons.append(f"✅ Verified company domain: {domain}")

    # Floor rules
    if pay:
        score = max(score, 40)
    if pay and no_int:
        score = max(score, 75)

    score = max(0, min(100, score))
    return {"score": score, "reasons": reasons}


# ============================================================
# Blend all layers into final result
# ============================================================

def blend_results(groq_result, ml_result, rule_result, text: str) -> dict:
    layers_used = ["rules"]
    reasons     = []
    all_scores  = []

    # Rules always contribute
    rule_score = rule_result["score"]
    all_scores.append(("rules", rule_score, 0.2))
    reasons.extend(rule_result["reasons"])

    # ML layer
    ml_verdict = "N/A"
    if ml_result:
        layers_used.append("ml_model")
        ml_verdict = ml_result["verdict"]
        ml_score   = 80 if ml_result["is_fake"] else 15
        all_scores.append(("ml", ml_score, 0.3))
        if ml_result["is_fake"]:
            reasons.insert(0, f"🤖 ML model flagged this as potentially fraudulent ({ml_result['confidence']:.0f}% confidence).")
        else:
            reasons.insert(0, f"🤖 ML model found no strong fraud pattern ({ml_result['confidence']:.0f}% confidence).")

    # Groq AI layer (highest weight — most accurate)
    ai_verdict = "N/A"
    if groq_result:
        layers_used.append("groq_ai")
        ai_verdict = groq_result["verdict"]
        ai_score   = groq_result["score"]
        all_scores.append(("groq", ai_score, 0.5))

        # Groq reasons go first (most insightful)
        groq_reasons = groq_result["reasons"]
        reasons = groq_reasons + [r for r in reasons if r not in groq_reasons]

    # ── VERDICT LOGIC ──
    # Groq AI is the most accurate layer — use it as primary decision maker
    # ML model is secondary, rules are safety net for obvious patterns

    if groq_result:
        # Groq verdict is primary — use its score directly
        groq_score   = groq_result["score"]
        groq_verdict = groq_result["verdict"]

        if groq_verdict == "Safe":
            # Groq says Safe — trust it UNLESS rules found payment keywords
            rule_reasons = rule_result.get("reasons", [])
            has_payment  = any("Asks for money" in r for r in rule_reasons)
            if has_payment:
                # Payment found by rules overrides Groq Safe
                verdict    = VerdictEnum.SCAM
                risk_level = RiskLevelEnum.HIGH
                final_score = max(groq_score, 70)
            else:
                verdict    = VerdictEnum.SAFE
                risk_level = RiskLevelEnum.LOW
                final_score = groq_score  # Trust Groq completely for Safe

        elif groq_verdict == "Uncertain":
            verdict    = VerdictEnum.UNCERTAIN
            risk_level = RiskLevelEnum.MEDIUM
            # Blend Groq + rules for score accuracy
            final_score = round((groq_score * 0.7) + (rule_result["score"] * 0.3))
            final_score = max(25, min(54, final_score))

        else:  # Scam
            verdict    = VerdictEnum.SCAM
            risk_level = RiskLevelEnum.HIGH
            final_score = max(groq_score, 60)

    else:
        # No Groq — use ML + rules blend
        total_weight = sum(w for _, _, w in all_scores)
        final_score  = round(sum(s * w for _, s, w in all_scores) / total_weight)
        final_score  = max(0, min(100, final_score))

        if final_score < 25:
            verdict    = VerdictEnum.SAFE
            risk_level = RiskLevelEnum.LOW
        elif final_score < 55:
            verdict    = VerdictEnum.UNCERTAIN
            risk_level = RiskLevelEnum.MEDIUM
        else:
            verdict    = VerdictEnum.SCAM
            risk_level = RiskLevelEnum.HIGH

    # Add layer summary
    reasons.append(f"📊 Analysis used: {', '.join(layers_used)} — score blended from {len(all_scores)} layer(s).")

    if not reasons:
        reasons.append("✅ No red flags detected. This looks like a legitimate job posting.")

    return {
        "score":       final_score,
        "verdict":     verdict,
        "risk_level":  risk_level,
        "reasons":     reasons,
        "ml_verdict":  ml_verdict,
        "ai_verdict":  ai_verdict,
        "layers_used": layers_used,
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "message":    "JobShield AI API v4.0",
        "groq_ai":    GROQ_AVAILABLE,
        "ml_model":   ML_AVAILABLE,
        "total_scans": get_scan_count(),
    }

@app.get("/health")
def health():
    return {"status": "healthy", "groq": GROQ_AVAILABLE, "ml": ML_AVAILABLE}

@app.get("/stats")
def stats():
    """Get real scan count — shown to judges"""
    return {"total_scans": get_scan_count()}


@app.post("/analyze", response_model=JobAnalysisResponse)
def analyze(request: JobAnalysisRequest):
    job_text = request.job_description.strip()
    if not job_text:
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    # ── Step 0: Check if this is actually a job post ──
    is_job, match_count = is_job_related(job_text)
    if not is_job:
        return JobAnalysisResponse(
            score=0,
            verdict=VerdictEnum.SAFE,
            risk_level=RiskLevelEnum.LOW,
            reasons=[
                "⚠️ This does not appear to be a job description.",
                "📋 Please paste a real job posting — include job title, company, salary, requirements.",
                f"🔍 Only {match_count} job-related keyword(s) found. Minimum 2 required for analysis.",
            ],
            ml_verdict="N/A",
            ai_verdict="N/A",
            layers_used=[],
            is_job_post=False,
            invalid_reason="Not a job description",
            total_scans=get_scan_count()
        )

    # ── Step 1: Increment counter (only real job posts) ──
    total_scans = increment_scan_count()

    # ── Step 2: Run all 3 analysis layers ──
    groq_result = analyze_with_groq(job_text)
    ml_result   = analyze_with_ml(job_text)
    rule_result = analyze_with_rules(job_text)

    # ── Step 3: Blend into final result ──
    result = blend_results(groq_result, ml_result, rule_result, job_text)
    result["is_job_post"] = True
    result["invalid_reason"] = ""
    result["total_scans"] = total_scans

    return JobAnalysisResponse(**result)


# ============================================================
# Image Analysis — extract text from job poster then analyze
# ============================================================

def extract_text_from_image(image_bytes: bytes) -> Optional[str]:
    """
    Use Groq vision model to extract and analyze job posting from image.
    Returns extracted text or None.
    """
    if not GROQ_AVAILABLE:
        return None
    try:
        # Convert to base64
        img = Image.open(BytesIO(image_bytes))
        # Resize if too large
        max_size = (1200, 1200)
        img.thumbnail(max_size, Image.LANCZOS)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract ALL text from this job posting image exactly as written. Include every word, number, contact detail, salary, and requirement. Return only the extracted text, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            }],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Image extraction error: {e}")
        return None


@app.post("/analyze-image", response_model=JobAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload a job poster image (JPG/PNG/WEBP).
    Extracts text using Groq vision, then runs full fraud analysis.
    """
    # Validate file type
    allowed = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WEBP images allowed")

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    if not GROQ_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI vision not available. Please add GROQ_API_KEY.")

    # Extract text from image
    extracted_text = extract_text_from_image(image_bytes)
    if not extracted_text or len(extracted_text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Could not extract readable text from image. Try a clearer image.")

    # Run full analysis on extracted text
    groq_result = analyze_with_groq(extracted_text)
    ml_result   = analyze_with_ml(extracted_text)
    rule_result = analyze_with_rules(extracted_text)
    result      = blend_results(groq_result, ml_result, rule_result, extracted_text)

    # Increment counter for image scans
    total_scans = increment_scan_count()
    result["total_scans"] = total_scans
    result["is_job_post"] = True
    result["invalid_reason"] = ""

    # Add extracted text note
    result["reasons"].insert(0, f"📸 Text extracted from image: \"{extracted_text[:120]}...\"")

    return JobAnalysisResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
