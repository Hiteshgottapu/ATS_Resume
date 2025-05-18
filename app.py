from flask import Flask, render_template, request, jsonify
import re
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as date_parse

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from fuzzywuzzy import fuzz
from thinc.api import prefer_gpu, require_gpu, require_cpu  # noqa: F401

app = Flask(__name__)

# --- NLP Setup ---
# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# --- Configuration & Constants (mostly same, but ALL_KNOWN_SKILLS will be used by PhraseMatcher) ---
KNOWN_PROGRAMMING_LANGUAGES = [
    "python", "java", "javascript", "c#", "c++", "php", "ruby", "swift", "kotlin",
    "typescript", "go", "rust", "scala", "perl", "objective-c", "sql", "html", "css"
]
KNOWN_FRAMEWORKS_LIBRARIES = [
    "react", "angular", "vue", "vue.js", "django", "flask", "spring", "spring boot", ".net", "node.js",
    "express.js", "ruby on rails", "laravel", "symfony", "jquery", "bootstrap",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "redux", "next.js"
]
KNOWN_TOOLS_PLATFORMS = [
    "git", "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "jira", "confluence",
    "linux", "windows", "macos", "mysql", "postgresql", "mongodb", "redis", "kafka",
    "elasticsearch", "terraform", "ansible", "ci/cd", "selenium", "junit", "postman", "figma"
]
ALL_KNOWN_SKILLS_ORIGINAL_CASE = list(set(KNOWN_PROGRAMMING_LANGUAGES + KNOWN_FRAMEWORKS_LIBRARIES + KNOWN_TOOLS_PLATFORMS))
# For PhraseMatcher, it's better to use lowercase for matching flexibility
ALL_KNOWN_SKILLS_LOWER = [skill.lower() for skill in ALL_KNOWN_SKILLS_ORIGINAL_CASE]


DEGREE_KEYWORDS_LOWER = [
    "b.s", "bs", "b.sc", "bsc", "bachelor", "bachelor's", "bachelors",
    "m.s", "ms", "m.sc", "msc", "master", "master's", "masters",
    "ph.d", "phd", "doctorate", "dphil",
    "associate", "diploma", "certificate", "certification"
]
# For normalizing skill names (e.g., js -> javascript)
SKILL_SYNONYMS = {
    "js": "javascript", "node": "node.js", "reactjs": "react", "angularjs": "angular",
    "postgres": "postgresql", "k8s": "kubernetes", "amazon web services": "aws",
    "google cloud platform": "gcp", "microsoft azure": "azure", "c sharp": "c#",
    "c plus plus": "c++"
}

# --- NLP Enhanced Helper Functions ---

def normalize_skill(skill_text):
    """Normalizes a skill string to a canonical form."""
    skill_lower = skill_text.lower().strip()
    # Replace common punctuation or joiners if they are part of the skill but might vary
    skill_lower = skill_lower.replace('-', ' ').replace('.', '') # e.g. node.js -> node js
    # Check synonyms
    normalized = SKILL_SYNONYMS.get(skill_lower, skill_lower)
    # Return the original case version if a lowercase match was found in our known list
    for original_skill in ALL_KNOWN_SKILLS_ORIGINAL_CASE:
        if original_skill.lower() == normalized:
            return original_skill
    return normalized # Fallback to the normalized lowercase if not in original case list

def extract_skills_nlp(text):
    doc = nlp(text)
    found_skills = set()
    related_skills = {}

    # Use PhraseMatcher with attr='LOWER'
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill.lower()) for skill in ALL_KNOWN_SKILLS_ORIGINAL_CASE]
    phrase_matcher.add("SKILL", patterns)
    matches = phrase_matcher(doc)
    for _, start, end in matches:
        skill_span = doc[start:end]
        found_skills.add(normalize_skill(skill_span.text))

    # 2. NER for PRODUCT/ORG/WORK_OF_ART (spaCy 3.x+)
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "WORK_OF_ART"]:
            ent_text = ent.text.strip()
            for known_skill in ALL_KNOWN_SKILLS_ORIGINAL_CASE:
                if fuzz.ratio(ent_text.lower(), known_skill.lower()) > 85:
                    found_skills.add(known_skill)
                    break

    # 3. Fuzzy match for possible skills in the text (single/multi-word tokens)
    tokens = set([t.text.lower() for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 2])
    for token in tokens:
        for known_skill in ALL_KNOWN_SKILLS_ORIGINAL_CASE:
            if fuzz.ratio(token, known_skill.lower()) > 90:
                found_skills.add(known_skill)

    # 4. Regex for explicitly listed skills (improved: more robust section extraction)
    skills_section_match = re.search(
        r"(?:skills|technologies|technical\s+skills|proficiencies|expertise)[\s:]*((?:.|\n)*?)(?:\n\n|\n[A-Z][a-z ()]+:|$)",
        text, re.IGNORECASE | re.MULTILINE
    )
    if skills_section_match:
        skills_text = skills_section_match.group(1)
        # Split by common delimiters and also handle bullets
        potential_skills = re.split(r'[,;/|\n•*-]+', skills_text)
        for ps in potential_skills:
            ps_clean = ps.strip()
            if ps_clean and len(ps_clean) > 1:
                # Fuzzy match for each listed skill
                for known_skill in ALL_KNOWN_SKILLS_ORIGINAL_CASE:
                    if fuzz.ratio(ps_clean.lower(), known_skill.lower()) > 85:
                        found_skills.add(known_skill)
                        break

    # 5. Additional: extract capitalized multi-word phrases (likely skills)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if chunk_text.istitle() and 2 <= len(chunk_text.split()) <= 4:
            for known_skill in ALL_KNOWN_SKILLS_ORIGINAL_CASE:
                if fuzz.ratio(chunk_text.lower(), known_skill.lower()) > 85:
                    found_skills.add(known_skill)
                    break

    # --- Identify related skills for missing skills ---
    # Example: If "python" is missing, but "django" or "flask" is present, suggest "python" is related.
    # Build a reverse mapping: for each known skill, what other skills are related?
    RELATED_SKILL_MAP = {
        # Technical Skills
        "python": ["django", "flask", "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "pytorch"],
        "javascript": ["react", "angular", "vue", "node.js", "express.js", "next.js", "jquery", "typescript"],
        "java": ["spring", "spring boot", "junit", "maven", "hibernate"],
        "c#": [".net", "asp.net", "entity framework", "visual studio", "microsoft"],
        "c++": ["qt", "boost"],
        "php": ["laravel", "symfony"],
        "ruby": ["ruby on rails"],
        "go": ["gin", "beego"],
        "swift": ["ios", "xcode"],
        "kotlin": ["android"],
        "typescript": ["angular", "react", "vue"],
        "sql": ["mysql", "postgresql", "sqlite", "database", "mssql", "oracle"],
        "mysql": ["sql", "database"],
        "postgresql": ["sql", "database"],
        "mongodb": ["nosql", "database"],
        "aws": ["terraform", "cloud", "devops", "lambda", "ec2", "s3", "cloudformation"],
        "azure": ["cloud", "devops", "azure devops", "azure functions", "microsoft", "microsoft azure"],
        "gcp": ["cloud", "google cloud platform", "bigquery"],
        "docker": ["kubernetes", "ci/cd", "containers"],
        "kubernetes": ["docker", "containers", "devops"],
        "linux": ["bash", "shell", "ubuntu", "centos", "redhat"],
        "windows": ["powershell", "dotnet", "microsoft"],
        "macos": ["swift", "xcode"],
        "ci/cd": ["jenkins", "github actions", "gitlab ci", "azure pipelines"],
        "jenkins": ["ci/cd", "devops"],
        "terraform": ["aws", "azure", "gcp", "infrastructure as code"],
        "ansible": ["automation", "devops"],
        "selenium": ["automation", "testing"],
        "junit": ["java", "testing"],
        "postman": ["api", "testing"],
        "figma": ["ui/ux", "design"],
        "redux": ["react", "javascript"],
        "pandas": ["python", "data analysis"],
        "numpy": ["python", "data science"],
        "scikit-learn": ["python", "machine learning"],
        "tensorflow": ["python", "machine learning", "keras"],
        "keras": ["python", "machine learning", "tensorflow"],
        "pytorch": ["python", "machine learning"],
        "elasticsearch": ["search", "database"],
        "kafka": ["streaming", "messaging"],
        "html": ["css", "javascript", "web"],
        "css": ["html", "javascript", "web"],
        "bootstrap": ["css", "html", "web"],
        "express.js": ["node.js", "javascript"],
        "node.js": ["javascript", "express.js"],
        "vue.js": ["vue", "javascript"],
        "next.js": ["react", "javascript"],
        # Microsoft Related
        "microsoft": ["windows", "azure", "office 365", "microsoft azure", "microsoft teams", "microsoft dynamics", "sharepoint", "visual studio", "c#", ".net", "power bi", "powerapps", "microsoft sql server", "microsoft office", "excel", "word"],
        "microsoft azure": ["azure", "cloud", "devops", "azure devops", "azure functions", "microsoft"],
        "visual studio": ["c#", ".net", "microsoft", "windows"],
        "office 365": ["microsoft", "excel", "word", "powerpoint", "outlook", "sharepoint", "teams"],
        "power bi": ["microsoft", "data analysis", "business intelligence"],
        "powerapps": ["microsoft", "office 365", "automation"],
        "microsoft sql server": ["sql", "database", "microsoft"],
        "sharepoint": ["microsoft", "office 365", "collaboration"],
        "microsoft teams": ["microsoft", "office 365", "collaboration", "communication"],
        "microsoft dynamics": ["crm", "erp", "microsoft"],
        "excel": ["microsoft", "office 365", "spreadsheet", "data analysis", "word"],
        "word": ["microsoft", "office 365", "excel", "documentation"],
        # Add more technical skills as needed

        # Non-Technical Skills
        "communication": ["presentation", "public speaking", "verbal communication", "written communication", "active listening"],
        "leadership": ["team management", "mentoring", "coaching", "decision making", "delegation"],
        "teamwork": ["collaboration", "cooperation", "cross-functional teams"],
        "problem solving": ["analytical thinking", "critical thinking", "troubleshooting"],
        "time management": ["prioritization", "scheduling", "deadline management"],
        "adaptability": ["flexibility", "resilience", "open-mindedness"],
        "creativity": ["innovation", "idea generation", "design thinking"],
        "attention to detail": ["accuracy", "thoroughness", "quality focus"],
        "project management": ["planning", "risk management", "resource allocation", "agile", "scrum", "kanban"],
        "customer service": ["client relations", "customer support", "empathy"],
        "negotiation": ["persuasion", "influencing", "conflict resolution"],
        "organizational skills": ["multitasking", "record keeping", "workflow optimization"],
        "self-motivation": ["initiative", "independence", "goal setting"],
        "work ethic": ["reliability", "dedication", "professionalism"],
        "emotional intelligence": ["self-awareness", "self-regulation", "social skills", "empathy"],
        # Add more non-technical skills as needed
    }
    
    # For each missing skill, check if any related skill is present
    all_resume_skills_lower = set(s.lower() for s in found_skills)
    for main_skill, rel_list in RELATED_SKILL_MAP.items():
        if main_skill not in all_resume_skills_lower:
            present_related = [rel for rel in rel_list if rel in all_resume_skills_lower]
            if present_related:
                related_skills[main_skill] = present_related

    # Return both found_skills and related_skills
    return sorted(list(set(skill for skill in found_skills if skill))), related_skills

def calculate_experience_from_dates_nlp(text):
    doc = nlp(text)
    total_experience_months = 0
    parsed_periods = []

    # Regex for explicit date ranges (still very useful)
    date_range_patterns = [
        r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{4})\s*-\s*(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{4}|\bPresent\b|\bCurrent\b)",
        r"(\d{1,2}/\d{4})\s*-\s*(\d{1,2}/\d{4}|\bPresent\b|\bCurrent\b)",
        r"(\d{4})\s*-\s*(\d{4}|\bPresent\b|\bCurrent\b)"
    ]

    extracted_date_texts = []
    for pattern in date_range_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            extracted_date_texts.append((match.group(1), match.group(2)))

    # Augment with spaCy's DATE entities for periods if regex missed them
    # This is more complex as spaCy might give individual dates, not ranges.
    # For simplicity, we'll primarily rely on regex for ranges here,
    # but spaCy helps by pre-processing the text for better tokenization.

    for start_str, end_str in extracted_date_texts:
        try:
            # date_parse can be finicky with just year, try to give it a month/day
            if re.fullmatch(r"\d{4}", start_str.strip()): start_str = f"Jan {start_str.strip()}"
            if re.fullmatch(r"\d{4}", end_str.strip()): end_str = f"Dec {end_str.strip()}"

            start_date = date_parse(start_str.replace(".",""), default=datetime(1,1,1))
            if end_str.lower().strip() in ["present", "current"]:
                end_date = datetime.now()
            else:
                end_date = date_parse(end_str.replace(".",""), default=datetime(1,1,1))

            if start_date.year > 1 and end_date.year > 1 and start_date < end_date:
                is_new_period = True
                for p_start, p_end in parsed_periods:
                    if max(start_date, p_start) < min(end_date, p_end): # Overlap
                        is_new_period = False; break
                if is_new_period:
                    parsed_periods.append((start_date, end_date))
                    delta = relativedelta(end_date, start_date)
                    total_experience_months += delta.years * 12 + delta.months + (1 if delta.days > 15 else 0) # Add a month if >15 days
        except Exception as e:
            # print(f"Date parsing error: {e} for '{start_str}' - '{end_str}'")
            continue
    return round(total_experience_months / 12, 1)

def extract_experience_years_nlp(text):
    # Look for explicit statements like "5 years of experience"
    # This regex is generally reliable for explicit mentions.
    match = re.search(r"(\d+\.?\d*)\s*(?:years?|yrs?)\s*(?:of|in)?\s*(?:professional|work|working|software|relevant)?\s*(?:experience|exp)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback to calculating from date ranges if no explicit statement
    calculated_exp = calculate_experience_from_dates_nlp(text)
    return calculated_exp if calculated_exp > 0 else 0


def extract_education_nlp(text):
    doc = nlp(text)
    qualifications = set()

    # Use NER for DEGREE/ORG/DATE
    for ent in doc.ents:
        if ent.label_ in ["ORG", "FAC", "GPE"]:
            line = ent.sent.text.strip()
            if any(kw in line.lower() for kw in DEGREE_KEYWORDS_LOWER):
                qualifications.add(line)
        if ent.label_ == "EDUCATION":
            qualifications.add(ent.text.strip())

    # Fuzzy match for degree keywords
    for line in text.split('\n'):
        for deg_kw in DEGREE_KEYWORDS_LOWER:
            if fuzz.partial_ratio(deg_kw, line.lower()) > 85 and len(line.strip()) > 5:
                qualifications.add(line.strip())


    # Check for certifications (often listed separately or with keywords)
    for line in text.split('\n'):
        if "certifi" in line.lower():
            cert_match = re.search(r"([\w\s-]+(?:Certificate|Certification|Certified)[\w\s-]*)", line, re.IGNORECASE)
            if cert_match and len(cert_match.group(1).strip()) > 5:
                qualifications.add(cert_match.group(1).strip())
            elif len(line.strip().split()) > 1 and len(line.strip()) > 5:
                qualifications.add(line.strip())
    return sorted(list(q for q in qualifications if q))

def extract_roles_projects_nlp(text):
    doc = nlp(text)
    roles = set()
    projects = set()

    # Look for sections explicitly titled "Experience", "Work Experience", "Projects"
    section_text = text
    experience_section_match = re.search(
        r"\b(Experience|Work\sExperience|Employment\sHistory|Projects)\b(?:[:\s\n]*((?:.|\n)*?))(?:\n\n\w|\n(?:[A-Z][A-Za-z ()]+):|$)",
        text, re.IGNORECASE | re.MULTILINE
    )
    if experience_section_match and experience_section_match.group(2):
        section_text = experience_section_match.group(2)
        doc_section = nlp(section_text)
    else:
        doc_section = doc

    # Heuristic 1: Noun chunks that look like job titles
    job_title_keywords = ["engineer", "developer", "manager", "analyst", "lead", "architect", "specialist", "consultant", "designer", "scientist", "intern"]
    for chunk in doc_section.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()
        if chunk_text.istitle() and any(keyword in chunk_lower for keyword in job_title_keywords):
            if len(chunk_text.split()) <= 5 and len(chunk_text) > 4:
                if not any(edu_kw in chunk_lower for edu_kw in ["university", "college", "degree", "education"]):
                    roles.add(chunk_text)

    # Heuristic 2: Lines starting with a capitalized phrase followed by "at" ORG or "|" ORG
    for line in section_text.split('\n'):
        line_strip = line.strip()
        if not line_strip: continue
        match = re.match(r"^([A-Z][A-Za-z\s,-]+(?:\s(?:[A-Z][A-Za-z&.\s]+))?)\s*(?:@|at|\|)\s*([A-Z][A-Za-z0-9\s.,&'-]+)", line_strip)
        if match:
            role_candidate = match.group(1).strip().rstrip(',')
            company_candidate = match.group(2).strip()
            if len(role_candidate.split()) <= 6 and len(role_candidate) > 4:
                if not any(edu_kw in role_candidate.lower() for edu_kw in ["university", "college", "degree", "education"]):
                    roles.add(f"{role_candidate} | {company_candidate}")

    # --- Project Extraction ---
    # Find "Projects" section and extract bullet points or lines under it
    project_section = None
    project_lines = []
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        if re.match(r"^\s*projects?\s*[:\-]?\s*$", line.strip(), re.IGNORECASE):
            project_section = idx
            break
    if project_section is not None:
        # Collect lines after "Projects" heading until next heading or empty line
        for l in lines[project_section+1:]:
            if l.strip() == "" or re.match(r"^[A-Z][A-Za-z\s]+:$", l.strip()):
                break
            # Bullet points or lines starting with dash/star
            if re.match(r"^\s*[\-\*\u2022]\s*(.+)", l):
                proj_title = re.sub(r"^\s*[\-\*\u2022]\s*", "", l).strip()
                if proj_title:
                    projects.add(proj_title)
            # Or lines that look like project titles (short, capitalized)
            elif len(l.strip()) > 3 and l.strip()[0].isupper() and len(l.strip().split()) < 10:
                projects.add(l.strip())

    # Fallback: look for lines with "Project:" or "Title:" or similar
    for l in lines:
        m = re.match(r"^\s*(?:Project|Title)\s*[:\-]\s*(.+)", l, re.IGNORECASE)
        if m:
            projects.add(m.group(1).strip())

    # Return both roles and projects, but keep backward compatibility
    all_roles_projects = sorted(list(roles | projects))
    return all_roles_projects


# --- JD Requirement Extraction (can also be enhanced with NLP, but kept simpler for now) ---
def extract_jd_requirements_nlp(jd_text):
    jd_doc = nlp(jd_text)
    requirements = {
        "mandatory_skills": [],
        "desired_experience_years": 0,
        "required_education": [],
        "preferred_certifications_specializations": []
    }

    # --- Improved Skill Extraction ---
    # Use only the first element (skills list) from extract_skills_nlp
    skills_list, _ = extract_skills_nlp(jd_text)
    all_jd_skills_found = set(skills_list)
    for ent in jd_doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "WORK_OF_ART", "SKILL"]:
            all_jd_skills_found.add(ent.text.strip())

    # --- Skill Classification with Context ---
    mandatory_keywords = [
        "required", "must have", "essential", "proficiency in", "strong experience with",
        "expertise in", "responsible for", "mandatory", "need to have"
    ]
    preferred_keywords = [
        "preferred", "nice to have", "plus", "desirable", "bonus", "good to have", "optional", "would be a plus"
    ]

    temp_mandatory = set()
    temp_preferred = set()
    for sent in jd_doc.sents:
        sent_text = sent.text.lower()
        for skill in all_jd_skills_found:
            skill_l = skill.lower()
            if skill_l in sent_text:
                if any(kw in sent_text for kw in mandatory_keywords):
                    temp_mandatory.add(skill)
                elif any(kw in sent_text for kw in preferred_keywords):
                    temp_preferred.add(skill)
    # Fallback: assign unclassified skills
    for skill in all_jd_skills_found:
        if skill not in temp_mandatory and skill not in temp_preferred:
            if len(temp_mandatory) < len(all_jd_skills_found) / 2:
                temp_mandatory.add(skill)
            else:
                temp_preferred.add(skill)

    requirements["mandatory_skills"] = sorted(temp_mandatory)
    requirements["preferred_certifications_specializations"] = sorted(temp_preferred)

    # --- Experience Extraction with NLP and Regex ---
    exp_patterns = [
        r"(?:minimum|at least|min)\s*(\d+\.?\d*)\s*(?:years?|yrs?)",
        r"(\d+\.?\d*)\+?\s*(?:years?|yrs?)\s*(?:of|in)?\s*(?:professional|relevant|software)?\s*experience"
    ]
    exp_found = False
    for pat in exp_patterns:
        exp_match = re.search(pat, jd_text.lower())
        if exp_match:
            requirements["desired_experience_years"] = float(exp_match.group(1))
            exp_found = True
            break
    if not exp_found:
        # Try to extract from sentences using spaCy NER
        for ent in jd_doc.ents:
            if ent.label_ == "CARDINAL":
                context = ent.sent.text.lower()
                if "year" in context and ("experience" in context or "exp" in context):
                    try:
                        requirements["desired_experience_years"] = float(ent.text)
                        break
                    except Exception:
                        continue

    # --- Education Extraction with NER and Regex ---
    edu_lines = set()
    for ent in jd_doc.ents:
        if ent.label_ in ["EDUCATION", "DEGREE"]:
            edu_lines.add(ent.text.strip())
    for line in jd_text.split('\n'):
        line_l = line.lower()
        if any(kw in line_l for kw in ["degree in", "bachelor", "master", "ph.d", "education:", "graduate", "undergraduate"]):
            edu_match = re.search(r"((?:Bachelor|Master|Ph\.D|Degree|Graduate|Undergraduate)[^.,;\n]*)", line, re.IGNORECASE)
            if edu_match:
                edu_lines.add(edu_match.group(1).strip())
            elif "degree" in line_l and len(line.strip().split()) > 2:
                edu_lines.add(line.strip())
    if not edu_lines:
        for deg_kw in DEGREE_KEYWORDS_LOWER:
            if deg_kw in jd_text.lower():
                edu_lines.add(deg_kw.title())
    requirements["required_education"] = sorted(edu_lines)

    # --- Certification Extraction with NER and Regex ---
    cert_lines = set()
    for ent in jd_doc.ents:
        if "certif" in ent.text.lower() or "certified" in ent.text.lower():
            cert_lines.add(ent.text.strip())
    for line in jd_text.split('\n'):
        if "certif" in line.lower() or "certified" in line.lower():
            cert_match = re.search(r"([\w\s-]+(?:Certificate|Certification|Certified)[\w\s-]*)", line, re.IGNORECASE)
            if cert_match and len(cert_match.group(1).strip()) > 5:
                cert_lines.add(cert_match.group(1).strip())
            elif len(line.strip().split()) > 1 and len(line.strip()) > 5:
                cert_lines.add(line.strip())
    requirements["preferred_certifications_specializations"] = sorted(set(requirements["preferred_certifications_specializations"]) | cert_lines)

    return requirements

# --- Fuzzy Skill Match Helper ---
def fuzzy_skill_match(skill, skills_set, threshold=85):
    """
    Returns True if the skill matches any skill in skills_set using fuzzy matching above the threshold.
    """
    for s in skills_set:
        if fuzz.ratio(skill.lower(), s.lower()) >= threshold:
            return True
    return False

# --- Education Fuzzy Match Helper ---
def fuzzy_edu_match(req_edu, resume_edu_list, threshold=85):
    """
    Returns True if the required education matches any entry in the resume education list using fuzzy matching above the threshold.
    """
    from fuzzywuzzy import fuzz
    for edu in resume_edu_list:
        if fuzz.partial_ratio(req_edu.lower(), edu.lower()) >= threshold:
            return True
    return False

# --- Scoring Logic (largely unchanged, relies on quality of extracted data) ---
def calculate_score(resume_details, jd_requirements, score_weights=None, threshold=70):
    if score_weights is None: score_weights = {'skills': 0.5, 'experience': 0.3, 'education': 0.2}
    score = 0
    skill_score_val, max_skill_score, matched_skills_list, unmatched_key_skills_list = 0, 0, [], []

    # Use fuzzy matching for skills
    resume_skills_set_norm = set(s.lower() for s in resume_details["skills"])
    jd_mandatory_skills_set_norm = set(s.lower() for s in jd_requirements["mandatory_skills"])
    jd_preferred_skills_set_norm = set(s.lower() for s in jd_requirements["preferred_certifications_specializations"] if isinstance(s, str))

    original_jd_skills_map = {}
    for s_list_name in ["mandatory_skills", "preferred_certifications_specializations"]:
        for s_orig in jd_requirements[s_list_name]:
            if isinstance(s_orig, str):
                original_jd_skills_map[s_orig.lower()] = s_orig

    # --- Skill Score Calculation ---
    if jd_mandatory_skills_set_norm:
        max_skill_score += len(jd_mandatory_skills_set_norm) * 2
        for req_skill_norm in jd_mandatory_skills_set_norm:
            or_skill_options = [opt.strip() for opt in req_skill_norm.split('/') if opt.strip()]
            if len(or_skill_options) > 1:
                if any(fuzzy_skill_match(opt_s, resume_skills_set_norm) for opt_s in or_skill_options):
                    skill_score_val += 2
                    matched_skills_list.extend([original_jd_skills_map.get(opt_s, opt_s) for opt_s in or_skill_options if fuzzy_skill_match(opt_s, resume_skills_set_norm)])
                else:
                    unmatched_key_skills_list.append(original_jd_skills_map.get(req_skill_norm, req_skill_norm))
            elif fuzzy_skill_match(req_skill_norm, resume_skills_set_norm):
                skill_score_val += 2
                matched_skills_list.append(original_jd_skills_map.get(req_skill_norm, req_skill_norm))
            else:
                unmatched_key_skills_list.append(original_jd_skills_map.get(req_skill_norm, req_skill_norm))

    if jd_preferred_skills_set_norm:
        max_skill_score += len(jd_preferred_skills_set_norm) * 1
        for pref_skill_norm in jd_preferred_skills_set_norm:
            or_skill_options = [opt.strip() for opt in pref_skill_norm.split('/') if opt.strip()]
            if len(or_skill_options) > 1:
                if any(fuzzy_skill_match(opt_s, resume_skills_set_norm) for opt_s in or_skill_options):
                    skill_score_val += 1
                    matched_skills_list.extend([original_jd_skills_map.get(opt_s, opt_s) for opt_s in or_skill_options if fuzzy_skill_match(opt_s, resume_skills_set_norm)])
            elif fuzzy_skill_match(pref_skill_norm, resume_skills_set_norm):
                skill_score_val += 1
                matched_skills_list.append(original_jd_skills_map.get(pref_skill_norm, pref_skill_norm))

    skill_percentage = (skill_score_val / max_skill_score) * 100 if max_skill_score > 0 else 100
    skill_score = round(skill_percentage, 1)
    score += skill_percentage * score_weights['skills']

    # --- Experience Score Calculation ---
    experience_score_val = 0
    jd_exp_years = jd_requirements.get("desired_experience_years", 0)
    resume_exp_years = resume_details.get("experience_years", 0)
    if jd_exp_years > 0:
        if resume_exp_years >= jd_exp_years:
            experience_score_val = 100
        elif resume_exp_years > 0:
            experience_score_val = min((resume_exp_years / jd_exp_years) * 75, 75)
    else:
        experience_score_val = 100
    experience_score = round(experience_score_val, 1)
    score += experience_score_val * score_weights['experience']

    # --- Education Score Calculation using fuzzy matching ---
    education_score_val = 0
    jd_edu_req = jd_requirements.get("required_education", [])
    resume_edu_list = resume_details.get("education", [])
    if jd_edu_req:
        match_found = False
        for req_edu_item in jd_edu_req:
            if fuzzy_edu_match(req_edu_item, resume_edu_list):
                match_found = True
                break
        if match_found: education_score_val = 100
        elif any("certifi" in req.lower() for req in jd_edu_req) and any("certifi" in edu.lower() for edu in resume_edu_list):
            education_score_val = 75
    else:
        education_score_val = 100
    education_score = round(education_score_val, 1)
    score += education_score_val * score_weights['education']

    final_score = round(min(max(score, 0), 100))
    recommendation = "Suitable" if final_score >= threshold else "Not Suitable"

    final_matched_skills = sorted(list(set(m for m in matched_skills_list if m)))
    final_unmatched_key_skills = []
    for jd_skill_orig_case in unmatched_key_skills_list:
        if jd_skill_orig_case and not fuzzy_skill_match(jd_skill_orig_case, resume_skills_set_norm):
            final_unmatched_key_skills.append(jd_skill_orig_case)

    # Return subtotals for transparency
    return {
        "matching_score": final_score,
        "matched_skills": final_matched_skills,
        "unmatched_key_skills": sorted(list(set(final_unmatched_key_skills))),
        "recommendation": recommendation,
        "score_breakdown": {
            "skills": skill_score,
            "experience": experience_score,
            "education": education_score,
            "weights": score_weights
        }
    }


# --- Main Processing Function (using NLP extractors) ---
def process_resume_and_jd(resume_text, jd_text, score_threshold=70):
    skills, related_skills = extract_skills_nlp(resume_text)
    resume_details = {
        "skills": skills,
        "related_skills": related_skills,
        "experience_years": extract_experience_years_nlp(resume_text),
        "education": extract_education_nlp(resume_text),
        "relevant_roles_projects": extract_roles_projects_nlp(resume_text)
    }
    jd_requirements = extract_jd_requirements_nlp(jd_text)
    match_analysis = calculate_score(resume_details, jd_requirements, threshold=score_threshold)

    return {
        "extracted_resume_details": resume_details,
        "job_requirements_summary": jd_requirements,
        "matching_analysis": match_analysis
    }

# --- Flask Routes (Unchanged from previous version) ---
from werkzeug.utils import secure_filename
import os

try:
    import docx
except ImportError:
    docx = None
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_stream):
    if not PyPDF2:
        return ""
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_stream):
    if not docx:
        return ""
    doc = docx.Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        resume_text = ""
        # Handle file upload
        if 'resume_file' in request.files and request.files['resume_file'].filename != '':
            resume_file = request.files['resume_file']
            if allowed_file(resume_file.filename):
                ext = resume_file.filename.rsplit('.', 1)[1].lower()
                if ext == 'pdf':
                    resume_text = extract_text_from_pdf(resume_file)
                elif ext == 'docx':
                    resume_text = extract_text_from_docx(resume_file)
                elif ext == 'txt':
                    resume_text = resume_file.read().decode('utf-8', errors='ignore')
            else:
                return "Unsupported file type for resume.", 400
        else:
            resume_text = request.form.get('resume_text', '')

        jd_text = request.form['jd_text']
        score_threshold = int(request.form.get('score_threshold', 70))

        if not resume_text or not jd_text:
            return "Please provide both resume (file or text) and job description text.", 400

        results_data = process_resume_and_jd(resume_text, jd_text, score_threshold)
        return render_template('results.html', data=results_data)
    return "Method not allowed", 405

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'resume_file' not in request.files or request.files['resume_file'].filename == '':
        return jsonify({'text': ''})
    resume_file = request.files['resume_file']
    if not allowed_file(resume_file.filename):
        return jsonify({'text': ''})
    ext = resume_file.filename.rsplit('.', 1)[1].lower()
    text = ""
    if ext == 'pdf':
        text = extract_text_from_pdf(resume_file)
    elif ext == 'docx':
        text = extract_text_from_docx(resume_file)
    elif ext == 'txt':
        text = resume_file.read().decode('utf-8', errors='ignore')
    return jsonify({'text': text})

# ================== Run the App ==================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
