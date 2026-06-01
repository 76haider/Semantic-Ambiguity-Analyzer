# Core analysis engine for semantic ambiguity detection

import spacy
from collections import Counter, defaultdict
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ============================================
# DYNAMIC ENTITY CLASS EXTRACTOR
# ============================================

class EntityExtractor:
    """Auto-detects entity classes from text content"""

    # Keyword mappings for dynamic classification
    DOMAIN_KEYWORDS = {
        "Political_Entity": ["president", "minister", "governor", "mayor", "senator",
                             "congress", "parliament", "election", "vote", "party",
                             "democrat", "republican", "policy", "government", "lawmaker"],
        "Sports_Entity": ["player", "team", "coach", "stadium", "match", "tournament",
                          "championship", "league", "trophy", "goal", "score", "referee",
                          "fan", "club", "captain", "midfielder", "striker"],
        "Business_Entity": ["company", "stock", "market", "profit", "revenue", "share",
                            "investor", "ceo", "bank", "trade", "economy", "startup",
                            "acquisition", "merger", "shareholder", "dividend"],
        "Technology_Entity": ["software", "hardware", "app", "digital", "ai", "cloud",
                              "data", "cyber", "algorithm", "server", "device", "tech",
                              "silicon", "engineer", "code", "network", "platform"],
        "Healthcare_Entity": ["hospital", "doctor", "patient", "treatment", "drug",
                              "vaccine", "surgery", "therapy", "clinical", "medicine",
                              "diagnosis", "pharma", "nurse", "health", "disease"],
        "Crime_Entity": ["police", "arrest", "court", "judge", "lawyer", "prison",
                         "criminal", "robbery", "theft", "suspect", "witness", "victim",
                         "attorney", "prosecutor", "defendant", "verdict"],
        "Education_Entity": ["school", "student", "teacher", "college", "university",
                             "curriculum", "exam", "degree", "scholarship", "classroom",
                             "professor", "campus", "academic", "course", "faculty"],
        "Entertainment_Entity": ["film", "movie", "actor", "director", "music", "star",
                                 "album", "concert", "show", "audience", "cinema",
                                 "celebrity", "oscar", "soundtrack", "premiere", "box office"]
    }

    @staticmethod
    def extract_dynamic_classes(text):
        """Extract and classify entities based on content keywords"""
        text_lower = text.lower()
        detected_classes = defaultdict(list)
        words = text_lower.split()

        for class_name, keywords in EntityExtractor.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_classes[class_name].append(keyword)

        # Convert to simple dict with counts
        result = {}
        for class_name, found_keywords in detected_classes.items():
            unique_keywords = list(set(found_keywords))
            if len(unique_keywords) >= 2:  # Min 2 keywords to confirm class
                result[class_name] = len(unique_keywords)

        return result

    @staticmethod
    def extract_spacy_entities(text):
        """Extract standard named entities using spaCy"""
        doc = nlp(text)
        entities = defaultdict(list)

        for ent in doc.ents:
            entities[ent.label_].append(ent.text)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return dict(entities)


# ============================================
# SEMANTIC AMBIGUITY ANALYZER
# ============================================

class SemanticAnalyzer:
    """8 semantic ambiguity detection functions"""

    # Common ambiguous words with multiple meanings
    AMBIGUOUS_WORDS = {
        "bank": ["financial institution", "river edge"],
        "bat": ["animal", "sports equipment"],
        "book": ["reading material", "to reserve"],
        "case": ["container", "legal matter", "instance"],
        "charge": ["fee", "accusation", "electrical"],
        "court": ["legal place", "sports area", "royal residence"],
        "fair": ["just", "exhibition", "light colored"],
        "fall": ["season", "to drop"],
        "file": ["document", "tool", "to submit"],
        "head": ["body part", "leader", "top"],
        "light": ["not heavy", "illumination", "pale"],
        "match": ["game", "equal", "flame starter"],
        "mine": ["belonging to me", "excavation"],
        "order": ["command", "sequence", "request"],
        "park": ["recreation area", "to leave vehicle"],
        "party": ["celebration", "political group", "legal person"],
        "right": ["correct", "direction", "entitlement"],
        "run": ["to move quickly", "to operate", "sequence"],
        "sentence": ["grammatical unit", "punishment"],
        "spring": ["season", "coiled metal", "water source"],
        "table": ["furniture", "to postpone", "data grid"],
        "watch": ["timepiece", "to observe"],
        "will": ["future intent", "legal document", "determination"]
    }

    # Ambiguous pronouns
    AMBIGUOUS_PRONOUNS = ["it", "they", "them", "their", "this", "that", "these", "those"]

    # Scope ambiguity patterns
    SCOPE_PATTERNS = [
        r"(old|young|tall|short)\s+(\w+)\s+and\s+(\w+)",
        r"(heavy|lazy|quick)\s+(\w+)\s+and\s+(\w+)",
    ]

    # Quantifier ambiguity words
    QUANTIFIERS = ["every", "each", "all", "some", "many", "few", "several", "any"]

    # Temporal ambiguity words
    TEMPORAL_WORDS = ["yesterday", "today", "tomorrow", "soon", "later",
                      "recently", "eventually", "next week", "last month",
                      "previously", "currently", "eventually", "shortly"]

    # ============================================
    # FUNCTION 1: LEXICAL AMBIGUITY
    # ============================================
    @staticmethod
    def lexical_ambiguity(text):
        """Detect words with multiple possible meanings"""
        words = text.lower().split()
        cleaned_words = [w.strip('.,!?;:()[]{}""\'') for w in words]
        ambiguous_found = []

        for word in cleaned_words:
            if word in SemanticAnalyzer.AMBIGUOUS_WORDS:
                ambiguous_found.append({
                    "word": word,
                    "meanings": SemanticAnalyzer.AMBIGUOUS_WORDS[word]
                })

        return ambiguous_found

    # ============================================
    # FUNCTION 2: REFERENTIAL AMBIGUITY
    # ============================================
    @staticmethod
    def referential_ambiguity(text):
        """Detect unclear pronoun references"""
        doc = nlp(text)
        sentences = list(doc.sents)
        ambiguous_refs = []

        for i, sent in enumerate(sentences):
            sent_text = sent.text
            for pronoun in SemanticAnalyzer.AMBIGUOUS_PRONOUNS:
                if pronoun in sent_text.lower().split():
                    # Check if pronoun could refer to multiple things
                    context = sent_text.lower()
                    # Simple check: if pronoun appears without clear antecedent
                    if i > 0:
                        prev_sent = sentences[i-1].text.lower()
                        # Count potential referents (nouns) in previous sentence
                        prev_doc = nlp(prev_sent)
                        noun_count = len([token for token in prev_doc
                                          if token.pos_ in ["NOUN", "PROPN"]])
                        if noun_count >= 3:
                            ambiguous_refs.append({
                                "pronoun": pronoun,
                                "sentence": sent_text.strip(),
                                "potential_referents": noun_count
                            })

        return ambiguous_refs

    # ============================================
    # FUNCTION 3: SCOPE AMBIGUITY
    # ============================================
    @staticmethod
    def scope_ambiguity(text):
        """Detect scope ambiguity like 'old men and women'"""
        doc = nlp(text)
        ambiguous_scopes = []

        for pattern in SemanticAnalyzer.SCOPE_PATTERNS:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                adjective, noun1, noun2 = match
                ambiguous_scopes.append({
                    "phrase": f"{adjective} {noun1} and {noun2}",
                    "ambiguity": f"Does '{adjective}' modify only '{noun1}' or both '{noun1} and {noun2}'?"
                })

        return ambiguous_scopes

    # ============================================
    # FUNCTION 4: ATTACHMENT AMBIGUITY
    # ============================================
    @staticmethod
    def attachment_ambiguity(text):
        """Detect prepositional phrase attachment ambiguity"""
        doc = nlp(text)
        attachment_issues = []

        for sent in doc.sents:
            # Find sentences with "with", "in", "on", "by" phrases
            for token in sent:
                if token.dep_ == "prep" and token.head.pos_ == "NOUN":
                    # Check if there are two possible attachment points
                    phrase_start = token.i
                    phrase_end = min(phrase_start + 4, len(sent))
                    phrase = sent[phrase_start:phrase_end]

                    attachment_issues.append({
                        "preposition": token.text,
                        "phrase": " ".join([t.text for t in phrase]),
                        "sentence": sent.text.strip()[:100]
                    })

        return attachment_issues

    # ============================================
    # FUNCTION 5: QUANTIFIER AMBIGUITY
    # ============================================
    @staticmethod
    def quantifier_ambiguity(text):
        """Detect quantifier scope ambiguity"""
        doc = nlp(text)
        quantifier_issues = []

        for sent in doc.sents:
            sent_text = sent.text.lower()
            words = sent_text.split()

            for q_word in SemanticAnalyzer.QUANTIFIERS:
                if q_word in words:
                    # Check if sentence has multiple interpretations
                    # based on quantifier scope
                    quantifier_issues.append({
                        "quantifier": q_word,
                        "sentence": sent.text.strip()[:120],
                        "ambiguity": f"Scope of '{q_word}' may vary"
                    })

        return quantifier_issues

    # ============================================
    # FUNCTION 6: TEMPORAL AMBIGUITY
    # ============================================
    @staticmethod
    def temporal_ambiguity(text):
        """Detect vague or ambiguous time references"""
        doc = nlp(text)
        temporal_issues = []

        for sent in doc.sents:
            sent_text = sent.text.lower()
            for temp_word in SemanticAnalyzer.TEMPORAL_WORDS:
                if temp_word in sent_text:
                    temporal_issues.append({
                        "temporal_word": temp_word,
                        "sentence": sent.text.strip()[:120],
                        "ambiguity": f"'{temp_word}' is vague - when exactly?"
                    })

        return temporal_issues

    # ============================================
    # FUNCTION 7: CONTEXTUAL AMBIGUITY
    # ============================================
    @staticmethod
    def contextual_ambiguity(text):
        """Detect missing context clues leading to ambiguity"""
        doc = nlp(text)
        context_issues = []
        sentences = list(doc.sents)

        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip()

            # Check for demonstratives without clear referent
            demonstratives = ["this", "that", "these", "those"]
            first_word = sent_text.split()[0].lower() if sent_text.split() else ""

            if first_word in demonstratives and i > 0:
                context_issues.append({
                    "issue": f"'{first_word}' refers to unclear antecedent",
                    "sentence": sent_text[:120]
                })

            # Check for undefined abbreviations or jargon
            abbreviations = re.findall(r'\b[A-Z]{2,5}\b', sent_text)
            for abbr in abbreviations:
                if abbr not in ["CEO", "USA", "UK", "AI"]:  # Common known ones
                    full_form_found = False
                    for prev_sent in sentences[:i]:
                        if abbr.lower() in prev_sent.text.lower():
                            full_form_found = True
                            break
                    if not full_form_found and i > 0:
                        context_issues.append({
                            "issue": f"Abbreviation '{abbr}' not defined in context",
                            "sentence": sent_text[:120]
                        })

        return context_issues

    # ============================================
    # FUNCTION 8: ENTITY OVERLAP
    # ============================================
    @staticmethod
    def entity_overlap(text):
        """Detect when same entity name could refer to different things"""
        doc = nlp(text)
        entity_occurrences = defaultdict(list)
        overlaps = []

        # Collect all entity occurrences
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "LOC", "PERSON"]:
                entity_occurrences[ent.text].append({
                    "label": ent.label_,
                    "position": ent.start_char
                })

        # Check for entities that appear with different labels
        for entity, occurrences in entity_occurrences.items():
            labels = set([occ["label"] for occ in occurrences])
            if len(labels) > 1:
                overlaps.append({
                    "entity": entity,
                    "labels": list(labels),
                    "ambiguity": f"'{entity}' could be {', '.join(labels)}"
                })
            elif len(occurrences) > 1 and entity.lower() in [
                "apple", "amazon", "delta", "shell", "visa", "orange"
            ]:
                overlaps.append({
                    "entity": entity,
                    "labels": labels,
                    "ambiguity": f"'{entity}' could have multiple meanings (brand vs common word)"
                })

        return overlaps

    # ============================================
    # FULL ANALYSIS - RUNS ALL 8 FUNCTIONS
    # ============================================
    @staticmethod
    def full_analysis(text):
        """Run all 8 ambiguity detection functions"""
        # Truncate text to ~300 words for analysis
        words = text.split()
        if len(words) > 350:
            text = " ".join(words[:350])

        results = {
            "Lexical Ambiguity": SemanticAnalyzer.lexical_ambiguity(text),
            "Referential Ambiguity": SemanticAnalyzer.referential_ambiguity(text),
            "Scope Ambiguity": SemanticAnalyzer.scope_ambiguity(text),
            "Attachment Ambiguity": SemanticAnalyzer.attachment_ambiguity(text),
            "Quantifier Ambiguity": SemanticAnalyzer.quantifier_ambiguity(text),
            "Temporal Ambiguity": SemanticAnalyzer.temporal_ambiguity(text),
            "Contextual Ambiguity": SemanticAnalyzer.contextual_ambiguity(text),
            "Entity Overlap": SemanticAnalyzer.entity_overlap(text)
        }

        # Calculate counts
        counts = {key: len(value) for key, value in results.items()}

        # Total score
        total_ambiguities = sum(counts.values())
        word_count = len(text.split())

        return {
            "results": results,
            "counts": counts,
            "total": total_ambiguities,
            "word_count": word_count,
            "density": round(total_ambiguities / max(word_count / 100, 1), 2)
        }


# ============================================
# HELPER: GET SEVERITY LABEL
# ============================================
def get_severity(count):
    """Return severity label and color"""
    if count >= 6:
        return "🔴 HIGH", "#E74C3C"
    elif count >= 3:
        return "🟡 MEDIUM", "#F39C12"
    else:
        return "🟢 LOW", "#27AE60"
