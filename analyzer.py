# Core analysis engine for semantic ambiguity detection

import spacy
from collections import Counter, defaultdict
import re
import numpy as np

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")


# ============================================
# DYNAMIC ENTITY CLASS EXTRACTOR
# ============================================

class EntityExtractor:

    DOMAIN_KEYWORDS = {
        "Political_Entity": ["president", "minister", "governor", "mayor", "senator",
                             "congress", "parliament", "election", "vote", "party",
                             "democrat", "republican", "policy", "government", "lawmaker",
                             "imran khan", "rally", "political", "politics", "leader"],
        "Sports_Entity": ["player", "team", "coach", "stadium", "match", "tournament",
                          "championship", "league", "trophy", "goal", "score", "referee",
                          "fan", "club", "captain", "midfielder", "striker", "cricket",
                          "football", "bowler", "batsman", "wicket", "ball", "bat",
                          "shadab khan", "babar azam", "all rounder", "batting", "bowling"],
        "Business_Entity": ["company", "stock", "market", "profit", "revenue", "share",
                            "investor", "ceo", "bank", "trade", "economy", "startup",
                            "acquisition", "merger", "shareholder", "dividend"],
        "Technology_Entity": ["software", "hardware", "app", "digital", "ai", "cloud",
                              "data", "cyber", "algorithm", "server", "device", "tech",
                              "silicon", "engineer", "code", "network", "platform",
                              "artificial intelligence", "machine learning", "coding"],
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
        text_lower = text.lower()
        detected_classes = defaultdict(list)

        for class_name, keywords in EntityExtractor.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_classes[class_name].append(keyword)

        result = {}
        for class_name, found_keywords in detected_classes.items():
            unique_keywords = list(set(found_keywords))
            if len(unique_keywords) >= 2:
                result[class_name] = len(unique_keywords)

        return result

    @staticmethod
    def extract_spacy_entities(text):
        doc = nlp(text)
        entities = defaultdict(list)

        for ent in doc.ents:
            entities[ent.label_].append(ent.text)

        for key in entities:
            entities[key] = list(set(entities[key]))

        return dict(entities)


# ============================================
# SEMANTIC ANALYZER - 10 FUNCTIONS
# ============================================

class SemanticAnalyzer:

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

    AMBIGUOUS_PRONOUNS = ["it", "they", "them", "their", "this", "that", "these", "those"]
    SCOPE_PATTERNS = [
        r"(old|young|tall|short)\s+(\w+)\s+and\s+(\w+)",
        r"(heavy|lazy|quick)\s+(\w+)\s+and\s+(\w+)",
    ]
    QUANTIFIERS = ["every", "each", "all", "some", "many", "few", "several", "any"]
    TEMPORAL_WORDS = ["yesterday", "today", "tomorrow", "soon", "later",
                      "recently", "eventually", "next week", "last month",
                      "previously", "currently", "eventually", "shortly"]

    # Semantic compatibility patterns - verbs and their expected objects
    VERB_CATEGORIES = {
        "consume": ["drink", "eat", "consume", "swallow", "sip", "devour"],
        "create": ["write", "build", "make", "construct", "paint", "draw"],
        "move": ["drive", "ride", "fly", "walk", "run", "swim"],
        "communicate": ["speak", "talk", "say", "tell", "discuss", "announce"],
    }

    # Nouns that can be consumed
    CONSUMABLE_CATEGORIES = ["water", "food", "tea", "coffee", "juice", "milk",
                              "drink", "meal", "bread", "rice", "fruit", "meat"]
    
    # Nouns that can be ridden/driven
    VEHICLE_CATEGORIES = ["car", "bike", "bus", "train", "cycle", "motorcycle",
                           "horse", "bicycle", "boat", "ship", "plane"]

    # Nouns that can be written/created
    CREATABLE_CATEGORIES = ["book", "letter", "story", "poem", "essay", "article",
                             "report", "novel", "code", "program", "software"]

    # ============================================
    # FUNCTION 1: LEXICAL AMBIGUITY
    # ============================================
    @staticmethod
    def lexical_ambiguity(text):
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
        doc = nlp(text)
        sentences = list(doc.sents)
        ambiguous_refs = []

        for i, sent in enumerate(sentences):
            sent_text = sent.text
            for pronoun in SemanticAnalyzer.AMBIGUOUS_PRONOUNS:
                words_in_sent = sent_text.lower().split()
                if pronoun in words_in_sent:
                    if i > 0:
                        prev_sent = sentences[i-1].text.lower()
                        prev_doc = nlp(prev_sent)
                        noun_count = len([token for token in prev_doc
                                          if token.pos_ in ["NOUN", "PROPN"]])
                        if noun_count >= 2:
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
        ambiguous_scopes = []

        for pattern in SemanticAnalyzer.SCOPE_PATTERNS:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                adjective, noun1, noun2 = match
                ambiguous_scopes.append({
                    "phrase": f"{adjective} {noun1} and {noun2}",
                    "ambiguity": f"Does '{adjective}' modify only '{noun1}' or both?"
                })

        return ambiguous_scopes

    # ============================================
    # FUNCTION 4: ATTACHMENT AMBIGUITY
    # ============================================
    @staticmethod
    def attachment_ambiguity(text):
        doc = nlp(text)
        attachment_issues = []

        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "prep" and token.head.pos_ == "NOUN":
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
        doc = nlp(text)
        quantifier_issues = []

        for sent in doc.sents:
            sent_text = sent.text.lower()
            words = sent_text.split()
            for q_word in SemanticAnalyzer.QUANTIFIERS:
                if q_word in words:
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
        doc = nlp(text)
        context_issues = []
        sentences = list(doc.sents)

        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            words = sent_text.split()
            if not words:
                continue

            demonstratives = ["this", "that", "these", "those"]
            first_word = words[0].lower()

            if first_word in demonstratives and i > 0:
                context_issues.append({
                    "issue": f"'{first_word}' refers to unclear antecedent",
                    "sentence": sent_text[:120]
                })

            abbreviations = re.findall(r'\b[A-Z]{2,5}\b', sent_text)
            for abbr in abbreviations:
                if abbr not in ["CEO", "USA", "UK", "AI", "ICC"]:
                    full_form_found = False
                    for prev_sent in sentences[:i]:
                        if abbr.lower() in prev_sent.text.lower():
                            full_form_found = True
                            break
                    if not full_form_found and i > 0:
                        context_issues.append({
                            "issue": f"Abbreviation '{abbr}' not defined",
                            "sentence": sent_text[:120]
                        })

        return context_issues

    # ============================================
    # FUNCTION 8: ENTITY OVERLAP
    # ============================================
    @staticmethod
    def entity_overlap(text):
        doc = nlp(text)
        entity_occurrences = defaultdict(list)
        overlaps = []

        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "LOC", "PERSON"]:
                entity_occurrences[ent.text].append({
                    "label": ent.label_,
                    "position": ent.start_char
                })

        for entity, occurrences in entity_occurrences.items():
            labels = set([occ["label"] for occ in occurrences])
            if len(labels) > 1:
                overlaps.append({
                    "entity": entity,
                    "labels": list(labels),
                    "ambiguity": f"'{entity}' could be {', '.join(labels)}"
                })

        return overlaps

    # ============================================
    # FUNCTION 9: VERB-OBJECT COMPATIBILITY (NEW)
    # ============================================
    @staticmethod
    def verb_object_compatibility(text):
        """Algorithmic detection of verb-object semantic mismatches"""
        doc = nlp(text)
        issues = []
        sentences = list(doc.sents)

        for sent in sentences:
            verb = None
            dobj = None
            subj = None

            # Extract verb, subject, and direct object
            for token in sent:
                if token.pos_ == "VERB":
                    verb = token
                if token.dep_ == "dobj":
                    dobj = token
                if token.dep_ == "nsubj":
                    subj = token

            # If we have verb and object, check compatibility
            if verb and dobj:
                verb_text = verb.lemma_.lower()
                obj_text = dobj.lemma_.lower()

                # Check semantic similarity using spaCy vectors
                verb_token = nlp(verb_text)[0] if nlp(verb_text) else None
                obj_token = nlp(obj_text)[0] if nlp(obj_text) else None

                # Category-based logic check
                mismatch_found = False
                mismatch_reason = ""

                # Check consume verbs (drink, eat) + object
                if verb_text in ["drink", "eat", "consume", "sip", "devour", "swallow"]:
                    # Check if object is in consumable categories OR has food/drink properties
                    obj_is_consumable = (
                        obj_text in SemanticAnalyzer.CONSUMABLE_CATEGORIES or
                        obj_text in ["water", "milk", "juice", "tea", "coffee", "food"]
                    )
                    # Check if object is clearly NOT consumable (animals, objects, abstract)
                    non_consumable_indicators = ["cricket", "football", "school", "building",
                                                  "book", "chair", "table", "car", "computer",
                                                  "phone", "paper", "rock", "stone", "glass",
                                                  "metal", "wood", "plastic", "brick", "wall"]
                    if obj_text in non_consumable_indicators or (
                        dobj.pos_ in ["PROPN", "NOUN"] and not obj_is_consumable and
                        len(obj_text) > 2
                    ):
                        mismatch_found = True
                        mismatch_reason = f"'{verb_text}' action cannot be performed on '{obj_text}' (not consumable)"

                # Check creation verbs (write, build, make)
                elif verb_text in ["write", "build", "construct", "paint", "draw"]:
                    # Objects that can be written/built/created
                    creatable = obj_text in SemanticAnalyzer.CREATABLE_CATEGORIES
                    physical_objects = ["building", "house", "bridge", "road", "wall", "tower"]
                    if verb_text == "write":
                        # Write requires text-based objects
                        text_objects = ["book", "letter", "story", "poem", "essay", "article",
                                        "report", "novel", "note", "message", "email", "code"]
                        if obj_text not in text_objects and dobj.pos_ == "NOUN":
                            mismatch_found = True
                            mismatch_reason = f"'write' requires a text-based object, not '{obj_text}'"
                    elif verb_text == "build":
                        if obj_text not in physical_objects and obj_text not in ["software", "app", "website"]:
                            mismatch_found = True
                            mismatch_reason = f"'build' requires a physical or digital structure, not '{obj_text}'"

                # Check movement verbs (drive, ride, fly)
                elif verb_text in ["drive", "ride"]:
                    vehicle_like = (
                        obj_text in SemanticAnalyzer.VEHICLE_CATEGORIES or
                        obj_text in ["car", "bike", "bus", "train", "cycle", "motorcycle",
                                     "horse", "bicycle", "boat", "ship", "plane", "scooter"]
                    )
                    if not vehicle_like and dobj.pos_ == "NOUN" and len(obj_text) > 2:
                        mismatch_found = True
                        mismatch_reason = f"'{verb_text}' requires a vehicle or rideable object, not '{obj_text}'"

                # Check subject-verb compatibility (subject can do the verb?)
                if subj and verb:
                    subj_text = subj.lemma_.lower()
                    # Objects/buildings cannot perform human actions
                    non_animate_subjects = ["building", "wall", "table", "chair", "rock",
                                             "stone", "book", "car", "computer", "phone",
                                             "paper", "glass", "metal", "wood", "plastic",
                                             "brick", "cricket", "football"]
                    if subj_text in non_animate_subjects:
                        human_verbs = ["write", "speak", "talk", "say", "tell", "discuss",
                                       "announce", "think", "believe", "know", "understand",
                                       "love", "hate", "want", "need", "decide", "choose",
                                       "sing", "dance", "laugh", "cry", "smile", "run",
                                       "walk", "jump", "swim", "fly", "eat", "drink", "read"]
                        if verb_text in human_verbs:
                            mismatch_found = True
                            mismatch_reason = f"'{subj_text}' (non-animate) cannot perform human action '{verb_text}'"

                if mismatch_found:
                    issues.append({
                        "verb": verb_text,
                        "object": obj_text,
                        "sentence": sent.text.strip(),
                        "reason": mismatch_reason
                    })

        return issues
    # ============================================
    # FUNCTION 10: SEMANTIC COHERENCE (NEW)
    # ============================================
    @staticmethod
    def semantic_coherence(text):
        """Check overall semantic coherence of sentences using algorithmic approach"""
        doc = nlp(text)
        issues = []
        sentences = list(doc.sents)

        opposite_pairs = [
            ({"colorless", "colourless"}, {"green", "red", "blue", "yellow", "pink", "purple", "orange", "brown", "black", "white", "gray", "grey"}),
            ({"dry", "dried"}, {"water", "ocean", "sea", "river", "lake", "rain", "liquid"}),
            ({"cold", "freezing", "icy"}, {"fire", "flame", "heat", "burning", "hot"}),
            ({"dark", "black", "pitch"}, {"light", "bright", "shining", "glowing"}),
            ({"silent", "quiet", "mute"}, {"noise", "sound", "loud", "scream", "shout"}),
            ({"living", "alive"}, {"dead", "corpse", "deceased"}),
            ({"square", "rectangular"}, {"circle", "round", "oval", "sphere"}),
            ({"solid", "hard"}, {"liquid", "gas", "soft"}),
            ({"hot", "burning"}, {"ice", "frozen", "cold"}),
            ({"bright", "shining"}, {"darkness", "shadow", "gloom"}),
            ({"open", "public"}, {"secret", "hidden", "private"}),
            ({"old", "ancient"}, {"new", "fresh", "modern"}),
            ({"deafening", "loud"}, {"silence", "quiet"}),
        ]

        for sent in sentences:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            words = set(sent_lower.split())

            for set1, set2 in opposite_pairs:
                found_set1 = words & set1
                found_set2 = words & set2
                if found_set1 and found_set2:
                    word1 = list(found_set1)[0]
                    word2 = list(found_set2)[0]
                    issues.append({
                        "sentence": sent_text,
                        "issue": f"Contradiction: '{word1}' and '{word2}' cannot logically exist together",
                        "pattern": f"{word1}_{word2}"
                    })
                    break

        return issues

    # ============================================
    # FULL ANALYSIS - RUNS ALL 10 FUNCTIONS
    # ============================================
    @staticmethod
    def full_analysis(text):
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
            "Entity Overlap": SemanticAnalyzer.entity_overlap(text),
            "Verb-Object Mismatch": SemanticAnalyzer.verb_object_compatibility(text),
            "Semantic Coherence": SemanticAnalyzer.semantic_coherence(text)
        }

        counts = {key: len(value) for key, value in results.items()}
        total_ambiguities = sum(counts.values())
        word_count = len(text.split())

        return {
            "results": results,
            "counts": counts,
            "total": total_ambiguities,
            "word_count": word_count,
            "density": round(total_ambiguities / max(word_count / 100, 1), 2)
        }


def get_severity(count):
    if count >= 6:
        return "HIGH", "#E74C3C"
    elif count >= 3:
        return "MEDIUM", "#F39C12"
    else:
        return "LOW", "#27AE60"
