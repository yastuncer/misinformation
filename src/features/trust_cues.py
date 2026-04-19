import math
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

auth_lex = [
    # Obligation / directive modality
    "must", "must not",
    "need", "need to",
    "have to", "has to", "have got to", "got to",
    "should", "should not",
    "cannot", "will not",

    "require", "required", "is required to", "are required to",
    "is necessary to", "it is necessary",
    "is obligated to", "are obligated to", "obliged to",
    "mandate", "mandated",
    "mandatory", "compulsory",
    "imperative", "essential", "vital",
    "comply", "comply with",
    "enforce", "enforced",

    # Certainty / epistemic closure
    "definitely", "certainly", "clearly", "obviously",
    "undeniably", "without doubt", "no doubt",
    "undoubtedly", "indisputably", "absolutely",
    "unequivocally", "unquestionably",
    "in fact",
    "always", "never",

    "fact", "facts", "truth", "proven",
    "guaranteed",
    "without question",
    "there is no question",
    "beyond question",

    "it is clear that",
    "the reality is",
    "the truth is",
    "make no mistake",
    "clearly shows",

    "proven fact", "scientific fact",

    # Evidence / epistemic authority
    "experts", "scientists", "doctors", "researchers",
    "studies", "research", "data", "evidence",
    "report", "reports",

    "according to",
    "confirmed",
    "proves", "demonstrates",

    "peer-reviewed", "peer reviewed",
    "meta-analysis", "systematic review",
    "clinical trial", "randomized trial",

    "latest study", "new study", "recent study",
    "published in", "journal",
    "the literature shows",
    "evidence suggests", "data shows",

    # Institutional authority
    "cdc", "who", "epa", "ipcc",
    "un", "united nations",
    "wmo", "national weather service",

    "government", "official", "government officials",
    "authorities", "regulators",
    "policy", "law", "regulation",

    "ipcc report", "who report",
    "public health experts", "climate scientists",
    "issued by", "released by", "statement from",

    # Instructional / directive framing
    "follow", "ensure", "make sure",
    "do this", "stop", "avoid",
    "do not", "avoid doing", "stop doing",

    "you must", "you need to",
    "you should", "you should not",

    "it is recommended", "it is advised",
    "we urge", "we strongly recommend",

    "take steps", "take action",
    "follow guidelines", "adhere to",

    # Dismissal of uncertainty
    "settled", "settled science",
    "scientific consensus",
    "conclusive",

    "debunked", "false claim", "myth",
    "no evidence", "zero evidence",

    "fact check", "verified",
    "resolved", "case closed",
    "conspiracy theory", "hoax"
]

hedge_lex = [
    # Epistemic uncertainty
    "maybe", "perhaps",
    "might", "may be", "could be",

    "likely", "unlikely",
    "probable", "improbable",
    "possibly", "potentially",

    "it seems that", "it appears that",
    "seems", "appears",
    "suggests that",
    "there is some evidence",

    # Attribution uncertainty
    "reportedly", "allegedly", "supposedly",
    "claims that", "it is claimed",
    "sources say", "some say",
    "experts say",
    "it is believed", "it is thought",

    # Approximation
    "about", "approximately", "roughly",
    "around", "nearly", "almost",
    "up to", "as much as",

    # Weakening / softening
    "somewhat", "relatively",
    "moderately", "partially",
    "arguably"
]

urg_lex = [
    # Immediate time pressure
    "now", "immediately", "right now",
    "asap", "urgent", "instantly",
    "quick", "quickly", "hurry",

    "right away", "at once", "without delay",
    "this minute", "this second",
    "as soon as possible",

    "do it now", "act immediately",

    # Deadlines / scarcity
    "today", "tonight",
    "before it's too late",
    "last chance", "final chance", "last opportunity",
    "limited time", "running out",
    "time is running out",
    "deadline", "expires soon",
    "closing soon", "final hours",

    # Escalation / alarm
    "alert", "warning", "critical",
    "emergency", "danger", "risk",
    "threat", "crisis",

    "catastrophe", "disaster", "collapse",
    "devastating", "dire", "alarming",

    "massive", "severe", "extreme",
    "unprecedented", "historic",

    # Threat framing
    "will kill", "will harm", "will destroy",
    "poses a risk", "endangers",
    "threatens", "jeopardizes",
    "putting lives at risk",

    # Calls to action
    "act", "act now", "respond",
    "do not wait", "take action",

    # Attention / engagement pressure
    "attention", "listen", "important",
    "must read", "share this", "spread this",
    "do not ignore", "wake up",

    # Headline urgency
    "breaking", "just in", "developing",
    "live", "just announced",
    "new findings", "latest update",
    "breaking news", "urgent update",

    "viral", "spread"
]

def build_patterns(terms):
    return [nlp.make_doc(term) for term in terms]

# Initialize matchers
auth_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
hedge_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
urg_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Assign lexicons to matchers
auth_matcher.add("AUTH", build_patterns(auth_lex))
hedge_matcher.add("HEDGE", build_patterns(hedge_lex))
urg_matcher.add("URG", build_patterns(urg_lex))


# Score post for authoritative language and urgency
def score_auth_urg(text):
    doc = nlp(text)

    # Match phrases to text
    auth_matches = auth_matcher(doc)
    hedge_matches = hedge_matcher(doc)
    urg_matches = urg_matcher(doc)

    # Raw counts
    auth_count = len(auth_matches)
    hedge_count = len(hedge_matches)
    urg_count = len(urg_matches)

    exclam_count = sum(1 for t in doc if t.text == "!") # Count exclamation points
    cap_words = [t.text for t in doc if t.is_upper and t.is_alpha and len(t.text) > 2] # Count words in all caps

    auth_score = (
        (auth_count - 0.5 * hedge_count) # Deduct hedging from authoritative language score
        / len(doc) # Normalize by post length
    ) 
    auth_score = max(auth_score, 0) # 0 as lower bound

    urg_score = (
        (urg_count 
        + 0.5 * min(exclam_count, 10) # Limit influence of exclamation marks
        + 0.5 * min(len(cap_words), 5)) # Limit influence of words in all caps
        / len(doc) # Normalize by post length
    )

    return auth_score, urg_score

# Squash scores into a range [0, 1]
def squash(x, k=10):
    return 1 / (1 + math.exp(-k * x))

def avg_auth_urg(texts):
    avg_auth, avg_urg = 0, 0

    for text in texts: 
        auth_score, urg_score = score_auth_urg(text)
        avg_auth += squash(auth_score)
        avg_urg += squash(urg_score)

    avg_auth /= len(texts) 
    avg_urg /= len(texts)
    
    return avg_auth, avg_urg
