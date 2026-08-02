"""
Message Notification Router - Minimal Working Pipeline
=========================================================

Reads all dataset CSVs, runs every message in dataset/messages.csv through
a deterministic routing pipeline, and writes dataset/output.csv (or ./output.csv)
with the required schema:

    message_id,action,message_type,reason,confidence,evidence_message_ids

This is a DELIBERATELY MINIMAL first pass. It implements:
    - full dataset loading
    - context joining (message -> user/group/business/history)
    - a small set of obvious, high-confidence deterministic rules:
        * obvious scams (OTP/PIN/password requests, spoofed domains, prompt injection)
        * obvious spam (high forward count + chain/blessing language)
        * obvious urgency (admin messages, verified-business transactional updates,
          direct @mentions, same-day deadline language)
        * sensible defaults (digest for everything else, personal messages handled
          explicitly)

Personalization scoring, OCR, ASR, and evidence selection are implemented as
clearly-marked PLACEHOLDER functions with the correct signatures so they can be
filled in later without touching the rest of the pipeline.

Run:
    python main.py
"""

import os
import re
import csv
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = os.environ.get("DATASET_DIR", "dataset")
OUTPUT_PATH = os.path.join(DATASET_DIR, "output.csv") if os.path.isdir(DATASET_DIR) else "output.csv"

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

FORWARDED_HIGH_THRESHOLD = 5  # forwarded_count >= this is "high" (chain-spam signal)

# Keyword lists (kept small and English + a few common Hindi-transliterated terms).
# Extend these over time; this is intentionally not exhaustive for v1.
OTP_PIN_PASSWORD_WORDS = ["otp", "pin", "password", "login code", "verification code", "6 digit"]
PAYMENT_ACTION_WORDS = ["pay", "scan", "qr", "transfer", "send screenshot", "processing fee", "clearance amount"]
URGENCY_WORDS = ["urgent", "immediately", "right away", "asap", "abhi", "jaldi", "aaj", "today", "now"]
CHAIN_PHRASES = ["forward this", "share this", "don't ignore", "do not ignore", "share with everyone",
                  "before midnight", "good luck", "blessing", "bhagwan", "positive energy"]
SHORTENED_URL_PATTERN = re.compile(r"\b(bit\.ly|tinyurl|t\.co|shorturl|link\.wame)\b", re.IGNORECASE)
DEADLINE_PATTERN = re.compile(
    r"\b(today|tonight|in \d+ ?min|within \d+ ?min|by \d{1,2}\s?(am|pm)?|"
    r"aaj|abhi|baje tak|jaldi)\b",
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r"@(\w+)")

# Prompt-injection phrases: checked against RAW, unnormalized text.
# These are messages attempting to instruct the router itself, not the user.
INJECTION_PHRASES = [
    "ignore previous",
    "ignore all previous",
    "routing override",
    "system note",
    "internal router metadata",
    "assistant instruction",
    "action=notify",
    "confidence=1",
    "mark this as notify",
    "mark as notify",
    "set action=",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_safe(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame. Returns an empty DataFrame (not an error)
    if the file is missing, since some dataset files are optional in principle.
    """
    if not os.path.exists(path):
        print(f"[warn] file not found, continuing with empty table: {path}", file=sys.stderr)
        return pd.DataFrame()
    return pd.read_csv(path)


def load_all_tables(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load every dataset CSV into a dict of DataFrames keyed by a short table name.
    """
    filenames = {
        "messages": "messages.csv",
        "users": "users.csv",
        "groups": "groups.csv",
        "group_members": "group_members.csv",
        "business_accounts": "business_accounts.csv",
        "user_business_history": "user_business_history.csv",
        "message_history": "message_history.csv",
        "message_events": "message_events.csv",
        "images": "images.csv",
        "voice_notes": "voice_notes.csv",
        "daily_notification_summary": "daily_notification_summary.csv",
    }
    tables = {}
    for key, fname in filenames.items():
        tables[key] = load_csv_safe(os.path.join(dataset_dir, fname))
    return tables


def build_indexes(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Build simple lookup dictionaries for O(1) access during context building.
    Every index gracefully degrades to an empty dict if the source table is empty.
    """
    idx: Dict[str, Any] = {}

    def to_index(df: pd.DataFrame, key_col: str) -> Dict[Any, dict]:
        if df.empty or key_col not in df.columns:
            return {}
        return df.set_index(key_col).to_dict("index")

    idx["user_by_id"] = to_index(tables["users"], "user_id")
    idx["business_by_id"] = to_index(tables["business_accounts"], "business_id")
    idx["group_by_id"] = to_index(tables["groups"], "group_id")

    # group membership keyed by (group_id, user_id)
    membership: Dict[Any, dict] = {}
    gm = tables["group_members"]
    if not gm.empty:
        for _, row in gm.iterrows():
            membership[(row["group_id"], row["user_id"])] = row.to_dict()
    idx["membership_by_group_user"] = membership

    # business history keyed by (user_id, business_id)
    bh_index: Dict[Any, dict] = {}
    bh = tables["user_business_history"]
    if not bh.empty:
        for _, row in bh.iterrows():
            bh_index[(row["user_id"], row["business_id"])] = row.to_dict()
    idx["business_history_by_user_business"] = bh_index

    # message events grouped by user_id (list of dict rows)
    events_by_user: Dict[Any, List[dict]] = {}
    me = tables["message_events"]
    if not me.empty:
        for user_id, group_df in me.groupby("user_id"):
            events_by_user[user_id] = group_df.to_dict("records")
    idx["events_by_user"] = events_by_user

    # image / voice file path lookups
    idx["image_by_id"] = to_index(tables["images"], "image_id")
    idx["voice_by_id"] = to_index(tables["voice_notes"], "voice_note_id")

    # --- Personalization indexes (new) ---
    history_by_sender: Dict[Any, List[dict]] = {}
    history_by_business: Dict[Any, List[dict]] = {}
    history_by_group: Dict[Any, List[dict]] = {}
    mh = tables["message_history"]
    if not mh.empty:
        for _, row in mh.iterrows():
            r = row.to_dict()
            uid = r.get("user_id")
            sid = r.get("sender_user_id")
            bid = r.get("business_id")
            gid = r.get("group_id")
            if uid is not None and sid is not None and pd.notna(sid):
                history_by_sender.setdefault((uid, sid), []).append(r)
            if uid is not None and bid is not None and pd.notna(bid):
                history_by_business.setdefault((uid, bid), []).append(r)
            if uid is not None and gid is not None and pd.notna(gid):
                history_by_group.setdefault((uid, gid), []).append(r)
    idx["history_by_user_sender"] = history_by_sender
    idx["history_by_user_business"] = history_by_business
    idx["history_by_user_group"] = history_by_group

    event_by_user_message: Dict[Any, dict] = {}
    me2 = tables["message_events"]
    if not me2.empty:
        for _, row in me2.iterrows():
            r = row.to_dict()
            event_by_user_message[(r.get("user_id"), r.get("message_id"))] = r
    idx["event_by_user_message"] = event_by_user_message

    return idx


# ---------------------------------------------------------------------------
# Context object
# ---------------------------------------------------------------------------

@dataclass
class MessageContext:
    """
    All information needed to route a single message, joined together in one
    place so downstream rule functions never need to touch raw DataFrames.
    """
    message_id: str
    user_id: str
    conversation_type: str
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: Optional[str]
    raw_message_text: str
    media_type: Optional[str]
    media_id: Optional[str]
    forwarded_count: int

    user_row: Optional[dict] = None
    group_row: Optional[dict] = None
    membership_row: Optional[dict] = None
    sender_membership_row: Optional[dict] = None
    business_row: Optional[dict] = None
    business_history_row: Optional[dict] = None
    sender_history_events: List[dict] = field(default_factory=list)

    media_extracted_text: Optional[str] = None  # filled by multimodal placeholder


def safe_get(d: Optional[dict], key: str, default=None):
    """Null-safe dict access used throughout context building."""
    if not d:
        return default
    val = d.get(key, default)
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return default
    return val


def build_context(row: pd.Series, idx: Dict[str, Any]) -> MessageContext:
    """
    Build a single MessageContext by joining a messages.csv row against every
    other table via the pre-built indexes.
    """
    user_id = row.get("user_id")
    group_id = row.get("group_id") if pd.notna(row.get("group_id")) else None
    business_id = row.get("business_id") if pd.notna(row.get("business_id")) else None
    sender_user_id = row.get("sender_user_id") if pd.notna(row.get("sender_user_id")) else None
    forwarded_count = row.get("forwarded_count")
    forwarded_count = int(forwarded_count) if pd.notna(forwarded_count) else 0
    message_text = row.get("message_text")
    message_text = message_text if isinstance(message_text, str) and pd.notna(message_text) else ""

    ctx = MessageContext(
        message_id=row.get("message_id"),
        user_id=user_id,
        conversation_type=row.get("conversation_type"),
        group_id=group_id,
        business_id=business_id,
        sender_user_id=sender_user_id,
        created_at=row.get("created_at"),
        raw_message_text=message_text,
        media_type=row.get("media_type") if pd.notna(row.get("media_type")) else None,
        media_id=row.get("media_id") if pd.notna(row.get("media_id")) else None,
        forwarded_count=forwarded_count,
    )

    ctx.user_row = idx["user_by_id"].get(user_id)
    ctx.business_row = idx["business_by_id"].get(business_id) if business_id else None
    ctx.group_row = idx["group_by_id"].get(group_id) if group_id else None
    ctx.membership_row = idx["membership_by_group_user"].get((group_id, user_id)) if group_id else None
    ctx.sender_membership_row = (
        idx["membership_by_group_user"].get((group_id, sender_user_id))
        if group_id and sender_user_id
        else None
    )
    ctx.business_history_row = (
        idx["business_history_by_user_business"].get((user_id, business_id)) if business_id else None
    )

    # Sender history: same user AND (same sender OR same business OR same group).
    all_user_events = idx["events_by_user"].get(user_id, [])
    ctx.sender_history_events = all_user_events  # v1: keep full list; filtering happens in personalization placeholder

    return ctx


# ---------------------------------------------------------------------------
# Placeholder: Multimodal extraction (OCR / ASR)
# ---------------------------------------------------------------------------

def extract_media_text(ctx: MessageContext, idx: Dict[str, Any]) -> str:
    """
    PLACEHOLDER for OCR (images) and ASR (voice notes).

    v1 behaviour: does not actually read media files. Always returns an empty
    string, which the rest of the pipeline correctly interprets as "degraded
    media" and falls back to metadata-only signals (sender, business, forward
    count) rather than crashing or guessing.

    TODO (future work):
        - if ctx.media_type == "image": run OCR on the resolved image path
          from idx["image_by_id"][ctx.media_id]["file_path"]
        - if ctx.media_type == "voice": run ASR on the resolved audio path
          from idx["voice_by_id"][ctx.media_id]["file_path"]
        - wrap both in try/except and always return "" on failure so callers
          never need to change.
    """
    if ctx.media_type not in ("image", "voice"):
        return ""
    # Placeholder: no real extraction yet.
    return ""


def get_effective_text(ctx: MessageContext) -> str:
    """
    Returns the text the rest of the pipeline should reason over: the raw
    message text, or extracted media text for image/voice messages.
    """
    if ctx.media_type in ("image", "voice"):
        return ctx.media_extracted_text or ""
    return ctx.raw_message_text or ""


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace. Used for keyword-category matching only
    (never used for prompt-injection detection, which runs on raw text)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def contains_any(text: str, keywords: List[str]) -> bool:
    """True if any keyword/phrase appears as a substring in text (case handled by caller)."""
    return any(kw in text for kw in keywords)


def domain_mismatch(business_row: Optional[dict]) -> Optional[bool]:
    """
    Returns True if sender domain differs from official domain, False if they
    match, or None if official_domain is missing (ambiguous / can't compare).
    """
    if not business_row:
        return None
    official = business_row.get("official_domain")
    used = business_row.get("domain_used_by_sender")
    if not isinstance(official, str) or official.strip() == "" or pd.isna(official):
        return None
    if not isinstance(used, str) or pd.isna(used):
        return None
    return official.strip().lower() != used.strip().lower()


def is_verified(business_row: Optional[dict]) -> bool:
    if not business_row:
        return False
    val = business_row.get("verified")
    return bool(val) and not pd.isna(val) and int(val) == 1


def mentions_user(text: str, user_id: str) -> bool:
    """Checks for an @user_id style direct mention in the text."""
    if not text or not user_id:
        return False
    return f"@{user_id}" in text


# ---------------------------------------------------------------------------
# Safety Engine (obvious scams / obvious spam)
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    action: str
    message_type: str
    rule_id: str
    reason: str
    confidence: float


def detect_injection(raw_text: str) -> bool:
    """
    Checks the RAW, unnormalized message text for prompt-injection style
    phrases attempting to instruct the router itself. This intentionally does
    NOT run on normalized text, since normalization could obscure the exact
    phrases (e.g. 'action=notify').
    """
    lowered = raw_text.lower()
    return any(phrase in lowered for phrase in INJECTION_PHRASES)


def evaluate_safety(ctx: MessageContext, effective_text: str) -> Optional[SafetyResult]:
    """
    Core deterministic safety rules. Ordered, first match wins.
    Only implements the "obvious" cases per the current scope:
        - OTP/PIN/password requests
        - spoofed/mismatched business domains
        - prompt injection attempts
        - high-forward-count chain/blessing spam
    Returns None if nothing obvious fired (message proceeds to Urgency/Default).
    """
    norm = normalize_text(effective_text)

    # Rule 1: prompt injection (checked on RAW text, highest priority safety trap)
    if detect_injection(ctx.raw_message_text):
        return SafetyResult(
            action="mute",
            message_type="scam",
            rule_id="safety:injection",
            reason="Message attempts to override routing instructions; muted for safety.",
            confidence=0.93,
        )

    # Rule 2: OTP / PIN / password request
    if contains_any(norm, OTP_PIN_PASSWORD_WORDS):
        return SafetyResult(
            action="mute",
            message_type="scam",
            rule_id="safety:credential_request",
            reason="Message requests OTP, PIN, or password verification; muted for safety.",
            confidence=0.9,
        )

    # Rule 3: spoofed / mismatched business domain
    mismatch = domain_mismatch(ctx.business_row)
    if mismatch is True:
        return SafetyResult(
            action="mute",
            message_type="scam",
            rule_id="safety:domain_mismatch",
            reason="Sender domain does not match the business's official domain.",
            confidence=0.9,
        )
    if mismatch is None and ctx.business_row is not None and not is_verified(ctx.business_row):
        return SafetyResult(
            action="mute",
            message_type="scam",
            rule_id="safety:missing_domain",
            reason="Unverified business with no listed official domain; treated as high risk.",
            confidence=0.85,
        )

    # Rule 4: suspicious/shortened link
    if SHORTENED_URL_PATTERN.search(norm):
        return SafetyResult(
            action="mute",
            message_type="scam",
            rule_id="safety:suspicious_link",
            reason="Message contains a shortened or suspicious link.",
            confidence=0.85,
        )

    # Rule 5: immediate payment demand (payment word + urgency word together)
    if contains_any(norm, PAYMENT_ACTION_WORDS) and contains_any(norm, URGENCY_WORDS):
        return SafetyResult(
            action="mute",
            message_type="scam",
            rule_id="safety:payment_fraud",
            reason="Message demands urgent payment or QR scan; muted for safety.",
            confidence=0.87,
        )

    # Rule 6: high forwarded-count chain/blessing spam
    if ctx.forwarded_count >= FORWARDED_HIGH_THRESHOLD and contains_any(norm, CHAIN_PHRASES):
        return SafetyResult(
            action="mute",
            message_type="spam",
            rule_id="safety:chain_forward",
            reason=f"Repeated chain-forward pattern (forwarded {ctx.forwarded_count} times).",
            confidence=0.85,
        )

    return None


# ---------------------------------------------------------------------------
# Urgency Engine (obvious urgency)
# ---------------------------------------------------------------------------

@dataclass
class UrgencyResult:
    score: float
    signals: List[str]


def evaluate_urgency(ctx: MessageContext, effective_text: str) -> UrgencyResult:
    """
    Small additive urgency scorer covering only the obvious signals:
        - admin sender in a group
        - verified business + delivery/appointment/refund language
        - direct @mention of the receiving user
        - same-day deadline language
    """
    norm = normalize_text(effective_text)
    signals: List[str] = []
    score = 0.0

    # Admin sender (uses the SENDER's membership row, not the receiver's)
    if ctx.sender_membership_row and str(ctx.sender_membership_row.get("role")) == "admin":
        signals.append("admin_sender")
        score += 0.25

    # Verified business + transactional language
    transactional_words = ["delivery", "appointment", "refund", "order", "pickup"]
    if is_verified(ctx.business_row) and contains_any(norm, transactional_words):
        signals.append("verified_transactional_business")
        score += 0.3

    # Direct mention
    if mentions_user(effective_text, ctx.user_id):
        signals.append("direct_mention")
        score += 0.25

    # Same-day deadline language
    if DEADLINE_PATTERN.search(norm):
        signals.append("same_day_deadline")
        score += 0.3

    return UrgencyResult(score=min(score, 1.0), signals=signals)


# ---------------------------------------------------------------------------
# Placeholder: Personalization Engine
# ---------------------------------------------------------------------------

@dataclass
class PersonalizationResult:
    engagement_score: float
    history_sample_size: int
    opted_out_flag: bool
    evidence_candidates: List[dict] = field(default_factory=list)
    # each candidate: {"message_id", "opened", "replied", "dismissed",
    #                  "muted", "reported", "created_at", "match_type"}

def _row_get(row: dict, key: str, default=None):
    """Null-safe helper for reading a value out of a history/event dict row."""
    if not row:
        return default
    val = row.get(key, default)
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return val

def evaluate_personalization(ctx: MessageContext, idx: Dict[str, Any]) -> PersonalizationResult:
    """
    Personalization engine with weighted match sources.

    Sender and business matches always count at full weight. Group matches
    are weighted down as group size grows. Thresholds are derived from the
    actual member_count distribution across groups.csv (natural gaps at
    ~50 and ~100 members separate small-community, mid-size, and
    broadcast-scale groups):

        member_count < 50   -> weight 1.0
        member_count < 100  -> weight 0.5
        member_count >= 100 -> weight 0.0 (excluded from scoring and evidence)

    Sparse history (fewer than 2 matched prior messages) is still reported,
    but flagged via history_sample_size so the Decision Engine won't let a
    single data point drive a routing decision.
    """
    user_id = ctx.user_id

    # Determine group weight once for this message's group (if any).
    group_weight = 1.0
    if ctx.group_id and ctx.group_row:
        member_count = _row_get(ctx.group_row, "member_count", 0) or 0
        try:
            member_count = int(member_count)
        except (TypeError, ValueError):
            member_count = 0
        if member_count >= 100:
            group_weight = 0.0
        elif member_count >= 50:
            group_weight = 0.5
        # else stays 1.0

    # O(1) bucket lookups, tagged with match_type before merging.
    tagged_rows: List[tuple] = []
    if ctx.sender_user_id:
        for r in idx["history_by_user_sender"].get((user_id, ctx.sender_user_id), []):
            tagged_rows.append((r, "sender", 1.0))
    if ctx.business_id:
        for r in idx["history_by_user_business"].get((user_id, ctx.business_id), []):
            tagged_rows.append((r, "business", 1.0))
    if ctx.group_id and group_weight > 0.0:
        for r in idx["history_by_user_group"].get((user_id, ctx.group_id), []):
            tagged_rows.append((r, "group", group_weight))

    # De-duplicate by message_id, preferring the highest-priority match_type
    # if the same message matched more than one bucket -- sender > business > group.
    priority = {"sender": 0, "business": 1, "group": 2}
    best_by_id: Dict[Any, tuple] = {}
    for r, match_type, weight in tagged_rows:
        mid = _row_get(r, "message_id")
        if not mid:
            continue
        if mid not in best_by_id or priority[match_type] < priority[best_by_id[mid][1]]:
            best_by_id[mid] = (r, match_type, weight)

    matched_history = list(best_by_id.values())

    # Opt-out check (unchanged, O(1)).
    opted_out = False
    if ctx.business_history_row is not None:
        opt_out_val = _row_get(ctx.business_history_row, "promotions_opted_out_at")
        opted_out = isinstance(opt_out_val, str) and opt_out_val.strip() != ""

    history_sample_size = len(matched_history)  # unweighted count, drives the sparse-history gate
    if history_sample_size == 0:
        return PersonalizationResult(
            engagement_score=0.0,
            history_sample_size=0,
            opted_out_flag=opted_out,
            evidence_candidates=[],
        )

    weighted_total = 0.0
    opened = replied = dismissed = muted = reported = 0.0
    candidates: List[dict] = []

    for h_row, match_type, weight in matched_history:
        mid = _row_get(h_row, "message_id")
        event = idx["event_by_user_message"].get((user_id, mid), {})

        is_opened = int(_row_get(event, "message_opened", 0) or 0)
        is_replied = int(_row_get(event, "message_replied", 0) or 0)
        is_dismissed = int(_row_get(event, "notification_dismissed", 0) or 0)
        is_muted = int(_row_get(event, "muted_after_message", 0) or 0)
        is_reported = int(_row_get(event, "message_reported", 0) or 0)

        opened += is_opened * weight
        replied += is_replied * weight
        dismissed += is_dismissed * weight
        muted += is_muted * weight
        reported += is_reported * weight
        weighted_total += weight

        candidates.append({
            "message_id": mid,
            "opened": is_opened,
            "replied": is_replied,
            "dismissed": is_dismissed,
            "muted": is_muted,
            "reported": is_reported,
            "created_at": _row_get(h_row, "created_at", ""),
            "match_type": match_type,
        })

    if weighted_total <= 0.0:
        # All matches were group-only in a broadcast-scale group and got zeroed out.
        return PersonalizationResult(
            engagement_score=0.0,
            history_sample_size=history_sample_size,
            opted_out_flag=opted_out,
            evidence_candidates=[],
        )

    opened_pct = opened / weighted_total
    replied_pct = replied / weighted_total
    dismissed_pct = dismissed / weighted_total
    muted_pct = muted / weighted_total
    reported_pct = reported / weighted_total

    raw_score = (
        opened_pct * 0.3
        + replied_pct * 0.4
        - dismissed_pct * 0.3
        - muted_pct * 0.5
        - reported_pct * 1.0
    )
    engagement_score = max(-1.0, min(1.0, raw_score))

    match_priority = {"sender": 0, "business": 1, "group": 2}
    candidates.sort(key=lambda c: str(c["created_at"]), reverse=True)
    candidates.sort(key=lambda c: match_priority.get(c["match_type"], 3))

    return PersonalizationResult(
        engagement_score=engagement_score,
        history_sample_size=history_sample_size,
        opted_out_flag=opted_out,
        evidence_candidates=candidates,
    )

# ---------------------------------------------------------------------------
# Business Trust Engine (lightweight, obvious-only version)
# ---------------------------------------------------------------------------

@dataclass
class TrustResult:
    label: str  # "trusted" | "suspicious" | "neutral"


def evaluate_business_trust(ctx: MessageContext) -> Optional[TrustResult]:
    """
    Minimal trust scoring: only flags the obvious cases. Anything not clearly
    trusted or clearly suspicious is "neutral" and does not gate routing.
    """
    if ctx.business_row is None:
        return None

    verified = is_verified(ctx.business_row)
    mismatch = domain_mismatch(ctx.business_row)
    reports = ctx.business_row.get("user_reports_30d") or 0
    sent = ctx.business_row.get("messages_sent_30d") or 1
    report_rate = (reports / sent) if sent else 0

    if verified and mismatch is False and report_rate <= 0.03:
        return TrustResult(label="trusted")
    if (not verified) and (mismatch is True or mismatch is None) and report_rate > 0.03:
        return TrustResult(label="suspicious")
    return TrustResult(label="neutral")


# ---------------------------------------------------------------------------
# Placeholder: Evidence Selection
# ---------------------------------------------------------------------------

def select_evidence(personalization: PersonalizationResult, decision_action: str) -> str:
    """
    Real evidence selector.

    Filters personalization's evidence_candidates to those consistent with the
    final decision's direction, then returns up to 3 message_ids (already
    ranked sender > business > group, then recency, by evaluate_personalization).

    - notify / digest -> cite candidates that were opened or replied to
    - mute             -> cite candidates that were dismissed, muted, or reported

    Returns "none" if there is no consistent supporting evidence, rather than
    fabricating or citing contradictory history.
    """
    if not personalization.evidence_candidates:
        return "none"

    if decision_action in ("notify", "digest"):
        relevant = [c for c in personalization.evidence_candidates if c["opened"] or c["replied"]]
    elif decision_action == "mute":
        relevant = [
            c for c in personalization.evidence_candidates
            if c["dismissed"] or c["muted"] or c["reported"]
        ]
    else:
        relevant = []

    if not relevant:
        return "none"

    top = relevant[:3]
    return ";".join(str(c["message_id"]) for c in top)


# ---------------------------------------------------------------------------
# Message type inference (shared helper, used by every rule)
# ---------------------------------------------------------------------------

def infer_type(ctx: MessageContext, effective_text: str, action: str = "digest") -> str:
    """
    Shared, best-effort message_type classifier used by every non-safety
    routing rule.

    `action` is now required so that the "urgent" label is only ever applied
    when the message was actually routed to notify -- otherwise a message
    could be typed "urgent" while simultaneously being sent to digest/mute,
    which is a contradictory, self-defeating output.
    """
    norm = normalize_text(effective_text)

    if ctx.conversation_type == "business":
        if contains_any(norm, ["offer", "discount", "% off", "sale", "promo"]):
            return "promotion"
        if contains_any(norm, ["refund", "payment", "invoice", "due"]):
            return "payment"
        return "business_update"

    if contains_any(norm, ["good morning", "good night", "blessed", "smile today"]):
        return "greeting"
    if ctx.forwarded_count >= FORWARDED_HIGH_THRESHOLD:
        return "forward"

    # Only label as "urgent" if the decision actually was to notify.
    if action == "notify" and (DEADLINE_PATTERN.search(norm) or mentions_user(effective_text, ctx.user_id)):
        return "urgent"

    if contains_any(norm, ["meeting", "reminder", "schedule", "event", "closes"]):
        return "event"
    if ctx.conversation_type == "personal":
        return "personal" if norm.strip() else "unknown"

    return "unknown"


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    action: str
    message_type: str
    reason: str
    confidence: float


def decide(
    ctx: MessageContext,
    effective_text: str,
    safety: Optional[SafetyResult],
    urgency: UrgencyResult,
    personalization: PersonalizationResult,
    trust: Optional[TrustResult],
) -> DecisionResult:
    """
    Ordered deterministic cascade. First applicable branch wins.
    Safety always runs first and is never overridden by personalization.
    """
    # 1. Safety gate (unchanged, always first, never bypassed)
    if safety is not None:
        return DecisionResult(
            action=safety.action,
            message_type=safety.message_type,
            reason=safety.reason,
            confidence=safety.confidence,
        )

    # 2. Opt-out OR strong negative engagement with enough history to trust it.
    strong_negative_engagement = (
        personalization.history_sample_size >= 2 and personalization.engagement_score <= -0.4
    )
    if personalization.opted_out_flag or strong_negative_engagement:
        reason = (
            "User has opted out of promotional messages from this business."
            if personalization.opted_out_flag
            else "User has consistently dismissed or muted similar messages from this sender/business/group."
        )
        return DecisionResult(
            action="mute",
            message_type=infer_type(ctx, effective_text, action="mute"),
            reason=reason,
            confidence=0.82,
        )

    # 3. Obvious urgency -> notify
    if urgency.score >= 0.55:
        top_signal = urgency.signals[0] if urgency.signals else "urgency signals"
        return DecisionResult(
            action="notify",
            message_type=infer_type(ctx, effective_text, action="notify"),
            reason=f"Message shows strong urgency signals ({top_signal}).",
            confidence=0.85,
        )

    # 4. Trusted business with mild urgency -> notify
    if trust is not None and trust.label == "trusted" and urgency.score >= 0.25:
        return DecisionResult(
            action="notify",
            message_type=infer_type(ctx, effective_text, action="notify"),
            reason="Verified business update relevant to the user.",
            confidence=0.75,
        )

    # 5. Suspicious business (not caught by hard Safety rules) -> mute
    if trust is not None and trust.label == "suspicious":
        return DecisionResult(
            action="mute",
            message_type="spam",
            reason="Sender business shows signs of risk (unverified or high report rate).",
            confidence=0.78,
        )

    # 5b. Positive engagement with enough history -> digest
    if personalization.history_sample_size >= 2 and personalization.engagement_score >= 0.3:
        return DecisionResult(
            action="digest",
            message_type=infer_type(ctx, effective_text, action="digest"),
            reason="User typically engages with messages like this; not urgent enough to interrupt.",
            confidence=0.72,
        )

    # 6. Sensible default (unchanged)
    if ctx.conversation_type == "personal":
        return DecisionResult(
            action="digest",
            message_type=infer_type(ctx, effective_text, action="digest"),
            reason="No strong urgency or risk signal; personal message defaulted to digest.",
            confidence=0.6,
        )

    return DecisionResult(
        action="digest",
        message_type=infer_type(ctx, effective_text, action="digest"),
        reason="No strong signal either way; defaulting to digest based on conversation context.",
        confidence=0.6,
    )

# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def route_message(row: pd.Series, idx: Dict[str, Any]) -> Dict[str, Any]:
    ctx = build_context(row, idx)
    ctx.media_extracted_text = extract_media_text(ctx, idx)
    effective_text = get_effective_text(ctx)

    safety = evaluate_safety(ctx, effective_text)
    urgency = evaluate_urgency(ctx, effective_text)
    personalization = evaluate_personalization(ctx, idx)          # <-- now takes idx
    trust = evaluate_business_trust(ctx)

    decision = decide(ctx, effective_text, safety, urgency, personalization, trust)
    evidence = select_evidence(personalization, decision.action)  # <-- new signature

    return {
        "message_id": ctx.message_id,
        "action": decision.action,
        "message_type": decision.message_type,
        "reason": decision.reason,
        "confidence": round(decision.confidence, 2),
        "evidence_message_ids": evidence,
    }


def run_pipeline(dataset_dir: str, output_path: str) -> None:
    """
    Top-level entry point: loads data, routes every message, writes output.csv.
    """
    print(f"Loading dataset from: {dataset_dir}")
    tables = load_all_tables(dataset_dir)

    if tables["messages"].empty:
        print("[error] messages.csv could not be loaded or is empty. Aborting.", file=sys.stderr)
        sys.exit(1)

    idx = build_indexes(tables)

    results = []
    total = len(tables["messages"])
    for i, row in tables["messages"].iterrows():
        try:
            result = route_message(row, idx)
        except Exception as exc:  # noqa: BLE001 - never let one bad row kill the run
            print(f"[warn] failed to route message_id={row.get('message_id')}: {exc}", file=sys.stderr)
            result = {
                "message_id": row.get("message_id"),
                "action": "digest",
                "message_type": "unknown",
                "reason": "Fallback: an error occurred while routing this message.",
                "confidence": 0.5,
                "evidence_message_ids": "none",
            }
        results.append(result)

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    # Sanity checks before writing
    assert len(out_df) == total, f"Expected {total} rows, got {len(out_df)}"
    assert list(out_df.columns) == OUTPUT_COLUMNS, "Output columns do not match required schema"

    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    run_pipeline(DATASET_DIR, OUTPUT_PATH)
