# app.py
# QA Radar Suite (Streamlit) — Trust / Risk / Quality Discovery + Judgment Mode
# Requirements (put in requirements.txt):
# streamlit==1.37.0
# requests==2.32.3
# beautifulsoup4==4.12.3

import re
import time
import json
import hashlib
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set

import requests
import streamlit as st
from bs4 import BeautifulSoup

# ----------------------------
# Configuration / Guardrails
# ----------------------------

DEFAULT_UA = "QA-Radar/2.0 (+trust-risk-discovery; human-in-the-loop)"
REQUEST_TIMEOUT = 12
MAX_PAGES_DEFAULT = 10
MAX_DEPTH_DEFAULT = 2
CRAWL_DELAY_SECONDS = 0.6  # polite
MAX_LINKS_PER_PAGE = 80

# Hard exclusions for safety/scope (we never interact with these)
BLOCKED_PATH_HINTS = [
    "login", "signin", "sign-in", "sign_in", "signon",
    "logout", "register", "signup", "sign-up",
    "checkout", "cart", "basket", "payment", "billing",
    "account", "profile", "settings",
    "oauth", "authorize", "auth", "sso",
    "password", "reset", "forgot",
    "subscribe", "plan", "pricing/checkout",
    "wp-admin", "admin"
]

SENSITIVE_QUERY_KEYS = {"token", "session", "auth", "code", "state", "password"}


# ----------------------------
# Data Models
# ----------------------------

@dataclass
class PageFetch:
    url: str
    status: Optional[int]
    elapsed_ms: Optional[int]
    content_type: Optional[str]
    error: Optional[str] = None
    title: Optional[str] = None
    h1_count: Optional[int] = None
    has_form: Optional[bool] = None
    has_password_field: Optional[bool] = None
    has_checkout_terms: Optional[bool] = None
    meta_desc_len: Optional[int] = None
    a_count: Optional[int] = None


@dataclass
class RiskSignal:
    code: str
    label: str
    detail: str


@dataclass
class DomainFinding:
    trust_domain: str
    endpoint: str
    observations: List[RiskSignal]
    interpretive_analysis: Optional[str]
    confidence_band: str
    senior_review_prompt: str
    scope_control: str
    client_ready_justification: str


@dataclass
class Report:
    origin: str
    audit_scope: str
    discovery_health: str
    confidence_level: str
    archetype: str
    endpoints_detected: int
    trust_domains_covered: int
    pages: List[PageFetch]
    findings: List[DomainFinding]
    doctrine_disclaimer: str
    generated_at_epoch: int


# ----------------------------
# Utility Functions
# ----------------------------

def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    # drop fragments
    parsed = parsed._replace(fragment="")
    # normalize path
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    parsed = parsed._replace(path=path)
    return urllib.parse.urlunparse(parsed)


def same_origin(a: str, b: str) -> bool:
    pa = urllib.parse.urlparse(a)
    pb = urllib.parse.urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def safe_url(url: str) -> bool:
    """Scope safety: avoid sensitive query keys; avoid blocked paths."""
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return False

    # block bad schemes
    if p.scheme not in ("http", "https"):
        return False

    # block sensitive query keys
    q = urllib.parse.parse_qs(p.query)
    for k in q.keys():
        if k.lower() in SENSITIVE_QUERY_KEYS:
            return False

    # path hints
    path = (p.path or "").lower()
    for hint in BLOCKED_PATH_HINTS:
        if hint in path:
            return False

    return True


def make_abs(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
        return None
    absu = urllib.parse.urljoin(base, href)
    absu = normalize_url(absu)
    # remove common tracking params
    parsed = urllib.parse.urlparse(absu)
    q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    q = [(k, v) for (k, v) in q if not k.lower().startswith("utm_")]
    parsed = parsed._replace(query=urllib.parse.urlencode(q))
    return urllib.parse.urlunparse(parsed)


def get_robots_parser(origin: str) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urllib.parse.urljoin(origin, "/robots.txt")
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        # treat as unknown -> allow, but remain polite
        pass
    return rp


def allowed_by_robots(rp: urllib.robotparser.RobotFileParser, url: str, ua: str) -> bool:
    try:
        return rp.can_fetch(ua, url)
    except Exception:
        return True


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


# ----------------------------
# Fetch + Parse
# ----------------------------

def fetch_page(session: requests.Session, url: str, ua: str) -> PageFetch:
    headers = {"User-Agent": ua, "Accept": "text/html,application/xhtml+xml"}
    try:
        t0 = time.time()
        r = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        elapsed_ms = int((time.time() - t0) * 1000)
        ct = r.headers.get("content-type", "")
        pf = PageFetch(
            url=r.url,
            status=r.status_code,
            elapsed_ms=elapsed_ms,
            content_type=ct,
        )

        if "text/html" not in ct.lower():
            return pf

        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else None
        h1_count = len(soup.find_all("h1"))
        meta_desc = soup.find("meta", attrs={"name": "description"})
        meta_len = len(meta_desc.get("content", "").strip()) if meta_desc and meta_desc.get("content") else 0

        forms = soup.find_all("form")
        has_form = len(forms) > 0
        has_password = soup.find("input", attrs={"type": "password"}) is not None

        text_lower = soup.get_text(" ", strip=True).lower()
        checkout_terms = any(t in text_lower for t in ["checkout", "card number", "cvv", "billing address", "payment"])

        a_tags = soup.find_all("a")
        pf.title = title
        pf.h1_count = h1_count
        pf.has_form = has_form
        pf.has_password_field = has_password
        pf.has_checkout_terms = checkout_terms
        pf.meta_desc_len = meta_len
        pf.a_count = len(a_tags)

        return pf

    except requests.exceptions.RequestException as e:
        return PageFetch(url=url, status=None, elapsed_ms=None, content_type=None, error=str(e))
    except Exception as e:
        return PageFetch(url=url, status=None, elapsed_ms=None, content_type=None, error=f"Parse error: {e}")


def extract_internal_links(html_text: str, base_url: str, origin: str) -> List[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        absu = make_abs(base_url, a.get("href"))
        if not absu:
            continue
        if not same_origin(origin, absu):
            continue
        if not safe_url(absu):
            continue
        links.append(absu)
        if len(links) >= MAX_LINKS_PER_PAGE:
            break
    # de-dupe while preserving order
    seen: Set[str] = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ----------------------------
# Discovery Engine (Bounded)
# ----------------------------

def bounded_discovery(origin_url: str, max_pages: int, max_depth: int, ua: str) -> Tuple[List[PageFetch], List[str], str]:
    """
    Returns: (page_fetches, discovered_urls, discovery_note)
    """
    origin_url = normalize_url(origin_url)
    origin_parsed = urllib.parse.urlparse(origin_url)
    origin = f"{origin_parsed.scheme}://{origin_parsed.netloc}"

    rp = get_robots_parser(origin)
    session = requests.Session()

    queue: List[Tuple[str, int]] = [(origin_url, 0)]
    visited: Set[str] = set()
    pages: List[PageFetch] = []
    discovered: List[str] = []

    discovery_note = ""
    blocked_by_robots = 0

    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)
        url = normalize_url(url)

        if url in visited:
            continue
        visited.add(url)

        if not safe_url(url):
            continue
        if not allowed_by_robots(rp, url, ua):
            blocked_by_robots += 1
            continue

        pf = fetch_page(session, url, ua)
        pages.append(pf)
        discovered.append(pf.url)

        # only parse HTML pages for further links
        if pf.status and pf.status < 400 and pf.content_type and "text/html" in (pf.content_type or "").lower():
            # fetch body again for link extraction (only if no error and HTML)
            try:
                headers = {"User-Agent": ua, "Accept": "text/html,application/xhtml+xml"}
                r = session.get(pf.url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
                if "text/html" in r.headers.get("content-type", "").lower():
                    if depth < max_depth:
                        links = extract_internal_links(r.text, pf.url, origin)
                        for link in links:
                            if link not in visited and len(queue) < (max_pages * 6):
                                queue.append((link, depth + 1))
            except Exception:
                pass

        time.sleep(CRAWL_DELAY_SECONDS)

    if blocked_by_robots > 0:
        discovery_note = f"Robots.txt restricted {blocked_by_robots} URL(s) during discovery."

    return pages, discovered, discovery_note


# ----------------------------
# Trust Domain Classification
# ----------------------------

def classify_trust_domain(url: str) -> str:
    p = urllib.parse.urlparse(url)
    path = (p.path or "/").lower()

    # Support Reliability
    if any(k in path for k in ["/help", "/support", "/contact", "/faq", "/docs", "/knowledge", "/kb"]):
        return "Support Reliability"

    # Transaction Safety
    if any(k in path for k in ["/pricing", "/plans", "/subscribe", "/order", "/purchase", "/buy"]):
        return "Transaction Safety"

    # Legal / Compliance
    if any(k in path for k in ["/privacy", "/terms", "/legal", "/cookies", "/gdpr", "/security"]):
        return "Legal / Compliance"

    # Brand Credibility (default)
    return "Brand Credibility"


def detect_signals(pf: PageFetch) -> List[RiskSignal]:
    signals: List[RiskSignal] = []

    if pf.error:
        signals.append(RiskSignal(
            code="FETCH_ERROR",
            label="Fetch error",
            detail=f"Could not retrieve page: {pf.error}"
        ))
        return signals

    if pf.status is None:
        signals.append(RiskSignal("NO_STATUS", "No HTTP status", "Request failed before receiving a response."))
        return signals

    if pf.status >= 400:
        signals.append(RiskSignal(
            code="HTTP_ERROR",
            label="HTTP error response",
            detail=f"Returned HTTP {pf.status}."
        ))

    # performance signal (very rough)
    if pf.elapsed_ms is not None and pf.elapsed_ms >= 2500:
        signals.append(RiskSignal(
            code="SLOW_RESPONSE",
            label="Slow response time",
            detail=f"Approx response time {pf.elapsed_ms}ms."
        ))

    # structural signals
    if pf.h1_count is not None and pf.h1_count == 0:
        signals.append(RiskSignal(
            code="NO_H1",
            label="No H1 detected",
            detail="No <h1> found; may be intentional brand-led structure or an accessibility/structure gap."
        ))
    if pf.h1_count is not None and pf.h1_count >= 3:
        signals.append(RiskSignal(
            code="MANY_H1",
            label="Non-standard heading hierarchy",
            detail=f"{pf.h1_count} H1 tags detected; may reduce clarity for some users/assistive tech."
        ))

    # metadata signal
    if pf.meta_desc_len is not None and pf.meta_desc_len == 0:
        signals.append(RiskSignal(
            code="NO_META_DESC",
            label="Missing meta description",
            detail="Meta description appears missing; may reduce clarity in search previews (brand trust surface)."
        ))

    # form sensitivity (we do NOT test forms; we flag scope constraints)
    if pf.has_password_field:
        signals.append(RiskSignal(
            code="SENSITIVE_FORM_PRESENT",
            label="Sensitive form present (scope guarded)",
            detail="Password field detected. Manual QA should proceed with explicit permission and strict scope."
        ))
    if pf.has_checkout_terms:
        signals.append(RiskSignal(
            code="PAYMENT_LANGUAGE_DETECTED",
            label="Transaction language detected (no payment testing)",
            detail="Checkout/payment-related terms detected. Tool will not interact with payments."
        ))

    return signals


def infer_archetype(pages: List[PageFetch]) -> str:
    # Light heuristic: if we see pricing/legal/support surfaces -> claim-heavy / regulated leaning
    urls = " ".join([p.url for p in pages if p.url])
    hit = sum(1 for k in ["/privacy", "/terms", "/legal", "/security", "/pricing", "/help", "/support"] if k in urls.lower())
    if hit >= 2:
        return "Claim-Heavy / Regulated"
    return "Standard Product / Marketing"


def discovery_health(pages: List[PageFetch], discovery_note: str) -> Tuple[str, str]:
    """
    Returns: (health, confidence_level)
    """
    if not pages:
        return "Limited", "Low"

    ok_html = 0
    hard_errors = 0

    for p in pages:
        if p.error or p.status is None:
            hard_errors += 1
            continue
        if p.status < 400 and p.content_type and "text/html" in (p.content_type or "").lower():
            ok_html += 1

    if ok_html == 0:
        return "Limited", "Low"

    if ok_html >= 6 and hard_errors == 0 and not discovery_note:
        return "High", "High"

    if ok_html >= 3:
        return "Medium", "Moderate"

    return "Limited", "Low"


# ----------------------------
# Judgment Mode (Doctrine-aligned)
# ----------------------------

DOCTRINE_DISCLAIMER = (
    "THIS SYSTEM PROVIDES DISCOVERY-LEVEL INTELLIGENCE TO SUPPORT SENIOR QA JUDGMENT. "
    "IT DOES NOT ISSUE FINAL ASSESSMENTS, SEVERITY RATINGS, OR REMEDIATION DIRECTIVES. "
    "FINAL AUTHORITY RESTS WITH THE HUMAN AUDITOR."
)

SCOPE_CONTROL_TEXT = (
    "Out of scope: internal CSS naming, minor layout preferences, non-user-impacting structural preferences. "
    "No login, no payment interactions, no credential entry."
)

def confidence_band_for_signals(signals: List[RiskSignal], health: str) -> str:
    # conservative bands; we never over-claim
    if health == "Limited":
        return "Low"
    if any(s.code in {"FETCH_ERROR", "HTTP_ERROR"} for s in signals):
        return "Low"
    if any(s.code in {"SLOW_RESPONSE"} for s in signals):
        return "Moderate"
    if any(s.code in {"MANY_H1", "NO_H1", "NO_META_DESC"} for s in signals):
        return "Moderate"
    return "Moderate" if health == "Medium" else "Moderate"


def interpretive_layer(signals: List[RiskSignal], health: str, trust_domain: str) -> str:
    # Human-sounding, measured interpretive note (no alarmism)
    if health == "Limited":
        return "Visibility is constrained; treat signals as indicative. Prefer manual validation before any claims."
    if not signals:
        return "Stability: no immediate trust-degrading signals detected within discovery scope. Intentionality: appears aligned with archetype expectations."
    # summarize most meaningful signals
    labels = [s.label for s in signals[:3]]
    return f"Signal set suggests potential review areas ({', '.join(labels)}). Prioritise only if user-impact path is credible."


def senior_prompt(trust_domain: str, health: str) -> str:
    if trust_domain == "Support Reliability":
        return "Senior Review Prompt: If a user hits a blocker, is the support path friction-free with clear escalation and resolution expectations?"
    if trust_domain == "Transaction Safety":
        return "Senior Review Prompt: Are pricing/plan signals evidence-parity aligned (what is promised vs. what is deliverable) without implying unsafe payment handling?"
    if trust_domain == "Legal / Compliance":
        return "Senior Review Prompt: Are privacy/terms/cookie signals clear, accessible, and consistent with the site's behavioural claims?"
    # Brand Credibility
    if health == "Limited":
        return "Senior Review Prompt: Given constrained visibility, is the limited signal set intentional (brand positioning), or is it an avoidable barrier to first-time trust?"
    return "Senior Review Prompt: Does the density of claims require clearer evidence-parity surfaces (proof, constraints, disclaimers) within visible scope?"


def client_justification(trust_domain: str, signals: List[RiskSignal]) -> str:
    if not signals:
        return f"Indicative signal in {trust_domain}: baseline consistency within discovery scope; worth confirming key trust surfaces during manual review."
    # keep it calm and defensible
    return f"Indicative signals in {trust_domain}; potential impact on perceived trust if expectations are misaligned. Manual validation recommended."


def build_report(origin: str, pages: List[PageFetch], discovery_note: str, judgment_mode: bool, audit_scope: str) -> Report:
    health, conf = discovery_health(pages, discovery_note)
    archetype = infer_archetype(pages)

    # Build findings per trust domain using up to 1 representative endpoint each (keeps output tight)
    by_domain: Dict[str, List[PageFetch]] = {}
    for p in pages:
        if not p.url:
            continue
        d = classify_trust_domain(p.url)
        by_domain.setdefault(d, []).append(p)

    # pick top representative endpoint per domain: prefer 200 HTML, else first
    findings: List[DomainFinding] = []
    for domain_name, plist in by_domain.items():
        rep = None
        for p in plist:
            if p.status and p.status < 400 and p.content_type and "text/html" in (p.content_type or "").lower():
                rep = p
                break
        rep = rep or plist[0]

        signals = detect_signals(rep)

        # Scope guard: if sensitive form present, keep observation but do NOT prescribe actions beyond manual review
        conf_band = confidence_band_for_signals(signals, health)

        interpret = None
        if judgment_mode:
            interpret = interpretive_layer(signals, health, domain_name)

        findings.append(DomainFinding(
            trust_domain=domain_name,
            endpoint=rep.url,
            observations=signals if judgment_mode else signals,  # always include observations; just changes narrative sections
            interpretive_analysis=interpret,
            confidence_band=conf_band if judgment_mode else conf_band,
            senior_review_prompt=senior_prompt(domain_name, health),
            scope_control=SCOPE_CONTROL_TEXT,
            client_ready_justification=client_justification(domain_name, signals),
        ))

    return Report(
        origin=origin,
        audit_scope=audit_scope,
        discovery_health=health,
        confidence_level=conf,
        archetype=archetype,
        endpoints_detected=len({p.url for p in pages if p.url}),
        trust_domains_covered=len(by_domain.keys()),
        pages=pages,
        findings=findings,
        doctrine_disclaimer=DOCTRINE_DISCLAIMER,
        generated_at_epoch=int(time.time())
    )


def report_to_markdown(r: Report, discovery_note: str) -> str:
    lines = []
    lines.append("# Trust Risk Discovery Brief (PRE-AUDIT)")
    lines.append("")
    lines.append("Discovery-level intelligence designed to support senior QA judgment. Bounded origin discovery only.")
    lines.append("")
    lines.append(f"**AUDIT SCOPE:** {r.audit_scope}")
    lines.append(f"**ORIGIN:** {r.origin}")
    lines.append("")
    lines.append("## DISCOVERY HEALTH")
    lines.append(f"- Health: **{r.discovery_health}**")
    lines.append(f"- Confidence Level: **{r.confidence_level}**")
    if discovery_note:
        lines.append(f"- Note: {discovery_note}")
    lines.append("")
    lines.append("## SITE SUMMARY")
    lines.append(f"- Automated discovery identified **{r.endpoints_detected}** endpoint(s) across **{r.trust_domains_covered}** trust domain(s).")
    lines.append(f"- Archetype: **{r.archetype}**")
    lines.append("")
    lines.append("## TRUST DOMAIN FINDINGS")
    for f in r.findings:
        lines.append("")
        lines.append(f"### {f.trust_domain}")
        lines.append(f"**Endpoint:** {f.endpoint}")
        if f.observations:
            lines.append("**Risk Signals (Observations):**")
            for s in f.observations:
                lines.append(f"- {s.label}: {s.detail}")
        else:
            lines.append("**Risk Signals (Observations):** None detected within discovery scope.")
        if f.interpretive_analysis:
            lines.append("")
            lines.append("**Interpretive Analysis (Judgment Mode):**")
            lines.append(f"- {f.interpretive_analysis}")
            lines.append(f"- Judgment Confidence Band: **{f.confidence_band}**")
        lines.append("")
        lines.append(f"**Senior Review Prompt:** {f.senior_review_prompt}")
        lines.append("")
        lines.append(f"**Scope Control (What Not To Fix):** {f.scope_control}")
        lines.append("")
        lines.append(f"**Client-Ready Justification:** {f.client_ready_justification}")
    lines.append("")
    lines.append("---")
    lines.append(r.doctrine_disclaimer)
    return "\n".join(lines)


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="QA Radar Suite — Trust Risk Discovery", layout="wide")

st.title("QA Radar Suite — Trust Risk Discovery")
st.caption("Discovery-level intelligence designed to support senior QA judgment. Bounded origin discovery only.")

with st.sidebar:
    st.subheader("Controls")
    judgment_mode = st.toggle("Judgment Mode", value=True, help="Adds interpretive reasoning + confidence bands + senior prompts.")
    ua = st.text_input("User-Agent", value=DEFAULT_UA)
    max_pages = st.slider("Max pages (bounded)", 1, 25, MAX_PAGES_DEFAULT)
    max_depth = st.slider("Max depth", 0, 4, MAX_DEPTH_DEFAULT)
    st.divider()
    st.caption("Safety / scope guardrails are enforced: no login, no payments, no credential paths.")

audit_scope = st.text_input(
    "Audit Scope (paste-ready)",
    value="Homepage, Pricing, Trust, and Primary Conversion flows. Origin-based bounded discovery."
)

col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    url_in = st.text_input("Target origin URL", placeholder="https://example.com")
with col2:
    run = st.button("Run Discovery", type="primary", use_container_width=True)

if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "last_md" not in st.session_state:
    st.session_state.last_md = ""
if "history" not in st.session_state:
    st.session_state.history = []

if run:
    target = normalize_url(url_in)
    if not target:
        st.error("Please enter a URL.")
    else:
        with st.status("Running bounded discovery (polite crawl)…", expanded=True) as status:
            st.write("Applying scope guardrails and robots.txt respect.")
            pages, discovered_urls, note = bounded_discovery(
                origin_url=target,
                max_pages=max_pages,
                max_depth=max_depth,
                ua=ua
            )
            status.update(label="Discovery complete.", state="complete", expanded=False)

        origin_parsed = urllib.parse.urlparse(target)
        origin = f"{origin_parsed.scheme}://{origin_parsed.netloc}"

        report = build_report(
            origin=origin,
            pages=pages,
            discovery_note=note,
            judgment_mode=judgment_mode,
            audit_scope=audit_scope
        )
        md = report_to_markdown(report, note)

        st.session_state.last_report = report
        st.session_state.last_md = md

        # history entry
        st.session_state.history.append({
            "origin": origin,
            "generated_at": report.generated_at_epoch,
            "discovery_health": report.discovery_health,
            "confidence": report.confidence_level,
            "endpoints": report.endpoints_detected,
            "domains": report.trust_domains_covered,
            "id": fingerprint(origin + str(report.generated_at_epoch))
        })

# Output
report: Optional[Report] = st.session_state.last_report

if report is None:
    st.info("Enter a URL and click **Run Discovery**. This produces a consultant-grade discovery brief (not a final audit).")
else:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Executive Snapshot")
        a, b, c, d = st.columns(4)
        a.metric("Discovery Health", report.discovery_health)
        b.metric("Confidence", report.confidence_level)
        c.metric("Endpoints", str(report.endpoints_detected))
        d.metric("Trust Domains", str(report.trust_domains_covered))

        st.caption(f"Archetype: **{report.archetype}**")
        st.caption("Reminder: This is discovery-level intelligence. No severity ratings. Human review required.")

        st.subheader("Findings (Trust Domains)")
        for f in report.findings:
            with st.expander(f"{f.trust_domain} — {f.endpoint}", expanded=True):
                if f.observations:
                    st.markdown("**Risk Signals (Observations)**")
                    for s in f.observations:
                        st.write(f"• **{s.label}** — {s.detail}")
                else:
                    st.write("No immediate trust-degrading signals detected within discovery scope.")

                if judgment_mode and f.interpretive_analysis:
                    st.markdown("**Interpretive Analysis (Judgment Mode)**")
                    st.write(f.interpretive_analysis)
                    st.write(f"**Judgment Confidence Band:** {f.confidence_band}")

                st.markdown("**Senior Review Prompt**")
                st.write(f.senior_review_prompt)

                st.markdown("**Scope Control (What Not To Fix)**")
                st.write(f.scope_control)

                st.markdown("**Client-Ready Justification**")
                st.write(f.client_ready_justification)

        st.subheader("Raw Crawl Log (Pages)")
        page_rows = [asdict(p) for p in report.pages]
        st.dataframe(page_rows, use_container_width=True)

    with right:
        st.subheader("Client-Ready Brief (Copy / Download)")
        st.text_area("Markdown Output", value=st.session_state.last_md, height=520)

        md_bytes = st.session_state.last_md.encode("utf-8")
        st.download_button(
            "Download Brief (.md)",
            data=md_bytes,
            file_name=f"qa_radar_discovery_{fingerprint(report.origin)}.md",
            mime="text/markdown",
            use_container_width=True
        )

        json_bytes = json.dumps(asdict(report), indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "Download Raw Data (.json)",
            data=json_bytes,
            file_name=f"qa_radar_discovery_{fingerprint(report.origin)}.json",
            mime="application/json",
            use_container_width=True
        )

        st.subheader("History (This Session)")
        if st.session_state.history:
            st.dataframe(st.session_state.history[::-1], use_container_width=True)
        else:
            st.caption("No scans yet in this session.")

        st.divider()
        st.caption(DOCTRINE_DISCLAIMER)
