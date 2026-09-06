/* ---------------- Shared nav + chrome ---------------- */
const NAV_ITEMS = [
  { href: "index.html", key: "dashboard", icon: "ti-layout-dashboard", label: "Dashboard" },
  { href: "screening.html", key: "screening", icon: "ti-scan", label: "New screening" },
  { href: "history.html", key: "history", icon: "ti-history", label: "Screening history" },
  { href: "identity.html", key: "identity", icon: "ti-affiliate", label: "Identity intelligence" },
  { href: "fraud.html", key: "fraud", icon: "ti-shield-exclamation", label: "Fraud cases" },
  { href: "checkpoints.html", key: "checkpoints", icon: "ti-map-pin", label: "Checkpoints" },
  { href: "audit.html", key: "audit", icon: "ti-file-clock", label: "Audit logs" },
  { href: "admin.html", key: "admin", icon: "ti-settings", label: "Admin settings" },
];

function renderChrome({ active, title, subtitle }) {
  const navHtml = NAV_ITEMS.map(n => `
    <a href="${n.href}" class="${n.key === active ? "active" : ""}">
      <i class="ti ${n.icon}"></i>${n.label}
    </a>`).join("");

  document.getElementById("sidebar").innerHTML = `
    <div class="brand">
      <div class="brand-mark"><i class="ti ti-shield-check"></i></div>
      <div>
        <div class="brand-name">BorderShield</div>
        <div class="brand-sub">AI DOCUMENT SCREENING</div>
      </div>
    </div>
    <div class="nav">${navHtml}</div>
    <div class="officer-card">
      <div class="officer-inner">
        <div class="avatar">RS</div>
        <div>
          <div class="officer-name">R. Sharma</div>
          <div class="officer-role">Immigration officer</div>
        </div>
      </div>
    </div>`;

  document.getElementById("topbar").innerHTML = `
    <div style="display:flex;align-items:center;gap:6px;">
      <button class="menu-btn" onclick="document.getElementById('sidebar').classList.add('open');document.getElementById('scrim').style.display='block';">
        <i class="ti ti-menu-2"></i>
      </button>
      <div>
        <h1 class="page-title">${title}</h1>
        ${subtitle ? `<p class="page-sub">${subtitle}</p>` : ""}
      </div>
    </div>
    <div class="status-pill"><i class="ti ti-antenna-bars-5"></i>All systems live</div>`;

  const scrim = document.getElementById("scrim");
  if (scrim) {
    scrim.addEventListener("click", () => {
      document.getElementById("sidebar").classList.remove("open");
      scrim.style.display = "none";
    });
  }
}

/* ---------------- Shared risk config ---------------- */
const RISK = {
  VALID: { label: "Valid", cls: "valid" },
  LOW: { label: "Low risk", cls: "low" },
  MEDIUM: { label: "Medium risk", cls: "medium" },
  SUSPICIOUS: { label: "Suspicious", cls: "suspicious" },
  HIGH: { label: "High risk", cls: "high" },
};

function riskBadge(key, size) {
  const r = RISK[key];
  return `<span class="badge ${r.cls} ${size === "sm" ? "sm" : ""}"><span class="badge-dot"></span>${r.label.toUpperCase()}</span>`;
}

/* ---------------- Shared mock data ---------------- */
const DOC_TYPES = [
  { id: "passport", label: "Passport", icon: "ti-book-2" },
  { id: "visa", label: "Visa", icon: "ti-plane" },
  { id: "national_id", label: "National ID", icon: "ti-id-badge-2" },
  { id: "license", label: "Driving licence", icon: "ti-license" },
  { id: "permit", label: "Permit", icon: "ti-file-text" },
];

const CHECKPOINTS = [
  { id: "CP-01", name: "Attari-Wagah Land Border", location: "Punjab, IN", officers: 14, screeningsToday: 312, avgRisk: 18, status: "Operational" },
  { id: "CP-02", name: "IGI Airport T3 — Immigration", location: "New Delhi, IN", officers: 41, screeningsToday: 2870, avgRisk: 12, status: "Operational" },
  { id: "CP-03", name: "Petrapole Land Port", location: "West Bengal, IN", officers: 9, screeningsToday: 204, avgRisk: 27, status: "Elevated" },
  { id: "CP-04", name: "CSMIA Mumbai — Terminal 2", location: "Mumbai, IN", officers: 36, screeningsToday: 2210, avgRisk: 15, status: "Operational" },
  { id: "CP-05", name: "Moreh Land Border", location: "Manipur, IN", officers: 6, screeningsToday: 88, avgRisk: 34, status: "Elevated" },
];

const OFFICERS = [
  { id: "OF-101", name: "R. Sharma", role: "Immigration officer", checkpoint: "CP-02", screenings: 184, flags: 6 },
  { id: "OF-102", name: "A. Nair", role: "Senior officer", checkpoint: "CP-04", screenings: 231, flags: 11 },
  { id: "OF-103", name: "K. Bano", role: "Immigration officer", checkpoint: "CP-01", screenings: 97, flags: 3 },
  { id: "OF-104", name: "S. Reddy", role: "Fraud analyst", checkpoint: "CP-03", screenings: 60, flags: 19 },
  { id: "OF-105", name: "T. Wangchuk", role: "Immigration officer", checkpoint: "CP-05", screenings: 42, flags: 8 },
];

const TEMPLATES = [
  { id: "TPL-IND-P-04", country: "India", type: "Passport", version: "v4 (2019 booklet)", fields: 14, status: "Active" },
  { id: "TPL-IND-V-02", country: "India", type: "Visa sticker", version: "v2", fields: 10, status: "Active" },
  { id: "TPL-USA-P-05", country: "United States", type: "Passport", version: "v5", fields: 15, status: "Active" },
  { id: "TPL-GBR-P-03", country: "United Kingdom", type: "Passport", version: "v3", fields: 14, status: "Active" },
  { id: "TPL-IND-DL-01", country: "India", type: "Driving licence", version: "v1 (state format)", fields: 9, status: "Review" },
  { id: "TPL-UAE-P-02", country: "UAE", type: "Passport", version: "v2", fields: 13, status: "Active" },
];

const VALIDATION_RULES = [
  { id: "VR-001", template: "TPL-IND-P-04", field: "document_number", rule: "Regex format", weight: "High" },
  { id: "VR-002", template: "TPL-IND-P-04", field: "mrz_checksum", rule: "Checksum verification", weight: "Critical" },
  { id: "VR-003", template: "TPL-IND-P-04", field: "expiry_date", rule: "Not expired", weight: "High" },
  { id: "VR-004", template: "TPL-IND-P-04", field: "issue_date", rule: "Chronological consistency", weight: "Medium" },
  { id: "VR-005", template: "TPL-USA-P-05", field: "photo_zone", rule: "Layout position match", weight: "Medium" },
  { id: "VR-006", template: "TPL-GBR-P-03", field: "font_kerning", rule: "Font consistency", weight: "Medium" },
];

const FRAUD_CASES = [
  { id: "FC-2201", title: "Repeated face match across 3 passport numbers", riskKey: "HIGH", status: "Under investigation", linked: 4, checkpoint: "Moreh Land Border", opened: "2026-09-03" },
  { id: "FC-2198", title: "Photo substitution on Uzbek national ID", riskKey: "SUSPICIOUS", status: "Under investigation", linked: 1, checkpoint: "Petrapole Land Port", opened: "2026-09-04" },
  { id: "FC-2184", title: "Template mismatch — counterfeit driving licence ring", riskKey: "SUSPICIOUS", status: "Escalated", linked: 6, checkpoint: "Attari-Wagah", opened: "2026-08-29" },
  { id: "FC-2170", title: "MRZ checksum fraud, forged student visa batch", riskKey: "HIGH", status: "Resolved", linked: 9, checkpoint: "IGI Airport T3", opened: "2026-08-20" },
];

const AUDIT_LOGS = [
  { id: "AL-99231", event: "SCR-4821", hash: "8f2a...c193", prev: "d4e1...771a", time: "2026-09-05 06:12:03", action: "Screening completed — VALID" },
  { id: "AL-99230", event: "SCR-4820", hash: "d4e1...771a", prev: "22b0...5f9c", time: "2026-09-05 05:47:51", action: "Screening completed — VALID" },
  { id: "AL-99229", event: "SCR-4819", hash: "22b0...5f9c", prev: "aa71...20e4", time: "2026-09-04 22:03:18", action: "Screening completed — MEDIUM RISK" },
  { id: "AL-99228", event: "SCR-4818", hash: "aa71...20e4", prev: "0c3f...889b", time: "2026-09-04 19:38:44", action: "Screening completed — VALID" },
  { id: "AL-99227", event: "SCR-4817", hash: "0c3f...889b", prev: "5591...0a2d", time: "2026-09-04 15:21:09", action: "Screening completed — SUSPICIOUS" },
  { id: "AL-99226", event: "SCR-4816", hash: "5591...0a2d", prev: "genesis", time: "2026-09-03 11:05:37", action: "Screening completed — HIGH RISK" },
];

const HISTORY = [
  { id: "SCR-4827", name: "Ananya Verma", documentType: "Passport", checkpoint: "IGI Airport T3", officer: "R. Sharma", date: "2026-09-05 06:12", riskKey: "VALID", score: 6, documentNumber: "P8827341" },
  { id: "SCR-4826", name: "Wei Zhang", documentType: "Visa", checkpoint: "CSMIA Mumbai — T2", officer: "A. Nair", date: "2026-09-05 05:47", riskKey: "VALID", score: 9, documentNumber: "V0093321" },
  { id: "SCR-4825", name: "Michael O'Connell", documentType: "National ID", checkpoint: "Attari-Wagah", officer: "K. Bano", date: "2026-09-04 22:03", riskKey: "MEDIUM", score: 46, documentNumber: "N4471029" },
  { id: "SCR-4824", name: "Priya Menon", documentType: "Passport", checkpoint: "IGI Airport T3", officer: "R. Sharma", date: "2026-09-04 19:38", riskKey: "VALID", score: 11, documentNumber: "P2234871" },
  { id: "SCR-4823", name: "Dilnoza Karimova", documentType: "Passport", checkpoint: "Petrapole Land Port", officer: "S. Reddy", date: "2026-09-04 15:21", riskKey: "SUSPICIOUS", score: 74, documentNumber: "P9081123" },
  { id: "SCR-4822", name: "Farid Al-Mansoori", documentType: "Passport", checkpoint: "Moreh Land Border", officer: "T. Wangchuk", date: "2026-09-03 11:05", riskKey: "HIGH", score: 93, documentNumber: "P5567201" },
  { id: "SCR-4821", name: "Josef Novak", documentType: "Permit", checkpoint: "CSMIA Mumbai — T2", officer: "A. Nair", date: "2026-09-03 09:52", riskKey: "VALID", score: 8, documentNumber: "R7712340" },
  { id: "SCR-4820", name: "Reeta Thapa", documentType: "Driving licence", checkpoint: "Attari-Wagah", officer: "K. Bano", date: "2026-09-02 14:16", riskKey: "MEDIUM", score: 38, documentNumber: "DL0093344" },
];

const IDENTITY_NETWORK = {
  nodes: [
    { id: "face-1", type: "face", label: "Biometric bh_9f21ac", x: 150, y: 160 },
    { id: "p1", type: "person", label: "Farid Al-Mansoori", x: 40, y: 60 },
    { id: "p2", type: "person", label: "Faisal Al-Mansour", x: 260, y: 50 },
    { id: "p3", type: "person", label: "F. Mansoori", x: 60, y: 260 },
    { id: "d1", type: "document", label: "Passport P5567201", x: 40, y: 130 },
    { id: "d2", type: "document", label: "Passport P4498213", x: 270, y: 140 },
    { id: "d3", type: "document", label: "National ID N7723410", x: 90, y: 250 },
    { id: "d4", type: "document", label: "Visa V0021940", x: 230, y: 250 },
    { id: "face-2", type: "face", label: "Biometric bh_22d0e1", x: 500, y: 90 },
    { id: "p4", type: "person", label: "Ananya Verma", x: 500, y: 200 },
    { id: "d5", type: "document", label: "Passport P8827341", x: 600, y: 140 },
  ],
  edges: [["face-1","p1"],["face-1","p2"],["face-1","p3"],["p1","d1"],["p2","d2"],["p3","d3"],["p1","d4"],["face-2","p4"],["p4","d5"]],
};

const SCENARIOS = {
  clean: { label: "Clean document", riskKey: "VALID", score: 6, person: { name: "Ananya Verma", nat: "Indian", dob: "1994-04-12" },
    mrzMatch: true, tamperFindings: [], faceScore: 98, liveness: 99, fraudMatches: 0,
    breakdown: [["Document authenticity",4,30],["Face verification",1,25],["Tampering signals",0,25],["Fraud database history",1,20]],
    notes: "All checks passed. Template, MRZ checksum, and biometric match are consistent with a genuine document." },
  discrepancy: { label: "Minor discrepancy", riskKey: "MEDIUM", score: 46, person: { name: "Michael O'Connell", nat: "Irish", dob: "1988-11-02" },
    mrzMatch: false, tamperFindings: [{ area: "Date of birth field", confidence: 61, note: "OCR / MRZ mismatch in DOB digit" }],
    faceScore: 88, liveness: 94, fraudMatches: 0,
    breakdown: [["Document authenticity",14,30],["Face verification",6,25],["Tampering signals",16,25],["Fraud database history",2,20]],
    notes: "MRZ checksum does not fully agree with the printed date of birth. Could be a scan artefact — recommend manual re-check." },
  tampered: { label: "Tampered document", riskKey: "SUSPICIOUS", score: 74, person: { name: "Dilnoza Karimova", nat: "Uzbek", dob: "1996-07-19" },
    mrzMatch: false, tamperFindings: [
      { area: "Photo substitution zone", confidence: 87, note: "Ghost image / pixel discontinuity around photo edge" },
      { area: "Document number", confidence: 79, note: "Font kerning inconsistent with issuing template" },
      { area: "Laminate layer", confidence: 68, note: "Reflectance pattern suggests re-lamination" }],
    faceScore: 54, liveness: 91, fraudMatches: 1,
    breakdown: [["Document authenticity",26,30],["Face verification",17,25],["Tampering signals",23,25],["Fraud database history",8,20]],
    notes: "Multiple physical tampering indicators detected around the photo and document number zones. Face match confidence is degraded relative to the reference photo." },
  fraud_ring: { label: "Fraud ring match", riskKey: "HIGH", score: 93, person: { name: "Farid Al-Mansoori", nat: "Unknown / disputed", dob: "1985-01-30" },
    mrzMatch: false, tamperFindings: [
      { area: "Photo substitution zone", confidence: 95, note: "High-confidence photo swap detected" },
      { area: "MRZ line 2", confidence: 90, note: "Checksum fails against printed data" }],
    faceScore: 22, liveness: 40, fraudMatches: 4,
    breakdown: [["Document authenticity",29,30],["Face verification",24,25],["Tampering signals",24,25],["Fraud database history",16,20]],
    notes: "Biometric hash matches 3 other identities previously flagged across 2 checkpoints. This face has been used with 4 different document numbers in the last 90 days." },
};

const STEPS = [
  { key: "detect", label: "Document type detection", icon: "ti-search" },
  { key: "ocr", label: "OCR extraction", icon: "ti-file-text" },
  { key: "template", label: "Template validation", icon: "ti-stack-2" },
  { key: "rules", label: "Validation rules", icon: "ti-adjustments" },
  { key: "mrz", label: "MRZ verification", icon: "ti-hash" },
  { key: "tamper", label: "Tampering detection", icon: "ti-shield-exclamation" },
  { key: "face", label: "Face verification", icon: "ti-face-id" },
  { key: "fraud", label: "Fraud intelligence check", icon: "ti-affiliate" },
  { key: "risk", label: "Risk score generation", icon: "ti-shield-check" },
];