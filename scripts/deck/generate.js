// SATA progress-report deck (v0) — built with pptxgenjs per the pptx skill.
// Palette: bio-inspired + control. Motif: number chip + left accent bar.

const pptxgen = require("pptxgenjs");
const path = require("path");

// ---- palette --------------------------------------------------------------
const DARK = "12302D";   // deep pine-teal  (header bands, conclusion)
const TEAL = "0E7C7B";   // primary
const TEALL = "DCEDEA";  // light teal fill
const AMBER = "E0922B";  // sharp accent (torque / energy)
const AMBERD = "9A5F12"; // amber text-safe on light
const CLAY = "B0573F";   // position-control (rigid / negative)
const CLAYL = "F1E2DD";  // light clay fill
const CREAM = "F4F1EA";  // content background
const INK = "243230";    // body text
const MUTE = "7C8A87";   // captions
const WHITE = "FFFFFF";
const LINEC = "E5DFD4";  // card border

const HEAD = "Georgia";
const BODY = "Calibri";

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.33, height: 7.5 });
pres.layout = "W";
pres.author = "SATA Progress Report";
pres.title = "Bio-inspired Adaptive Locomotion via Torque-based Learning";

const sh = () => ({ type: "outer", color: "000000", blur: 5, offset: 2, angle: 135, opacity: 0.16 });

function chip(s, x, y, d, label, fill, tc = WHITE, fs = 20) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(String(label), { x, y, w: d, h: d, align: "center", valign: "middle",
    fontSize: fs, bold: true, color: tc, fontFace: HEAD, margin: 0 });
}

function card(s, x, y, w, h, accent) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE },
    line: { color: LINEC, width: 1 }, shadow: sh() });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.1, h, fill: { color: accent } });
}

function header(s, n, title) {
  s.background = { color: CREAM };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 1.05, fill: { color: DARK } });
  chip(s, 0.5, 0.27, 0.52, n, AMBER, DARK, 22);
  s.addText(title, { x: 1.22, y: 0.16, w: 11.6, h: 0.72, fontSize: 25, bold: true,
    color: WHITE, fontFace: HEAD, valign: "middle", margin: 0 });
}

function caption(s, y, txt) {
  s.addText(txt, { x: 0.55, y, w: 12.2, h: 0.4, fontSize: 13, italic: true,
    color: MUTE, fontFace: BODY, margin: 0 });
}

function refs(s, txt) {
  s.addText([{ text: "Refs  ", options: { bold: true, color: TEAL } },
             { text: txt, options: { color: MUTE } }],
    { x: 0.55, y: 7.06, w: 12.2, h: 0.32, fontSize: 10, italic: true, fontFace: BODY, margin: 0 });
}

function down(s, x, y, color) {
  s.addText("▼", { x, y, w: 0.4, h: 0.28, align: "center", valign: "middle",
    fontSize: 12, color, margin: 0 });
}

// ============================================================ SLIDE 1
let s = pres.addSlide();
header(s, 1, "Locomotion Is a Control Problem");

// LEFT — problem card
card(s, 0.55, 1.45, 5.45, 2.05, CLAY);
s.addText("THE PROBLEM", { x: 0.8, y: 1.6, w: 5.0, h: 0.35, fontSize: 13, bold: true,
  color: CLAY, fontFace: BODY, charSpacing: 2, margin: 0 });
s.addText("Position-based locomotion is accurate under known conditions but rigid — "
  + "it struggles with compliance, disturbance handling, unknown terrain, and the "
  + "sim-to-real gap.",
  { x: 0.8, y: 1.98, w: 5.0, h: 1.4, fontSize: 15, color: INK, fontFace: BODY, margin: 0, valign: "top" });

s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 3.7, w: 5.45, h: 0.78, fill: { color: TEALL } });
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 3.7, w: 0.1, h: 0.78, fill: { color: TEAL } });
s.addText("→  Torque-based control enables compliant, adaptive interaction.",
  { x: 0.8, y: 3.7, w: 5.1, h: 0.78, fontSize: 14, bold: true, color: TEAL, fontFace: BODY, valign: "middle", margin: 0 });

// RIGHT — comparison columns
const colY = 1.45;
const posX = 6.35, torX = 9.85, colW = 2.95;
s.addShape(pres.shapes.RECTANGLE, { x: posX, y: colY, w: colW, h: 0.5, fill: { color: CLAY } });
s.addText("POSITION CONTROL", { x: posX, y: colY, w: colW, h: 0.5, align: "center", valign: "middle",
  fontSize: 12.5, bold: true, color: WHITE, fontFace: BODY, charSpacing: 1, margin: 0 });
s.addShape(pres.shapes.RECTANGLE, { x: torX, y: colY, w: colW, h: 0.5, fill: { color: TEAL } });
s.addText("TORQUE CONTROL", { x: torX, y: colY, w: colW, h: 0.5, align: "center", valign: "middle",
  fontSize: 12.5, bold: true, color: WHITE, fontFace: BODY, charSpacing: 1, margin: 0 });

const posSteps = ["Command position", "Rigid behavior", "Poor adaptation"];
const torSteps = ["Command force", "Compliant interaction", "Adaptive response"];
let yy = 2.15;
for (let i = 0; i < 3; i++) {
  s.addShape(pres.shapes.RECTANGLE, { x: posX, y: yy, w: colW, h: 0.6, fill: { color: CLAYL }, line: { color: CLAY, width: 1 } });
  s.addText(posSteps[i], { x: posX, y: yy, w: colW, h: 0.6, align: "center", valign: "middle", fontSize: 13, color: INK, fontFace: BODY, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: torX, y: yy, w: colW, h: 0.6, fill: { color: TEALL }, line: { color: TEAL, width: 1 } });
  s.addText(torSteps[i], { x: torX, y: yy, w: colW, h: 0.6, align: "center", valign: "middle", fontSize: 13, color: INK, fontFace: BODY, margin: 0 });
  if (i < 2) {
    down(s, posX + colW / 2 - 0.2, yy + 0.61, CLAY);
    down(s, torX + colW / 2 - 0.2, yy + 0.61, TEAL);
  }
  yy += 0.95;
}

// BOTTOM — research question
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 5.5, w: 12.23, h: 1.0, fill: { color: DARK } });
s.addText("RESEARCH QUESTION", { x: 0.85, y: 5.62, w: 11.6, h: 0.3, fontSize: 11, bold: true, color: AMBER, fontFace: BODY, charSpacing: 2, margin: 0 });
s.addText("How can robots achieve safer and more adaptive locomotion in unknown environments?",
  { x: 0.85, y: 5.9, w: 11.6, h: 0.5, fontSize: 18, bold: true, color: WHITE, fontFace: HEAD, margin: 0 });
refs(s, "SATA · Chen et al. (torque control) · Lee et al. (terrain) · Miki et al. (perceptive loco.)");
s.addNotes(
  "[~40 sec]\nTALKING POINTS:\n"
  + "- Frame this as a CONTROL problem, not an RL talk.\n"
  + "- Position control commands a joint angle; a low-level PD loop turns the error into torque. Stiff and accurate when the world is known.\n"
  + "- It fails on exactly what this course cares about: compliance, disturbance rejection, unknown terrain, sim-to-real gap.\n"
  + "- Torque control commands force directly, so the robot can yield and adapt instead of resisting.\n"
  + "TRANSITION: 'The question is how to get adaptive locomotion in unknown environments — SATA is one bio-inspired answer.'");

// ============================================================ SLIDE 2
s = pres.addSlide();
header(s, 2, "SATA: Bio-inspired Adaptation in Torque Control");
caption(s, 1.18, "An RL torque policy wrapped by a biomechanical layer and a growth curriculum.");

// LEFT — vertical flow
const fx = 2.75, fw = 3.55;
const nodes = ["Observation", "Torque Policy (RL)", "Biomechanical Layer", "Torque Output", "Robot"];
const hot = [false, true, true, false, false];
let ny = 1.75;
const centers = [];
for (let i = 0; i < nodes.length; i++) {
  const fill = hot[i] ? TEALL : WHITE;
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: ny, w: fw, h: 0.6, fill: { color: fill }, line: { color: hot[i] ? TEAL : LINEC, width: hot[i] ? 1.5 : 1 }, shadow: sh() });
  s.addText(nodes[i], { x: fx, y: ny, w: fw, h: 0.6, align: "center", valign: "middle", fontSize: 14, bold: hot[i], color: INK, fontFace: BODY, margin: 0 });
  centers.push(ny + 0.3);
  if (i < nodes.length - 1) down(s, fx + fw / 2 - 0.2, ny + 0.61, MUTE);
  ny += 0.8;
}
// feedback arrow (up) on the left
s.addShape(pres.shapes.LINE, { x: 2.4, y: centers[1], w: 0, h: centers[4] - centers[1],
  line: { color: AMBER, width: 2.5, beginArrowType: "triangle" } });
s.addText("Fatigue\nfeedback", { x: 0.55, y: (centers[1] + centers[4]) / 2 - 0.45, w: 1.7, h: 0.9,
  align: "right", valign: "middle", fontSize: 12, bold: true, color: AMBERD, fontFace: BODY, margin: 0 });

// RIGHT — highlight cards
const hx = 7.05, hw = 5.75;
card(s, hx, 1.7, hw, 1.5, TEAL);
chip(s, hx + 0.28, 1.95, 0.55, "M", TEAL, WHITE, 18);
s.addText("Biomechanical Model", { x: hx + 1.0, y: 1.92, w: hw - 1.2, h: 0.4, fontSize: 18, bold: true, color: TEAL, fontFace: HEAD, margin: 0 });
s.addText("activation  ·  muscle  ·  fatigue", { x: hx + 1.0, y: 2.4, w: hw - 1.2, h: 0.6, fontSize: 14, color: INK, fontFace: BODY, margin: 0 });

card(s, hx, 3.45, hw, 1.5, AMBER);
chip(s, hx + 0.28, 3.7, 0.55, "G", AMBER, WHITE, 18);
s.addText("Growth Mechanism", { x: hx + 1.0, y: 3.67, w: hw - 1.2, h: 0.4, fontSize: 18, bold: true, color: AMBERD, fontFace: HEAD, margin: 0 });
s.addText("torque limit  ·  reward  ·  frequency", { x: hx + 1.0, y: 4.15, w: hw - 1.2, h: 0.6, fontSize: 14, color: INK, fontFace: BODY, margin: 0 });

s.addText("Goal: smooth, safe torques that progressively unlock capability — deployed zero-shot (sim-to-real).",
  { x: hx, y: 5.2, w: hw, h: 0.9, fontSize: 13, italic: true, color: MUTE, fontFace: BODY, margin: 0, valign: "top" });
refs(s, "SATA · Hill (muscle dynamics) · Liu et al. (fatigue) · Bellegarda & Ijspeert (CPG-RL)");
s.addNotes(
  "[~60 sec]\nTALKING POINTS:\n"
  + "- One message: SATA injects bio-inspired adaptation INTO a torque policy. Don't read equations.\n"
  + "- Flow: observation -> RL torque policy -> biomechanical layer -> torque -> robot; fatigue state feeds back.\n"
  + "- Biomechanical model: activation smooths the command, a muscle model prevents abrupt torque, fatigue discourages overusing joints.\n"
  + "- Growth mechanism: torque limit, reward, and control frequency unlock progressively — like an animal maturing.\n"
  + "- Net effect: safe torques plus generalization, deployed zero-shot.\n"
  + "TRANSITION: 'That's what SATA is — here is how WE plan to study it.'");

// ============================================================ SLIDE 3
s = pres.addSlide();
header(s, 3, "Plan: Understand → Reproduce → Analyze");
caption(s, 1.18, "A feasible 3-phase study — we interpret SATA, we do not redesign the RL.");

const phases = [
  { n: "1", t: "Reproduce", c: TEAL, b: "Set up Isaac Gym on a CUDA server (container / venv) and run the official SATA training.", out: "simulation running", opt: false },
  { n: "2", t: "Ablation", c: TEAL, b: "Disable fatigue · change torque limit · modify growth schedule · vary terrain.", out: "behavioral data", opt: false },
  { n: "3", t: "Adaptive Interpretation", c: TEAL, b: "Growth → gain scheduling\nFatigue → feedback\nTorque limit → adaptive constraint", out: "analysis & discussion", opt: false },
  { n: "4", t: "Residual Compensation", c: AMBER, b: "τ_total = τ_SATA + τ_comp\nPursue only if feasible.", out: "robustness in boundary cases", opt: true },
];
const pTop = 1.7, pH = 3.15, pW = 2.86, pGap = 0.3;
let px = 0.55;
for (let i = 0; i < phases.length; i++) {
  const p = phases[i];
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: pTop, w: pW, h: pH, fill: { color: WHITE }, line: { color: LINEC, width: 1 }, shadow: sh() });
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: pTop, w: pW, h: 0.85, fill: { color: p.c } });
  chip(s, px + 0.18, pTop + 0.17, 0.5, p.n, WHITE, p.c, 18);
  s.addText(p.opt ? "PHASE 4 · OPTIONAL" : "PHASE " + p.n, { x: px + 0.8, y: pTop + 0.13, w: pW - 0.9, h: 0.28, fontSize: 9.5, bold: true, color: WHITE, fontFace: BODY, charSpacing: 1, margin: 0 });
  s.addText(p.t, { x: px + 0.8, y: pTop + 0.4, w: pW - 0.9, h: 0.4, fontSize: 14, bold: true, color: WHITE, fontFace: BODY, margin: 0, valign: "top" });
  s.addText(p.b, { x: px + 0.22, y: pTop + 1.02, w: pW - 0.44, h: 1.35, fontSize: 12.5, color: INK, fontFace: BODY, margin: 0, valign: "top" });
  // output footer inside card
  s.addShape(pres.shapes.LINE, { x: px + 0.22, y: pTop + pH - 0.72, w: pW - 0.44, h: 0, line: { color: LINEC, width: 1 } });
  s.addText([{ text: "OUTPUT  ", options: { bold: true, color: p.opt ? AMBERD : TEAL } },
             { text: p.out, options: { color: INK } }],
    { x: px + 0.22, y: pTop + pH - 0.62, w: pW - 0.44, h: 0.5, fontSize: 11.5, fontFace: BODY, margin: 0, valign: "top" });
  if (i < phases.length - 1)
    s.addText("→", { x: px + pW - 0.02, y: pTop + 0.2, w: pGap + 0.04, h: 0.5, align: "center", valign: "middle", fontSize: 18, bold: true, color: MUTE, margin: 0 });
  px += pW + pGap;
}

s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 5.55, w: 12.23, h: 0.95, fill: { color: DARK } });
s.addText("APPROACH", { x: 0.85, y: 5.66, w: 11.6, h: 0.3, fontSize: 11, bold: true, color: AMBER, fontFace: BODY, charSpacing: 2, margin: 0 });
s.addText("We understand and analyze adaptive behavior — we do not redesign the RL algorithm.",
  { x: 0.85, y: 5.94, w: 11.6, h: 0.45, fontSize: 16, bold: true, color: WHITE, fontFace: HEAD, margin: 0 });
refs(s, "SATA · RL2AC · DecAP");
s.addNotes(
  "[~50 sec]\nTALKING POINTS:\n"
  + "- Feasibility is the point of this slide: executable, not just a paper review.\n"
  + "- Phase 1: stand up the official SATA pipeline in Isaac Gym — proves we can run it.\n"
  + "- Phase 2: controlled ablations — toggle fatigue, retune torque limit / growth, change terrain; observe behavior shifts.\n"
  + "- Phase 3: map each mechanism to an adaptive-control reading (analogy, not equivalence).\n"
  + "- Phase 4 residual term only if time allows — not promised.\n"
  + "TRANSITION: 'So what do we expect to deliver?'");

// ============================================================ SLIDE 4
s = pres.addSlide();
header(s, 4, "Expected Contribution");

s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.4, w: 12.23, h: 0.8, fill: { color: TEALL } });
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.4, w: 0.1, h: 0.8, fill: { color: TEAL } });
s.addText("We study adaptive behavior — not propose a new RL algorithm.",
  { x: 0.85, y: 1.4, w: 11.8, h: 0.8, fontSize: 18, bold: true, color: TEAL, fontFace: HEAD, valign: "middle", margin: 0 });

// LEFT — expected outputs
s.addText("EXPECTED OUTPUTS", { x: 0.55, y: 2.55, w: 5.9, h: 0.4, fontSize: 15, bold: true, color: DARK, fontFace: BODY, charSpacing: 1.5, margin: 0 });
const outs = ["Simulation reproduction", "Behavior analysis (ablation)", "Adaptive-control interpretation"];
let oy = 3.1;
for (let i = 0; i < outs.length; i++) {
  chip(s, 0.6, oy, 0.45, i + 1, TEAL, WHITE, 15);
  s.addText(outs[i], { x: 1.2, y: oy, w: 5.2, h: 0.45, fontSize: 15, color: INK, fontFace: BODY, valign: "middle", margin: 0 });
  oy += 0.85;
}

// RIGHT — research questions (2x2)
s.addText("RESEARCH QUESTIONS", { x: 6.9, y: 2.55, w: 5.9, h: 0.4, fontSize: 15, bold: true, color: DARK, fontFace: BODY, charSpacing: 1.5, margin: 0 });
const rqs = [["RQ1", "Why torque control?"], ["RQ2", "What creates adaptation?"],
             ["RQ3", "How does it relate to adaptive control?"], ["RQ4", "Can lightweight compensation help?"]];
const rqW = 2.86, rqH = 1.3, rqGap = 0.2;
for (let i = 0; i < 4; i++) {
  const cx = 6.9 + (i % 2) * (rqW + rqGap);
  const cy = 3.1 + Math.floor(i / 2) * (rqH + rqGap);
  card(s, cx, cy, rqW, rqH, i === 3 ? AMBER : TEAL);
  s.addText(rqs[i][0], { x: cx + 0.22, y: cy + 0.15, w: rqW - 0.4, h: 0.35, fontSize: 13, bold: true, color: i === 3 ? AMBERD : TEAL, fontFace: HEAD, margin: 0 });
  s.addText(rqs[i][1], { x: cx + 0.22, y: cy + 0.5, w: rqW - 0.4, h: rqH - 0.65, fontSize: 13, color: INK, fontFace: BODY, margin: 0, valign: "top" });
}

// BOTTOM — conclusion
s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 6.15, w: 12.23, h: 1.0, fill: { color: DARK } });
s.addText("IN ONE LINE", { x: 0.85, y: 6.26, w: 11.6, h: 0.28, fontSize: 11, bold: true, color: AMBER, fontFace: BODY, charSpacing: 2, margin: 0 });
s.addText("This project investigates how bio-inspired torque control creates adaptive locomotion behaviors, and interprets these mechanisms through an adaptive control perspective.",
  { x: 0.85, y: 6.52, w: 11.6, h: 0.55, fontSize: 14, bold: true, color: WHITE, fontFace: BODY, margin: 0, valign: "top" });
s.addNotes(
  "[~30 sec]\nTALKING POINTS:\n"
  + "- Close the loop, no overclaiming: the contribution is understanding, not a new algorithm.\n"
  + "- Three concrete outputs: reproduction, behavior analysis, control-theoretic interpretation.\n"
  + "- The four research questions map onto the four slides.\n"
  + "CLOSING LINE (read the bottom band): bio-inspired torque control -> adaptive behavior -> interpreted through adaptive control.\n"
  + "TRANSITION: 'Happy to take questions.'");

const out = path.resolve(__dirname, "../../docs/progress-report-v0.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("saved", out));
