// MediaPipe Tasks Vision hand landmarker running locally in the browser (no video leaves the device).
import { OUT_OF_FRAME_MARGIN } from "./protocol.js";

// MediaPipe hand landmark indices.
const INDEX = { wrist: 0, thumb_tip: 4, index_mcp: 5, index_tip: 8, pinky_mcp: 17 };
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class LandmarkDetector {
  constructor(cfg) { this.cfg = cfg; this.landmarker = null; this.lastTs = 0; this.delegate = null; }

  async load() {
    const { FilesetResolver, HandLandmarker } = await import(this.cfg.model.bundle_path);
    const fileset = await FilesetResolver.forVisionTasks(this.cfg.model.wasm_path);
    const v = this.cfg.vision;
    const make = (delegate) => HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: this.cfg.model.model_path, delegate },
      runningMode: "VIDEO", numHands: 1,
      minHandDetectionConfidence: v.min_detection_confidence,
      minHandPresenceConfidence: v.min_presence_confidence,
      minTrackingConfidence: v.min_tracking_confidence,
    });
    try { this.landmarker = await make("GPU"); this.delegate = "GPU"; }
    catch { this.landmarker = await make("CPU"); this.delegate = "CPU"; }   // GPU is unavailable on many phones/browsers
  }

  /** Synchronous inference on the current video frame. Returns the observation payload fields. */
  detect(video, wantedIds) {
    const w = video.videoWidth, h = video.videoHeight;              // pixel space of the frame MediaPipe sees
    const ts = Math.max(this.lastTs + 1, Math.round(performance.now()));   // MediaPipe needs strictly increasing ms
    this.lastTs = ts;
    const t0 = performance.now();
    const res = this.landmarker.detectForVideo(video, ts);
    const inferenceMs = performance.now() - t0;
    const lms = res.landmarks?.[0];
    if (!lms) return { status: "no_hand", hand: "unknown", landmarks: [], handednessScore: null, inferenceMs, imageWidth: w, imageHeight: h };

    const cat = res.handedness?.[0]?.[0];
    let hand = "unknown";
    if (cat?.categoryName === "Left" || cat?.categoryName === "Right") {
      // MediaPipe labels assume a mirrored (selfie) image; we send the un-mirrored frame, so swap when configured.
      const flip = this.cfg.camera.flip_handedness;
      hand = (cat.categoryName === "Right") !== flip ? "right" : "left";
    }
    let outOfFrame = false;
    const mx = OUT_OF_FRAME_MARGIN;
    const landmarks = wantedIds.filter((id) => id in INDEX).map((id) => {
      const p = lms[INDEX[id]];
      let nx = p.x, ny = p.y;
      if (nx < -mx || nx > 1 + mx || ny < -mx || ny > 1 + mx) outOfFrame = true;
      nx = clamp(nx, -mx, 1 + mx); ny = clamp(ny, -mx, 1 + mx);
      return {
        id, x_px: nx * w, y_px: ny * h,
        // The hand model reports no per-landmark confidence. Only forward one if the runtime provides it.
        confidence: typeof p.visibility === "number" && Number.isFinite(p.visibility) ? clamp(p.visibility, 0, 1) : null,
        x_norm: nx, y_norm: ny,
        z_model: Number.isFinite(p.z) ? p.z : null,                 // model-relative, not metric
      };
    });
    return { status: "tracked", hand, landmarks, handednessScore: typeof cat?.score === "number" ? clamp(cat.score, 0, 1) : null,
             inferenceMs, imageWidth: w, imageHeight: h, outOfFrame };
  }

  close() { try { this.landmarker?.close(); } catch { /* ignore */ } this.landmarker = null; }
}
