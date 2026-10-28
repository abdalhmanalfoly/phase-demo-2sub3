// Camera access: explicit permission request, preview, and guaranteed track release.
export function classifyCameraError(err) {
  const name = err?.name ?? "Error";
  if (name === "InsecureContextError") {
    return { permission: "unknown", camera: "unavailable", code: "insecure_context",
      message: "Camera access needs HTTPS or localhost. Open this page over https:// (see docs/setup.md)." };
  }
  if (name === "NotAllowedError" || name === "SecurityError") {
    return { permission: "denied", camera: "idle", code: "permission_denied",
      message: "Camera permission was denied. Allow camera access in the browser settings and press Start again." };
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return { permission: "granted", camera: "unavailable", code: "no_camera", message: "No camera was found on this device." };
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return { permission: "granted", camera: "unavailable", code: "camera_busy", message: "The camera is in use by another app or tab." };
  }
  if (name === "OverconstrainedError") {
    return { permission: "granted", camera: "unavailable", code: "constraints", message: "The camera cannot satisfy the requested settings." };
  }
  return { permission: "unknown", camera: "unavailable", code: "camera_error", message: `Camera error: ${err?.message ?? name}` };
}

export class CameraController {
  constructor() { this.stream = null; this.video = null; }

  async start(video, { facing, width, height, frameRate }) {
    if (!globalThis.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      const e = new Error("insecure context or mediaDevices unavailable"); e.name = "InsecureContextError"; throw e;
    }
    this.stop();                                           // never hold two streams
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: facing }, width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: frameRate } },
    });
    this.stream = stream; this.video = video;
    video.muted = true; video.playsInline = true; video.srcObject = stream;
    await video.play();
    if (!video.videoWidth) await new Promise((res) => video.addEventListener("loadedmetadata", res, { once: true }));
    const s = stream.getVideoTracks()[0]?.getSettings?.() ?? {};
    // Tracks can end on their own (unplugged camera, OS revoke); surface that to the caller.
    stream.getVideoTracks().forEach((t) => { t.onended = () => this.onEnded?.(); });
    return { width: video.videoWidth, height: video.videoHeight, facing: s.facingMode === "environment" ? "environment" : s.facingMode === "user" ? "user" : "unknown" };
  }

  stop() {
    if (this.stream) { this.stream.getTracks().forEach((t) => { t.onended = null; t.stop(); }); this.stream = null; }
    if (this.video) { this.video.pause?.(); this.video.srcObject = null; }
  }
}
