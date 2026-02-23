#!/usr/bin/env node
// adapters/freezeOmni_adapter_socketio.js
// Bridges orchestrator (WebRTC, 48 kHz PCM16, 10 ms cadence) <-> Freeze-Omni server (Socket.IO, 16 kHz uplink JSON Int16, 24 kHz downlink Int16 bytes).

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const wrtc = require('wrtc');
const WebSocket = require('ws');
const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;
const { spawn } = require('child_process');
const ffmpegPath = require('ffmpeg-static');
const { io } = require('socket.io-client');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

// Audio configuration from environment variables with defaults
const DEFAULT_WIRE_SR = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;
const DEFAULT_FO_UP_SR = parseInt(process.env.FO_UPLINK_SAMPLE_RATE, 10) || 16000;
const DEFAULT_FO_DOWN_SR = parseInt(process.env.FO_DOWNLINK_SAMPLE_RATE, 10) || 24000;
const DEFAULT_FO_URL = process.env.FREEZE_OMNI_URL || 'https://localhost:8081';

// ─── CLI ─────────────────────────────────────────────────────────────────────
const argv = yargs(hideBin(process.argv))
    .option('role', { type: 'string', choices: ['A', 'B'], demandOption: true })
    .option('signalUrl', { type: 'string', demandOption: true })
    .option('foUrl', { type: 'string', default: DEFAULT_FO_URL, describe: 'Freeze-Omni server URL' })
    .option('insecureTLS', { type: 'boolean', default: false, describe: 'If true, do not verify TLS certificates (development only).' })
    .option('caCertPath', { type: 'string', default: '', describe: 'Path to a PEM CA certificate to trust for TLS.' })
    .option('prefillPrompt', { type: 'string', default: '', describe: 'Optional system prompt to set on server via prompt_text.' })
    .option('uplinkMs', { type: 'number', default: 20, describe: 'Frame duration to send to FO (ms) at 16 kHz.' })
    .option('downlinkMs', { type: 'number', default: 20, describe: 'Nominal pacing from FO to orchestrator (ms) after resample to 48 kHz.' })
    .option('logFrames', { type: 'boolean', default: false })
    .option('saveDebug', { type: 'boolean', default: false, describe: 'Save server audio to wav for debugging.' })
    .help().argv;

const ROLE = argv.role;
const SIGNAL_URL = argv.signalUrl;
const FO_URL = argv.foUrl;
const INSECURE_TLS = argv.insecureTLS;
const CA_CERT_PATH = argv.caCertPath;
const PREFILL_PROMPT = argv.prefillPrompt;
const UPLINK_MS = Math.max(5, Math.min(100, argv.uplinkMs | 0));
const DOWNLINK_MS = Math.max(5, Math.min(100, argv.downlinkMs | 0));
const LOG_FRAMES = !!argv.logFrames;
const SAVE_DEBUG = !!argv.saveDebug;

// Wire sample rate and sizes (from env or defaults)
const WIRE_SR = DEFAULT_WIRE_SR;
const FRAME_10MS_BYTES = Math.floor(WIRE_SR * 0.01) * 2; // 10 ms @ WIRE_SR PCM16 mono

// Freeze-Omni rates (from env or defaults)
const FO_UP_SR = DEFAULT_FO_UP_SR;
const FO_DOWN_SR = DEFAULT_FO_DOWN_SR;

// ─── State ───────────────────────────────────────────────────────────────────
let ws;                   // WS signaling to orchestrator
let pc;                   // RTCPeerConnection toward orchestrator
let orchSource;           // RTCAudioSource to send audio to orchestrator
let orchSink;             // RTCAudioSink to receive audio from orchestrator
let wirePacer = null;     // setInterval for pacing FO->Orch

let sio;                  // Socket.IO client toward Freeze-Omni

let downFF = null;        // 48k -> 16k (to FO)
let upFF = null;          // 24k -> 48k (to orchestrator)
let downB = Buffer.alloc(0); // bytes at 16k, s16le, ready to encode into FO event
let upB = Buffer.alloc(0);   // bytes at 48k from FO after upsample
let debugFF = null;       // optional writer

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

(async () => {
    console.log(`[Adapter-${ROLE}] Starting Freeze-Omni adapter`);
    initFfmpegPipes();
    await startOrchBridge();
    await startFreezeOmniClient();
})();

// ─── Orchestrator leg ────────────────────────────────────────────────────────
async function startOrchBridge() {
    ws = new WebSocket(SIGNAL_URL);
    await new Promise((res) => ws.once('open', res));

    pc = new wrtc.RTCPeerConnection();
    orchSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
    pc.addTrack(orchSource.createTrack());

    pc.ontrack = ({ track }) => {
        if (track.kind !== 'audio') return;
        if (orchSink) orchSink.stop();
        orchSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
        let first = true;
        orchSink.ondata = ({ samples }) => {
            const view = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
            if (first) { console.log(`[Adapter-${ROLE}] orchSink first bytesIn=${view.length}`); first = false; }
            // Push orchestrator audio (48k) into downFF (48k -> 16k)
            if (downFF && !downFF.killed) {
                try { downFF.stdin.write(view); } catch { }
            }
        };
    };

    pc.onicecandidate = ({ candidate }) => {
        if (!candidate) return;
        try {
            ws.send(JSON.stringify({
                role: ROLE, type: 'candidate', candidate: {
                    candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex
                }
            }));
        } catch { }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    ws.send(JSON.stringify({ role: ROLE, type: 'offer', sdp: offer.sdp }));

    ws.on('message', async (raw) => {
        const msg = JSON.parse(raw);
        if (msg.type === 'answer' && msg.role === ROLE) {
            await pc.setRemoteDescription(new wrtc.RTCSessionDescription({ type: 'answer', sdp: msg.sdp }));
            console.log(`[Adapter-${ROLE}] Orchestrator answer applied.`);
        }
        if (msg.type === 'candidate' && msg.role === ROLE) {
            try { await pc.addIceCandidate(msg.candidate); } catch { }
        }
    });

    startWirePacer();
}

function startWirePacer() {
    if (wirePacer) return;
    const FRAME_10MS = FRAME_10MS_BYTES; // 960 bytes
    wirePacer = setInterval(() => {
        if (!orchSource) return;
        for (let i = 0; i < 10; i += 1) {
            let out;
            if (upB.length >= FRAME_10MS) {
                out = upB.slice(0, FRAME_10MS);
                upB = upB.slice(FRAME_10MS);
            } else if (upB.length > 0) {
                out = Buffer.alloc(FRAME_10MS);
                upB.copy(out, 0, 0, upB.length);
                upB = Buffer.alloc(0);
            } else {
                out = Buffer.alloc(FRAME_10MS);
            }
            const ab = new ArrayBuffer(FRAME_10MS);
            new Uint8Array(ab).set(new Uint8Array(out.buffer, out.byteOffset, FRAME_10MS));
            const samples = new Int16Array(ab);
            orchSource.onData({ samples, sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
        }
    }, 100);
}

// ─── Freeze-Omni leg (Socket.IO) ─────────────────────────────────────────────
async function startFreezeOmniClient() {
    const sioOpts = {
        transports: ['websocket'],
        reconnection: true,
        rejectUnauthorized: !INSECURE_TLS,
    };
    if (CA_CERT_PATH) {
        try { sioOpts.ca = fs.readFileSync(CA_CERT_PATH); } catch (e) {
            console.warn(`[Adapter-${ROLE}] Failed to read CA cert at ${CA_CERT_PATH}:`, e?.message || e);
        }
    }
    sio = io(FO_URL, sioOpts);

    sio.on('connect', () => {
        console.log(`[Adapter-${ROLE}] Connected to Freeze-Omni at ${FO_URL}`);
        if (PREFILL_PROMPT) {
            try { sio.emit('prompt_text', PREFILL_PROMPT); } catch { }
        }
        try { sio.emit('recording-started'); } catch { }
    });

    sio.on('disconnect', (reason) => {
        console.log(`[Adapter-${ROLE}] Freeze-Omni disconnected: ${reason}`);
    });

    // Receive downlink PCM (Int16 bytes) at 24 kHz, feed into upsampler
    sio.on('audio', (data) => {
        // data is expected to be a Buffer (binary). Handle ArrayBuffer/Uint8Array as well.
        let buf;
        if (Buffer.isBuffer(data)) buf = data;
        else if (data?.buffer) buf = Buffer.from(data.buffer);
        else if (Array.isArray(data)) buf = Buffer.from(Uint8Array.from(data));
        else return;

        if (SAVE_DEBUG && debugFF && !debugFF.killed) {
            try { debugFF.stdin.write(buf); } catch { }
        }
        if (upFF && !upFF.killed) {
            try { upFF.stdin.write(buf); } catch { }
        }
    });

    sio.on('stop_tts', () => {
        // Clear pending audio so we can interrupt promptly
        upB = Buffer.alloc(0);
    });
}

// ─── Resampling (ffmpeg) ─────────────────────────────────────────────────────
function initFfmpegPipes() {
    // 48k -> 16k for uplink towards Freeze-Omni
    downFF = spawn(ffmpegPath, [
        '-fflags', 'nobuffer', '-flags', 'low_delay', '-probesize', '32', '-analyzeduration', '0',
        '-f', 's16le', '-ar', String(WIRE_SR), '-ac', '1', '-i', 'pipe:0',
        '-f', 's16le', '-ar', String(FO_UP_SR), '-ac', '1', 'pipe:1'
    ]);
    downFF.stdout.on('data', (d) => {
        downB = Buffer.concat([downB, d]);
        sendFoUplinkFrames();
    });
    downFF.stderr.on('data', () => { });

    // 24k -> 48k for downlink from Freeze-Omni
    upFF = spawn(ffmpegPath, [
        '-fflags', 'nobuffer', '-flags', 'low_delay', '-probesize', '32', '-analyzeduration', '0',
        '-f', 's16le', '-ar', String(FO_DOWN_SR), '-ac', '1', '-i', 'pipe:0',
        '-f', 's16le', '-ar', String(WIRE_SR), '-ac', '1', 'pipe:1'
    ]);
    upFF.stdout.on('data', (d) => { upB = Buffer.concat([upB, d]); });
    upFF.stderr.on('data', () => { });

    if (SAVE_DEBUG) {
        // Save raw server (24k) to wav for debugging
        const { spawn: _spawn } = require('child_process');
        const sessionDir = ensureDebugDir();
        const dbgPath = path.join(sessionDir, `${ROLE}_fo_server.wav`);
        debugFF = _spawn(ffmpegPath, [
            '-fflags', 'nobuffer', '-flags', 'low_delay', '-probesize', '32', '-analyzeduration', '0',
            '-f', 's16le', '-ar', String(FO_DOWN_SR), '-ac', '1', '-i', 'pipe:0',
            '-f', 'wav', '-ac', '1', '-ar', String(FO_DOWN_SR), dbgPath
        ]);
        debugFF.stderr.on('data', () => { });
        console.log(`[Adapter-${ROLE}] Debug FO server audio → ${dbgPath}`);
    }
}

function ensureDebugDir() {
    const outRoot = process.env.RECORD_DIR ? path.resolve(process.env.RECORD_DIR) : path.join(__dirname, '..', 'outputs');
    const dir = path.join(outRoot, `adapter_${new Date().toISOString().replace(/[:.]/g, '-')}`);
    try { fs.mkdirSync(dir, { recursive: true }); } catch { }
    return dir;
}

// Send 16 kHz Int16 blocks as Socket.IO 'audio' events (JSON bytes), ~UPLINK_MS per frame
function sendFoUplinkFrames() {
    const bytesPerSample = 2;
    const samplesPerFrame = Math.round((FO_UP_SR * UPLINK_MS) / 1000);
    const CHUNK = samplesPerFrame * bytesPerSample;
    while (downB.length >= CHUNK) {
        const frame = downB.slice(0, CHUNK);
        downB = downB.slice(CHUNK);
        if (!sio || !sio.connected) continue;
        try {
            // Socket.IO demo client sends Uint8 array of Int16 buffer; emulate that
            const uint8 = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
            sio.emit('audio', JSON.stringify({ sample_rate: FO_UP_SR, audio: Array.from(uint8) }));
            if (LOG_FRAMES) console.log(`[Adapter-${ROLE}] → FO ${UPLINK_MS}ms frame (${CHUNK} bytes)`);
        } catch (e) {
            // swallow send errors; rely on reconnection
        }
    }
}

// ─── Shutdown ────────────────────────────────────────────────────────────────
async function shutdown() {
    console.log(`[Adapter-${ROLE}] Shutting down...`);
    try { if (sio && sio.connected) { try { sio.emit('recording-stopped'); } catch { }; sio.close(); } } catch { }
    try { if (wirePacer) clearInterval(wirePacer); } catch { }
    try { if (orchSink) orchSink.stop(); } catch { }
    try { if (pc) pc.close(); } catch { }
    try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch { }
    try { if (downFF && !downFF.killed) { downFF.stdin.end(); downFF.kill(); } } catch { }
    try { if (upFF && !upFF.killed) { upFF.stdin.end(); upFF.kill(); } } catch { }
    try { if (debugFF && !debugFF.killed) { debugFF.stdin.end(); debugFF.kill(); } } catch { }
    process.exit(0);
}