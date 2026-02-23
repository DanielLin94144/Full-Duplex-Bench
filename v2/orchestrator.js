// orchestrator.js
// Central orchestrator with built-in signaling for WebRTC
console.log('[Orch] Loading modules...');

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
console.log('[Orch] Loading wrtc...');
const wrtc = require('wrtc');
console.log('[Orch] wrtc loaded successfully');
const { RTCPeerConnection, RTCSessionDescription } = wrtc;
const { RTCAudioSink, RTCAudioSource } = wrtc.nonstandard;
const wav = require('wav');
console.log('[Orch] All modules loaded');

// Global error handlers to catch crashes
process.on('uncaughtException', (err) => {
  console.error('[Orch] UNCAUGHT EXCEPTION:', err);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('[Orch] UNHANDLED REJECTION:', reason);
});

// Config - uses environment variables with defaults from .env.example
const env = process.env;
const SIGNAL_PORT = parseInt(env.SIGNAL_PORT, 10) || 9100;
const RECORD_DIR = env.RECORD_DIR || 'outputs';
const WIRE_SAMPLE_RATE = parseInt(env.WIRE_SAMPLE_RATE, 10) || 48000;

// State
let offers = { A: null, B: null };
let candidates = { A: [], B: [] };
let pcs = { A: null, B: null };
let sinks = { A: null, B: null };
let sources = { A: null, B: null };
let writers = { A: null, B: null };
let ready = { A: false, B: false };
let forwardBuffers = { A: Buffer.alloc(0), B: Buffer.alloc(0) };
// Debug/forwarding state
let firstLogsRemaining = { A: 12, B: 12 }; // log first few sink events per role
let framesOutPerSec = { A: 0, B: 0 };
let framesOutWindowStart = Date.now();
let backlogBytes = { A: 0, B: 0 };
let pacers = { A: null, B: null };

// Create output directory
const outDir = RECORD_DIR;
fs.mkdirSync(outDir, { recursive: true });

// WAV Writers helper - Record at WIRE_SAMPLE_RATE to match WebRTC
function createWriter(role) {
  const file = path.join(outDir, `${role}.wav`);
  return new wav.FileWriter(file, { sampleRate: WIRE_SAMPLE_RATE, channels: 1, bitDepth: 16 });
}

// Setup HTTP + WS signaling server
const app = express();
const server = http.createServer(app);

server.on('error', (err) => console.error('[Orch] HTTP server error:', err));
server.on('upgrade', (req, socket, head) => {
  console.log(`[Orch] HTTP upgrade request: ${req.url}`);
});

const wss = new WebSocket.Server({ server, path: '/signal' });
wss.on('error', (err) => console.error('[Orch] WebSocket server error:', err));

wss.on('connection', ws => {
  console.log('[Orch] New WebSocket connection');

  ws.on('error', (err) => {
    console.error('[Orch] WebSocket client error:', err.message);
  });

  ws.on('close', (code, reason) => {
    console.log(`[Orch] WebSocket closed: code=${code} reason=${reason || 'none'}`);
  });

  let connectionsSetup = false;

  ws.on('message', async msg => {
    const data = JSON.parse(msg);
    const { role, type, sdp, candidate } = data;
    console.log(`[Orch] Received: role=${role} type=${type}`);

    if (type === 'offer') {
      offers[role] = sdp;
      if (offers.A && offers.B && !connectionsSetup) {
        connectionsSetup = true;
        setupPeerConnections();
      }
      return;
    }
    if (type === 'candidate') {
      if (pcs[role]) pcs[role].addIceCandidate(candidate);
      else candidates[role].push(candidate);
      return;
    }
  });

  ws.send(JSON.stringify({ type: 'ready' }));
});

server.listen(SIGNAL_PORT, () => console.log(`Orchestrator signaling on port ${SIGNAL_PORT}`));

// Create and configure PeerConnections
async function setupPeerConnections() {
  console.log('[Orch] Setting up peer connections...');
  ['A', 'B'].forEach(role => {
    console.log(`[Orch] Creating RTCPeerConnection for role ${role}`);
    const pc = new RTCPeerConnection();
    pcs[role] = pc;

    // Local source track for sending to adapter
    const source = new RTCAudioSource({ sampleRate: WIRE_SAMPLE_RATE, channelCount: 1 });
    sources[role] = source;
    pc.addTrack(source.createTrack());

    // Apply remote offer and ICE candidates
    console.log(`[Orch] Setting remote description for role ${role}`);
    pc.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp: offers[role] }));
    candidates[role].forEach(c => pc.addIceCandidate(c));

    // Answer back
    pc.createAnswer().then(answer => {
      pc.setLocalDescription(answer);
      wss.clients.forEach(c => {
        if (c.readyState === WebSocket.OPEN) {
          c.send(JSON.stringify({ role, type: 'answer', sdp: answer.sdp }));
        }
      });
    });

    // Handle incoming audio
    pc.ontrack = ({ track }) => {
      if (track.kind !== 'audio') return;
      const sink = new RTCAudioSink(track, { sampleRate: WIRE_SAMPLE_RATE, channelCount: 1, bitDepth: 16 });
      sinks[role] = sink;
      writers[role] = createWriter(role);

      let sinkEventCount = { A: 0, B: 0 };
      sink.ondata = ({ samples }) => {
        const viewBuf = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
        const other = role === 'A' ? 'B' : 'A';
        sinkEventCount[role] += 1;
        if (firstLogsRemaining[role] > 0) {
          const mod960 = viewBuf.length % 960;
          const mod1920 = viewBuf.length % 1920;
          console.log(`[Orch] sink(${role})#${sinkEventCount[role]} bytesIn=${viewBuf.length} mod960=${mod960} mod1920=${mod1920}`);
          firstLogsRemaining[role] -= 1;
        }
        // Buffer audio for the other side only; pacing happens via per-role scheduler
        forwardBuffers[other] = Buffer.concat([forwardBuffers[other], viewBuf]);
        backlogBytes[other] = forwardBuffers[other].length;

        // Also record raw incoming audio (what we received from this role)
        writers[role].write(viewBuf);

        // *** Per-second frames-out stats ***
        // const now = Date.now();
        // if (now - framesOutWindowStart >= 1000) {
        //   console.log(`[Orch] framesOut/s A=${framesOutPerSec.A} B=${framesOutPerSec.B} | backlogBytes A=${backlogBytes.A} B=${backlogBytes.B}`);
        //   framesOutPerSec = { A: 0, B: 0 };
        //   framesOutWindowStart = now;
        // }
      };

      ready[role] = true;
      checkBothReady();

      // Start per-role 100ms pacer for the opposite direction
      const target = role === 'A' ? 'B' : 'A';
      if (!pacers[target]) {
        console.log(`[Orch] Starting pacer for role ${target}`);
        pacers[target] = setInterval(() => {
          const src = sources[target];
          if (!src) return;
          const FRAME_10MS = Math.floor(WIRE_SAMPLE_RATE * 0.01) * 2; // bytes, 10 ms @ WIRE_SAMPLE_RATE mono PCM16
          // Emit ten 10ms frames per 100ms tick
          for (let i = 0; i < 10; i += 1) {
            let out;
            const bufNow = forwardBuffers[target];
            if (bufNow.length >= FRAME_10MS) {
              out = bufNow.slice(0, FRAME_10MS);
              forwardBuffers[target] = bufNow.slice(FRAME_10MS);
            } else if (bufNow.length > 0) {
              out = Buffer.alloc(FRAME_10MS);
              bufNow.copy(out, 0, 0, bufNow.length);
              forwardBuffers[target] = Buffer.alloc(0);
            } else {
              out = Buffer.alloc(FRAME_10MS);
            }
            // Exact 960-byte ArrayBuffer for wrtc
            const ab = new ArrayBuffer(FRAME_10MS);
            new Uint8Array(ab).set(new Uint8Array(out.buffer, out.byteOffset, FRAME_10MS));
            const samples = new Int16Array(ab);
            src.onData({ samples, sampleRate: WIRE_SAMPLE_RATE, bitsPerSample: 16, channelCount: 1 });
            framesOutPerSec[target] += 1;
          }
        }, 100);
      }
    };

    pcs[role].onicecandidate = ({ candidate }) => {
      if (!candidate) return;
      const init = { candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex };
      wss.clients.forEach(c => {
        if (c.readyState === WebSocket.OPEN) (
          c.send(JSON.stringify({ role, type: 'candidate', candidate: init }))
        );
      });
    };
  });

  // No global silence injector; pacers enforce real-time cadence
}

function checkBothReady() {
  if (ready.A && ready.B) console.log('Both peers ready—streaming and recording.');
}

// Handle shutdown and mix
process.on('SIGINT', async () => {
  console.error('[Orch] SIGINT caught; outDir =', outDir);
  console.log('\nShutting down gracefully...');

  // Stop sinks
  ['A', 'B'].forEach(r => sinks[r] && sinks[r].stop());

  // Close WAV writers
  const closePromises = ['A', 'B'].map(role => new Promise(res => {
    if (writers[role]) writers[role].end(() => res());
    else res();
  }));
  await Promise.all(closePromises);

  // Close peer connections
  ['A', 'B'].forEach(r => { if (pcs[r]) pcs[r].close(); });

  // Stop pacers
  ['A', 'B'].forEach(role => { if (pacers[role]) clearInterval(pacers[role]); });

  // Mixdown - create stereo WAV with A on left channel, B on right channel
  const a = path.join(outDir, 'A.wav');
  const b = path.join(outDir, 'B.wav');
  const combined = path.join(outDir, 'combined.wav');

  console.error('[Orch] A.wav exists?', fs.existsSync(a));
  console.error('[Orch] B.wav exists?', fs.existsSync(b));

  if (fs.existsSync(a) && fs.existsSync(b)) {
    // Use amerge to create stereo output: A.wav -> left channel, B.wav -> right channel
    // -ac 2 ensures stereo output, duration=longest pads the shorter track with silence
    const cmd = `ffmpeg -i ${a} -i ${b} -filter_complex "[0:a][1:a]amerge=inputs=2,pan=stereo|c0<c0|c1<c1[aout]" -map "[aout]" -ac 2 ${combined}`;
    exec(cmd, (err, stdout, stderr) => {
      console.error('[Orch] ffmpeg stdout:', stdout);
      console.error('[Orch] ffmpeg stderr:', stderr);
      if (err) console.error('Error creating combined stereo mix:', err);
      else console.log('Combined stereo mix created (A=left, B=right):', combined);
      process.exit();
    });
  } else {
    console.error('Warning: Audio files not found, skipping mix creation');
    process.exit();
  }
});
