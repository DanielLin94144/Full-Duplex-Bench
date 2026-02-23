#!/usr/bin/env node
/**
 * gpt4oRealtime_adapter.js (Role A/B)
 * Bridges your orchestrator (WebRTC @ 48 kHz PCM16, 50 ms cadence) <->
 * OpenAI Realtime (WebRTC) using an ephemeral session token from openai_token_server.js.
 *
 * Key behaviors
 *  - Reuses your orchestrator's WS signaling contract (/signal) and 48 kHz timing.
 *  - Creates a separate RTCPeerConnection to OpenAI; audio both directions via WebRTC tracks.
 *  - Uses a data channel to send `session.update` and (optionally) response control messages.
 */

require('dotenv').config();
const fs = require('fs');
const wrtc = require('wrtc');
const WebSocket = require('ws');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;

// Configuration from environment variables with defaults
const DEFAULT_MODEL = process.env.OPENAI_REALTIME_MODEL || 'gpt-realtime-2025-08-28';
const DEFAULT_TOKEN_SERVER = process.env.OPENAI_TOKEN_SERVER || 'http://localhost:3002';
const DEFAULT_TOKEN_SERVER_PORT = parseInt(process.env.TOKEN_SERVER_PORT, 10) || 3002;
const WIRE_SR = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;  // Orchestrator wire sample rate
const CHUNK_BYTES = Math.floor(WIRE_SR * 0.05) * 2;  // 50 ms @ WIRE_SR PCM16 mono

// Prefer global fetch (Node >=18). Fallback to node-fetch if needed.
const _fetch = (typeof fetch !== 'undefined') ? fetch : (async (...args) => (await import('node-fetch')).default(...args));

const argv = yargs(hideBin(process.argv))
  .scriptName('gpt4oRealtime_adapter')
  .usage('$0 -r <A|B> -s <ws://host:port/signal> [opts]')
  .option('role', { alias: 'r', type: 'string', choices: ['A', 'B'], demandOption: true })
  .option('signalUrl', { alias: 's', type: 'string', demandOption: true })
  .option('tokenServer', { type: 'string', default: DEFAULT_TOKEN_SERVER })
  .option('model', { type: 'string', default: DEFAULT_MODEL })
  .option('voice', { type: 'string', default: 'alloy' })
  .option('systemPrompt', { type: 'string', default: '' })
  .option('turnMode', { type: 'string', choices: ['server_vad', 'none'], default: 'server_vad' })
  .option('vadMode', { type: 'string', choices: ['slow', 'fast'], default: 'slow', describe: 'VAD timing mode: slow (default, more patient) or fast (quicker response)' })
  .option('logEvents', { type: 'boolean', default: true })
  .option('prefill', { type: 'string', default: '', describe: 'Initial user text to append as history' })
  .option('prefillFile', { type: 'string', default: '', describe: 'Path to JSON array of prior turns [{role, text}]' })
  .option('autostart', { type: 'boolean', default: true, describe: 'Create response automatically after prefill' })
  .help().argv;

const ROLE = argv.role;
const SIGNAL_URL = argv.signalUrl;
const TOKEN_SERVER_URL = argv.tokenServer;
const MODEL = argv.model;
const VOICE = argv.voice;
const SYSTEM_PROMPT = argv.systemPrompt || '';
const TURN_MODE = argv.turnMode;   // 'server_vad' or 'none' (push-to-talk)
const VAD_MODE = argv.vadMode;     // 'slow' (default) or 'fast' (stricter threshold, shorter silence)
const PREFILL_TEXT = argv.prefill || '';
const PREFILL_FILE = argv.prefillFile || '';

// Default to autostart for Role A (Examiner), disable for Role B (Examinee) to avoid simultaneous talking
const AUTOSTART = argv.autostart !== undefined ? argv.autostart : (ROLE === 'A');

// ------------- Globals -------------
let orchPC, orchSource, orchSink;             // to Orchestrator
let aiPC, aiSource, aiSink, aiDC;             // to OpenAI Realtime
let ws;                                       // WS signaling to orchestrator

let uplinkBuf = Buffer.alloc(0);              // Orchestrator -> OpenAI bytes
let downlinkBuf = Buffer.alloc(0);            // OpenAI -> Orchestrator bytes
let uplinkPacer = null;
let downlinkPacer = null;
let openaiSpeaking = false;                   // heuristic flag
let prefillSent = false;                      // ensure prefill runs once

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

(async function main() {
  console.log(`[Adapter-${ROLE}] Starting (signal=${SIGNAL_URL}) model=${MODEL} voice=${VOICE} turn=${TURN_MODE} vad=${VAD_MODE}`);
  await connectOrchestrator();
  await connectOpenAI();
  startBridgePacers();
})();

// ------------- Orchestrator leg -------------
async function connectOrchestrator() {
  orchPC = new wrtc.RTCPeerConnection();
  orchSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  const outTrack = orchSource.createTrack();
  orchPC.addTrack(outTrack);

  orchPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (orchSink) orchSink.stop();
    orchSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
    let count = 0;
    orchSink.ondata = ({ samples }) => {
      const view = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      if (count++ < 5) console.log(`[Adapter-${ROLE}] orchSink first bytesIn=${view.length}`);
      uplinkBuf = Buffer.concat([uplinkBuf, view]);
    };
  };

  ws = new WebSocket(SIGNAL_URL);
  ws.on('error', (err) => console.error(`[Adapter-${ROLE}] Orchestrator WS error:`, err));
  await new Promise((res) => ws.once('open', res));

  orchPC.onicecandidate = ({ candidate }) => {
    if (!candidate) return;
    ws.send(JSON.stringify({
      role: ROLE, type: 'candidate', candidate: {
        candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex
      }
    }));
  };

  const offer = await orchPC.createOffer();
  await orchPC.setLocalDescription(offer);
  ws.send(JSON.stringify({ role: ROLE, type: 'offer', sdp: offer.sdp }));

  ws.on('message', async (msg) => {
    const data = JSON.parse(msg);
    if (data.type === 'answer' && data.role === ROLE) {
      await orchPC.setRemoteDescription({ type: 'answer', sdp: data.sdp });
      console.log(`[Adapter-${ROLE}] Orchestrator answer applied.`);
    } else if (data.type === 'candidate' && data.role === ROLE) {
      // Orchestrator broadcasts; only pick candidates destined for our peer
      try { await orchPC.addIceCandidate(data.candidate); } catch { }
    }
  });
}

// ------------- OpenAI leg (WebRTC) -------------
async function connectOpenAI() {
  // 1) Get ephemeral session
  const session = await postJSON(`${TOKEN_SERVER_URL}/session`, {
    model: MODEL,
    // You can preconfigure these here or send via data channel later:
    // voice: VOICE,
    // instructions: SYSTEM_PROMPT,
    // turn_detection: TURN_MODE === 'server_vad' ? { type: 'server_vad' } : undefined,
  });
  const EPHEMERAL_KEY = session?.client_secret?.value;
  if (!EPHEMERAL_KEY) throw new Error('Ephemeral key missing from token server response.');

  // 2) Build PC toward OpenAI
  aiPC = new wrtc.RTCPeerConnection();
  aiSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  const aiOutTrack = aiSource.createTrack();
  aiPC.addTrack(aiOutTrack);

  // Receive audio back from OpenAI
  aiPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (aiSink) aiSink.stop();
    aiSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
    let first = true;
    aiSink.ondata = ({ samples }) => {
      const view = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      if (first) { console.log(`[Adapter-${ROLE}] OpenAI sink first bytesIn=${view.length}`); first = false; }
      downlinkBuf = Buffer.concat([downlinkBuf, view]);
      openaiSpeaking = true; // heuristic; timer cleared in pacer when silence padded
    };
  };

  // Data channel for Realtime events
  aiDC = aiPC.createDataChannel('oai-events');
  aiDC.onopen = () => {
    console.log(`[Adapter-${ROLE}] OpenAI data channel open. VAD mode: ${VAD_MODE}`);
    // Build turn_detection config based on VAD_MODE
    let turn_detection_config;
    if (TURN_MODE === 'none') {
      turn_detection_config = undefined;
    } else if (VAD_MODE === 'fast') {
      // Fast mode: stricter threshold, shorter silence duration for quicker response
      turn_detection_config = {
        type: 'server_vad',
        create_response: true,
        interrupt_response: false,
        threshold: 0.6,
        prefix_padding_ms: 1200,
        silence_duration_ms: 600
      };
    } else {
      // Slow mode (default): more patient, allows interruption
      turn_detection_config = {
        type: 'server_vad',
        create_response: true,
        interrupt_response: true
      };
    }

    // Send initial session.update
    const msg = {
      type: 'session.update',
      session: {
        instructions: SYSTEM_PROMPT,
        voice: VOICE,
        turn_detection: turn_detection_config,
        input_audio_transcription: undefined, // add if you want live transcripts
      }
    };
    safeSendDC(msg);
  };
  let connectionStartTime = Date.now();

  aiDC.onmessage = (ev) => {
    try {
      if (argv.logEvents) console.log(`[Adapter-${ROLE}] ← event`, ev.data);
      const evt = JSON.parse(ev.data);
      if (evt?.type === 'session.updated' && !prefillSent) {
        sendPrefillHistory();
      }

      // Prevent Examinee from speaking before Examiner
      if (ROLE === 'B' && evt?.type === 'response.created') {
        if (Date.now() - connectionStartTime < 4000) {
          console.log(`[Adapter-B] Canceling early response to wait for Examiner`);
          safeSendDC({ type: 'response.cancel' });
        }
      }
    } catch { }
  };
  aiDC.onerror = (e) => console.error(`[Adapter-${ROLE}] datachannel error`, e);

  // 3) Offer/answer via HTTP SDP exchange
  // Make sure ICE candidates are gathered before POSTing the SDP
  const offer = await aiPC.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: false });
  await aiPC.setLocalDescription(offer);
  await waitForIceGatheringComplete(aiPC);

  const sdpAnswer = await postSDP(`https://api.openai.com/v1/realtime?model=${encodeURIComponent(MODEL)}`, EPHEMERAL_KEY, aiPC.localDescription.sdp);
  await aiPC.setRemoteDescription({ type: 'answer', sdp: sdpAnswer });
  console.log(`[Adapter-${ROLE}] OpenAI answer applied.`);
}

function safeSendDC(obj) {
  if (!aiDC || aiDC.readyState !== 'open') return;
  try { aiDC.send(JSON.stringify(obj)); } catch (e) { console.error('DC send error', e); }
}

// ------------- Bridging -------------
function startBridgePacers() {
  if (uplinkPacer) clearInterval(uplinkPacer);
  if (downlinkPacer) clearInterval(downlinkPacer);

  const bridgeStartTime = Date.now();

  // Orchestrator -> OpenAI: emit ten 10ms frames per 100ms tick
  const FRAME_10MS = Math.floor(WIRE_SR * 0.01) * 2; // bytes at WIRE_SR PCM16 mono
  uplinkPacer = setInterval(() => {
    // For Role B (examinee), ignore the first 4 seconds of audio from orchestrator 
    // to prevent fake silence/noise from triggering VAD before the examiner speaks.
    const shouldMute = (ROLE === 'B' && (Date.now() - bridgeStartTime < 4000));

    for (let i = 0; i < 10; i += 1) {
      let out;
      if (uplinkBuf.length >= FRAME_10MS) {
        out = uplinkBuf.slice(0, FRAME_10MS);
        uplinkBuf = uplinkBuf.slice(FRAME_10MS);
      } else if (uplinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS);
        uplinkBuf.copy(out, 0, 0, uplinkBuf.length);
        uplinkBuf = Buffer.alloc(0);
      } else {
        out = Buffer.alloc(FRAME_10MS); // silence
      }

      if (shouldMute) {
        out.fill(0); // overwrite with pure silence
      }

      const ab = new ArrayBuffer(FRAME_10MS);
      new Uint8Array(ab).set(out);
      const samples = new Int16Array(ab);
      aiSource.onData({ samples, sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
    }
  }, 100);

  // OpenAI -> Orchestrator: emit ten 10ms frames per 100ms tick
  downlinkPacer = setInterval(() => {
    for (let i = 0; i < 10; i += 1) {
      let out;
      if (downlinkBuf.length >= FRAME_10MS) {
        out = downlinkBuf.slice(0, FRAME_10MS);
        downlinkBuf = downlinkBuf.slice(FRAME_10MS);
        openaiSpeaking = true;
      } else if (downlinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS);
        downlinkBuf.copy(out, 0, 0, downlinkBuf.length);
        downlinkBuf = Buffer.alloc(0);
        openaiSpeaking = true;
      } else {
        out = Buffer.alloc(FRAME_10MS); // silence
        openaiSpeaking = false;
      }
      const ab = new ArrayBuffer(FRAME_10MS);
      new Uint8Array(ab).set(out);
      const samples = new Int16Array(ab);
      orchSource.onData({ samples, sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
    }
  }, 100);
}

// ------------- Utils -------------
async function postJSON(url, body) {
  const r = await _fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`POST ${url} ${r.status}`);
  return await r.json();
}

async function postSDP(url, bearer, sdp) {
  const r = await _fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/sdp', 'Authorization': `Bearer ${bearer}` }, body: sdp });
  if (!r.ok) {
    const t = await r.text().catch(() => '');
    throw new Error(`SDP POST failed ${r.status}: ${t}`);
  }
  return await r.text();
}

function waitForIceGatheringComplete(pc) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise(resolve => {
    const check = () => {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', check);
    // Safety timeout in case some environments never report complete
    setTimeout(() => { pc.removeEventListener('icegatheringstatechange', check); resolve(); }, 1500);
  });
}

async function shutdown() {
  console.log(`[Adapter-${ROLE}] Shutting down...`);
  try { if (uplinkPacer) clearInterval(uplinkPacer); } catch { }
  try { if (downlinkPacer) clearInterval(downlinkPacer); } catch { }
  try { if (orchSink) orchSink.stop(); } catch { }
  try { if (aiSink) aiSink.stop(); } catch { }
  try { if (orchPC) orchPC.close(); } catch { }
  try { if (aiPC) aiPC.close(); } catch { }
  try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch { }
  process.exit(0);
}

// ------------- Prefill helpers -------------
function coercePrefillTurns() {
  // Accept either CLI --prefill as one user turn, or --prefillFile as JSON array of {role, text}
  // Valid roles per Realtime message are 'system'|'user'|'assistant'.
  try {
    if (PREFILL_FILE) {
      const raw = fs.readFileSync(PREFILL_FILE, 'utf8');
      const arr = JSON.parse(raw);
      if (Array.isArray(arr) && arr.length > 0) {
        return arr
          .filter(t => t && typeof t.role === 'string' && typeof t.text === 'string' && t.text.length > 0)
          .map(t => ({ role: t.role, text: t.text }));
      }
    }
  } catch (e) {
    console.warn(`[Adapter-${ROLE}] Failed to parse prefillFile:`, e?.message || e);
  }
  if (PREFILL_TEXT) {
    return [{ role: 'user', text: PREFILL_TEXT }];
  }
  return [];
}

function sendPrefillHistory() {
  const turns = coercePrefillTurns();
  if (!turns.length) {
    prefillSent = true;
    if (AUTOSTART) {
      safeSendDC({ type: 'response.create', response: { modalities: ['audio', 'text'] } });
    }
    return;
  }
  // Send systemPrompt as a system message if provided
  // if (SYSTEM_PROMPT) {
  //   safeSendDC({
  //     type: 'conversation.item.create',
  //     item: { type: 'message', role: 'system', content: [{ type: 'input_text', text: SYSTEM_PROMPT }] }
  //   });
  // }
  for (const t of turns) {
    safeSendDC({
      type: 'conversation.item.create',
      item: { type: 'message', role: t.role, content: [{ type: 'input_text', text: t.text }] }
    });
  }
  prefillSent = true;
  if (AUTOSTART) {
    safeSendDC({ type: 'response.create', response: { modalities: ['audio', 'text'] } });
  }
}
