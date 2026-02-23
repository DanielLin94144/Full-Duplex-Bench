#!/usr/bin/env node
/**
 * moshi_adapter_seed.js (Role A/B)
 * Bridges orchestrator (WebRTC 48 kHz PCM16, 10 ms) <-> Moshi WS (Opus @24 kHz, 80 ms)
 */

require('dotenv').config();
const WebSocket = require('ws');
const wrtc = require('wrtc');
const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

// Optional Opus implementations (prefer native)
let OpusEncoder; // disabled for 24 kHz pipeline
try { OpusEncoder = null; } catch {}
let OpusScript = null;
try { OpusScript = require('opusscript'); } catch {}

// Audio configuration from environment variables with defaults
const ORCH_SR = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;
const MOSHI_SR = parseInt(process.env.MOSHI_SAMPLE_RATE, 10) || 24000;
const ORCH_10MS_SMP = Math.floor(ORCH_SR * 0.01);    // 10 ms samples
const ORCH_10MS_BYTES = ORCH_10MS_SMP * 2;           // Int16 mono bytes
const MOSHI_80MS_SMP = Math.floor(MOSHI_SR * 0.08);  // 80 ms @MOSHI_SR
const MOSHI_20MS_SMP = Math.floor(MOSHI_SR * 0.02);  // 20 ms @MOSHI_SR, Opus frame size

// Default Moshi URL from environment
const DEFAULT_MOSHI_URL = process.env.MOSHI_URL || 'ws://localhost:8998/api/chat';

const argv = yargs(hideBin(process.argv))
  .scriptName('moshi_adapter_seed')
  .usage('$0 --role <A|B> --signalUrl <ws://host:port/signal> [--moshiUrl <ws://host:port/api/chat>] [opts]')
  .option('role', { type: 'string', choices: ['A','B'], demandOption: true })
  .option('signalUrl', { type: 'string', demandOption: true })
  .option('moshiUrl', { type: 'string', default: DEFAULT_MOSHI_URL, describe: 'Moshi WebSocket URL' })
  .option('waitForHandshakeMs', { type: 'number', default: 800 })
  .option('log', { type: 'boolean', default: true })
  .help().argv;

const ROLE = argv.role;
const SIGNAL_URL = argv.signalUrl;
const MOSHI_URL = argv.moshiUrl;
const HANDSHAKE_MS = argv.waitForHandshakeMs;
const LOG = argv.log;

let orchPC, orchSource, orchSink, orchWS;
let moshiWS;
let uplinkBuf48k = Buffer.alloc(0);
let upPacer = null;

// Utilities
function i16ToF32(buf){
  const out = new Float32Array(buf.length/2);
  for (let i=0, j=0; i<buf.length; i+=2, j++) out[j] = Math.max(-1, Math.min(1, buf.readInt16LE(i) / 32768));
  return out;
}
function f32ToI16(f){
  const out = Buffer.alloc(f.length * 2);
  for (let i=0; i<f.length; i++) out.writeInt16LE((Math.max(-1, Math.min(1, f[i])) * 32768)|0, i*2);
  return out;
}
function ds48kTo24k(f){
  const n = f.length & ~1; // even length
  const out = new Float32Array(n/2);
  for (let i=0, j=0; i<n; i+=2, j++) out[j] = 0.5 * (f[i] + f[i+1]);
  return out;
}
function us24kTo48k(f){
  if (f.length === 0) return new Float32Array(0);
  const out = new Float32Array(f.length * 2);
  for (let i=0, j=0; i<f.length-1; i++, j+=2){
    out[j] = f[i];
    out[j+1] = 0.5 * (f[i] + f[i+1]);
  }
  out[out.length-2] = f[f.length-1];
  out[out.length-1] = f[f.length-1];
  return out;
}

let opusEncNative = null; // force JS encoder to avoid 48k-only native encoder issues
if (!OpusScript) throw new Error('opusscript is required (npm i opusscript)');
const opusScriptEnc = new OpusScript(MOSHI_SR, 1, OpusScript.Application.AUDIO);
const opusScriptDec = new OpusScript(MOSHI_SR, 1, OpusScript.Application.AUDIO);

// ─────────────────────────────────────────────────────────────────────────────
// Minimal Ogg Opus mux/demux (single-stream, one packet per page)
// ─────────────────────────────────────────────────────────────────────────────
const OGG_CAPTURE = Buffer.from('4f676753', 'hex'); // 'OggS'

// CRC32 per Ogg spec (MSB-first, poly 0x04C11DB7, init 0, no final xor)
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++){
    let r = i << 24;
    for (let j = 0; j < 8; j++){
      if (r & 0x80000000) r = ((r << 1) ^ 0x04C11DB7) >>> 0; else r = (r << 1) >>> 0;
    }
    table[i] = r >>> 0;
  }
  return table;
})();

function crc32Ogg(buf){
  let crc = 0 >>> 0;
  for (let i = 0; i < buf.length; i++){
    const idx = ((crc >>> 24) ^ (buf[i] & 0xff)) & 0xff;
    crc = (((crc << 8) >>> 0) ^ CRC_TABLE[idx]) >>> 0;
  }
  return crc >>> 0;
}

function le32(n){ const b = Buffer.alloc(4); b.writeUInt32LE(n >>> 0, 0); return b; }
function le16(n){ const b = Buffer.alloc(2); b.writeUInt16LE(n & 0xffff, 0); return b; }
function le64(n){
  const b = Buffer.alloc(8);
  // n fits in 53-bit JS number; we only need small increments here
  const lo = n >>> 0;
  const hi = Math.floor(n / 0x100000000) >>> 0;
  b.writeUInt32LE(lo, 0);
  b.writeUInt32LE(hi, 4);
  return b;
}

class OpusOggMuxer {
  constructor(sampleRate, channels){
    this.sampleRate = sampleRate;
    this.channels = channels;
    this.serial = (Math.random() * 0xffffffff) >>> 0;
    this.seq = 0;
    this.granulePos = 0; // increment per packet
  }

  buildPage(packet, headerType, granulePos){
    // Build Ogg page with one logical packet (lacing values for packet)
    const laces = [];
    let remaining = packet.length;
    while (remaining >= 255){ laces.push(255); remaining -= 255; }
    laces.push(remaining);
    const segCount = laces.length;
    const lacesBuf = Buffer.from(laces);
    const header = Buffer.concat([
      OGG_CAPTURE,
      Buffer.from([0x00]), // stream structure version
      Buffer.from([headerType & 0xff]),
      le64(granulePos),
      le32(this.serial),
      le32(this.seq++),
      le32(0x00000000), // CRC placeholder
      Buffer.from([segCount & 0xff]),
      lacesBuf
    ]);
    const page = Buffer.concat([header, packet]);
    const withZeroCRC = Buffer.from(page);
    // Zero CRC field before computing
    withZeroCRC.writeUInt32LE(0, 22);
    const crc = crc32Ogg(withZeroCRC);
    page.writeUInt32LE(crc >>> 0, 22);
    return page;
  }

  buildIdHeader(){
    // OpusHead
    const head = Buffer.concat([
      Buffer.from('4f70757348656164', 'hex'), // 'OpusHead'
      Buffer.from([0x01]),                    // version
      Buffer.from([this.channels & 0xff]),    // channel count
      le16(312),                              // pre-skip typical
      le32(48000),                            // Opus nominal input sample rate
      le16(0),                                // output gain
      Buffer.from([0x00])                     // channel mapping family 0 (mono/stereo)
    ]);
    // BOS page
    return this.buildPage(head, 0x02 /* BOS */, 0);
  }

  buildTagsHeader(){
    // OpusTags with vendor string and zero user comments
    const vendor = Buffer.from('MoshiAdapterSeed', 'utf8');
    const vendorLen = le32(vendor.length);
    const userCommentListLength = le32(0);
    const tags = Buffer.concat([
      Buffer.from('4f70757354616773', 'hex'), // 'OpusTags'
      vendorLen,
      vendor,
      userCommentListLength
    ]);
    return this.buildPage(tags, 0x00, 0);
  }

  buildAudioPage(opusPacket, samplesPerPacket){
    this.granulePos = (this.granulePos + (samplesPerPacket >>> 0)) >>> 0;
    return this.buildPage(opusPacket, 0x00, this.granulePos);
  }
}

class OggDemuxer {
  constructor(onPacket){
    this.buf = Buffer.alloc(0);
    this.onPacket = onPacket;
    this.continued = false;
    this.partial = Buffer.alloc(0);
  }
  push(data){
    this.buf = Buffer.concat([this.buf, data]);
    while (this.buf.length >= 27){
      if (!this.buf.slice(0,4).equals(OGG_CAPTURE)){
        // Search for sync
        const idx = this.buf.indexOf(OGG_CAPTURE);
        if (idx < 0){ this.buf = Buffer.alloc(0); return; }
        this.buf = this.buf.slice(idx);
        if (this.buf.length < 27) return;
      }
      const pageSegments = this.buf.readUInt8(26);
      const minSize = 27 + pageSegments;
      if (this.buf.length < minSize) return;
      const segmentTable = this.buf.slice(27, 27 + pageSegments);
      let bodyLen = 0;
      for (let i=0;i<pageSegments;i++) bodyLen += segmentTable[i];
      const pageLen = 27 + pageSegments + bodyLen;
      if (this.buf.length < pageLen) return;
      const headerType = this.buf.readUInt8(5);
      const body = this.buf.slice(27 + pageSegments, pageLen);
      // Parse lacing into packets (handle continued packets)
      let offset = 0;
      for (let i=0;i<pageSegments;i++){
        const seg = segmentTable[i];
        const segData = body.slice(offset, offset + seg);
        offset += seg;
        this.partial = Buffer.concat([this.partial, segData]);
        if (seg < 255){
          // packet complete
          const pkt = this.partial;
          this.partial = Buffer.alloc(0);
          // Skip header packets (OpusHead/OpusTags) by magic strings
          if (pkt.length >= 8){
            const magic = pkt.slice(0,8);
            if (magic.equals(Buffer.from('4f70757348656164','hex')) || magic.equals(Buffer.from('4f70757354616773','hex'))){
              continue;
            }
          }
          this.onPacket(pkt);
        }
      }
      this.buf = this.buf.slice(pageLen);
    }
  }
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

(async function main(){
  if (LOG) console.log(`[MoshiSeed-${ROLE}] Start (signal=${SIGNAL_URL}) moshi=${MOSHI_URL}`);
  await connectOrchestrator();
  await connectMoshi();
  startBridge();
})();

async function connectOrchestrator(){
  orchPC = new wrtc.RTCPeerConnection();
  orchSource = new RTCAudioSource({ sampleRate: ORCH_SR, channelCount: 1 });
  const outTrack = orchSource.createTrack();
  orchPC.addTrack(outTrack);

  orchPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (orchSink) try { orchSink.stop(); } catch {}
    orchSink = new RTCAudioSink(track, { sampleRate: ORCH_SR, channelCount: 1, bitDepth: 16 });
    orchSink.ondata = ({ samples }) => {
      const view = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      uplinkBuf48k = Buffer.concat([uplinkBuf48k, view]);
    };
  };

  orchWS = new WebSocket(SIGNAL_URL);
  await new Promise(res => orchWS.once('open', res));

  orchPC.onicecandidate = ({ candidate }) => {
    if (!candidate) return;
    orchWS.send(JSON.stringify({ role: ROLE, type: 'candidate', candidate: {
      candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex
    }}));
  };

  const offer = await orchPC.createOffer();
  await orchPC.setLocalDescription(offer);
  orchWS.send(JSON.stringify({ role: ROLE, type: 'offer', sdp: offer.sdp }));

  orchWS.on('message', async (msg) => {
    const data = JSON.parse(msg);
    if (data.type === 'answer' && data.role === ROLE) {
      await orchPC.setRemoteDescription({ type: 'answer', sdp: data.sdp });
      if (LOG) console.log(`[MoshiSeed-${ROLE}] Orchestrator answer applied`);
    } else if (data.type === 'candidate' && data.role === ROLE) {
      try { await orchPC.addIceCandidate(data.candidate); } catch {}
    }
  });
}

async function connectMoshi(){
  moshiWS = new WebSocket(MOSHI_URL);
  await new Promise(res => moshiWS.once('open', res));
  // server sends 0x00 handshake; we just wait briefly
  await new Promise(r => setTimeout(r, HANDSHAKE_MS));
  // Send Opus Ogg headers so the server's OpusStreamReader can initialize
  try {
    opusMux = new OpusOggMuxer(MOSHI_SR, 1);
    const idPg = opusMux.buildIdHeader();
    const tagsPg = opusMux.buildTagsHeader();
    moshiWS.send(Buffer.concat([Buffer.from([1]), idPg]));
    moshiWS.send(Buffer.concat([Buffer.from([1]), tagsPg]));
  } catch (e) {
    if (LOG) console.warn(`[MoshiSeed-${ROLE}] Failed to send Opus headers:`, e?.message || e);
  }
  // Downlink demuxer: parse Ogg pages into Opus packets
  oggDemux = new OggDemuxer((pkt) => {
    try {
      const decoded = opusScriptDec.decode(pkt, MOSHI_20MS_SMP);
      let f24;
      if (decoded instanceof Float32Array) f24 = decoded;
      else if (decoded instanceof Int16Array){
        f24 = new Float32Array(decoded.length);
        for (let i=0;i<decoded.length;i++) f24[i] = Math.max(-1, Math.min(1, decoded[i]/32768));
      } else if (Buffer.isBuffer(decoded)){
        const i16 = new Int16Array(decoded.buffer, decoded.byteOffset, decoded.byteLength/2);
        f24 = new Float32Array(i16.length);
        for (let i=0;i<i16.length;i++) f24[i] = Math.max(-1, Math.min(1, i16[i]/32768));
      } else return;
      const f48 = us24kTo48k(f24);
      const i16b = f32ToI16(f48);
      for (let p=0; p<i16b.length; p+=ORCH_10MS_BYTES){
        const slice = i16b.subarray(p, p+ORCH_10MS_BYTES);
        if (slice.length < ORCH_10MS_BYTES) break;
        const ab = new ArrayBuffer(ORCH_10MS_BYTES);
        new Uint8Array(ab).set(slice);
        const samples = new Int16Array(ab);
        orchSource.onData({ samples, sampleRate: ORCH_SR, bitsPerSample: 16, channelCount: 1 });
      }
    } catch (e) {
      if (LOG) console.warn(`[MoshiSeed-${ROLE}] decode error:`, e?.message || e);
    }
  });
  moshiWS.on('message', (msg) => {
    if (!(msg instanceof Buffer) || msg.length === 0) return;
    const mt = msg[0];
    const payload = msg.subarray(1);
    if (mt === 1) oggDemux.push(payload);
    else if (mt === 2){ if (LOG) console.log(`[MoshiSeed-${ROLE}] [TEXT]`, payload.toString('utf8')); }
  });
}

function startBridge(){
  // Uplink: every 80 ms, ship 4x 20ms Opus frames as separate MT=1 WS messages
  upPacer = setInterval(() => {
    try {
      const needBytes = ORCH_10MS_BYTES * 8; // 80 ms worth at 48k
      let src = uplinkBuf48k;
      let chunk;
      if (src.length >= needBytes) {
        chunk = src.slice(0, needBytes);
        uplinkBuf48k = src.slice(needBytes);
      } else if (src.length > 0) {
        chunk = Buffer.alloc(needBytes);
        src.copy(chunk, 0, 0, src.length);
        uplinkBuf48k = Buffer.alloc(0);
      } else {
        chunk = Buffer.alloc(needBytes); // silence
      }
      const f32 = i16ToF32(chunk);
      const f24 = ds48kTo24k(f32);
      for (let off=0; off<f24.length; off+=MOSHI_20MS_SMP){
        const slice = f24.subarray(off, off+MOSHI_20MS_SMP);
        if (slice.length < MOSHI_20MS_SMP) break;
        const pcm16 = f32ToI16(slice);
        let pkt;
        if (opusEncNative && typeof opusEncNative.encode === 'function') {
          pkt = Buffer.from(opusEncNative.encode(pcm16, MOSHI_20MS_SMP));
        } else if (opusScriptEnc) {
          pkt = Buffer.from(opusScriptEnc.encode(pcm16, MOSHI_20MS_SMP));
        } else {
          throw new Error('No Opus encoder available');
        }
        // Wrap into Ogg page and send as MT=1
        const page = opusMux.buildAudioPage(pkt, /*granule_step@48k*/ 960);
        moshiWS.send(Buffer.concat([Buffer.from([1]), page]));
      }
    } catch (e) {
      if (LOG) console.warn(`[MoshiSeed-${ROLE}] uplink error:`, e?.message || e);
    }
  }, 80);
}

// stateful mux/demux instances
let opusMux = null;
let oggDemux = null;

async function shutdown(){
  if (LOG) console.log(`[MoshiSeed-${ROLE}] Shutting down…`);
  try { if (upPacer) clearInterval(upPacer); } catch {}
  try { if (orchSink) orchSink.stop(); } catch {}
  try { if (orchPC) await orchPC.close(); } catch {}
  try { if (moshiWS && moshiWS.readyState === WebSocket.OPEN) moshiWS.close(); } catch {}
  try { if (orchWS && orchWS.readyState === WebSocket.OPEN) orchWS.close(); } catch {}
  process.exit(0);
}


