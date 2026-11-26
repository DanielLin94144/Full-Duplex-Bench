'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// Bound native hangs as well as JavaScript failures.
if (process.argv[2] !== '--worker') {
  const result = spawnSync(process.execPath, [__filename, '--worker'], {
    stdio: 'inherit', timeout: 15000, killSignal: 'SIGKILL',
  });
  if (result.error || result.signal) {
    console.error(result.error || `WebRTC test terminated by ${result.signal}`);
  }
  process.exit(result.status === 0 ? 0 : 1);
}

async function testAudio() {
  const wrtc = require('wrtc');
  const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;
  const peers = [];
  const sources = [];
  const tracks = [];
  const sinks = [];
  const errors = [];
  const audibleFrames = [0, 0];
  let timer;

  try {
    for (let i = 0; i < 2; i++) {
      const peer = new wrtc.RTCPeerConnection();
      peers.push(peer);
      const source = new RTCAudioSource();
      sources.push(source);
      const track = source.createTrack();
      tracks.push(track);
      peer.addTrack(track);
      peer.ontrack = ({ track: remoteTrack }) => {
        tracks.push(remoteTrack);
        const sink = new RTCAudioSink(remoteTrack);
        sinks.push(sink);
        sink.ondata = ({ samples, sampleRate, channelCount, bitsPerSample }) => {
          if (sampleRate === 48000 && channelCount === 1 && bitsPerSample === 16
              && samples.length === 480 && samples.some(value => Math.abs(value) > 100)) {
            audibleFrames[i]++;
          }
        };
      };
    }

    // Wait for each remote description before passing its ICE candidates.
    const pending = [[], []];
    for (let i = 0; i < 2; i++) {
      peers[i].onicecandidate = ({ candidate }) => {
        if (!candidate) return;
        const other = 1 - i;
        if (!peers[other].remoteDescription) pending[other].push(candidate);
        else peers[other].addIceCandidate(candidate).catch(error => errors.push(error));
      };
    }
    const offer = await peers[0].createOffer();
    await peers[0].setLocalDescription(offer);
    await peers[1].setRemoteDescription(new wrtc.RTCSessionDescription(offer));
    for (const candidate of pending[1].splice(0)) await peers[1].addIceCandidate(candidate);
    const answer = await peers[1].createAnswer();
    await peers[1].setLocalDescription(answer);
    await peers[0].setRemoteDescription(new wrtc.RTCSessionDescription(answer));
    for (const candidate of pending[0].splice(0)) await peers[0].addIceCandidate(candidate);

    let frame = 0;
    timer = setInterval(() => {
      sources.forEach((source, i) => {
        const samples = new Int16Array(480);
        for (let j = 0; j < samples.length; j++) {
          samples[j] = Math.round(10000 * Math.sin(
            2 * Math.PI * (440 + i * 220) * (frame * 480 + j) / 48000,
          ));
        }
        source.onData({ samples, sampleRate: 48000, bitsPerSample: 16, channelCount: 1 });
      });
      frame++;
    }, 10);

    const deadline = Date.now() + 8000;
    while (audibleFrames.some(count => count < 100) && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    assert.deepEqual(errors, [], 'ICE candidate errors');
    assert.deepEqual(peers.map(peer => peer.connectionState), ['connected', 'connected']);
    assert(audibleFrames.every(count => count >= 100),
      `Expected audio in both directions; received ${audibleFrames.join(', ')} frames`);
    return `WebRTC audio passed on ${process.platform}/${process.arch}, Node ${process.version}`;
  } finally {
    clearInterval(timer);
    sinks.forEach(sink => sink.stop());
    tracks.forEach(track => track.stop());
    peers.forEach(peer => peer.close());
  }
}

// Match the adapters' explicit exit. This checks audio transport, not native
// teardown at natural process exit, which can crash with both wrtc versions.
testAudio().then(
  message => process.stdout.write(`${message}\n`, () => process.exit(0)),
  error => process.stderr.write(`${error.stack}\n`, () => process.exit(1)),
);
