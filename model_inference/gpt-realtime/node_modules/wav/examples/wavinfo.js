
/**
 * Pipe a WAVE file to stdin, or specify the filename as the first argument,
 * and information about the wave file will be printed out.
 *
 * Mimics the `wavinfo` program:
 *   http://www.morphet.org.uk/comp/wavtools.html
 */

var fs = require('fs');
var wav = require('../');
var filename = process.argv[2];

var input;
var reader = new wav.Reader();

if (filename) {
  console.log('Header info for: %s', filename);
  input = fs.createReadStream(filename);
} else {
  console.log('Header info for: STDIN');
  input = process.stdin;
}

input.pipe(reader);

// the "readable" event will always come *after* the "format" event, but by now
// a few final properties will have been parsed like "subchunk2Size" that we want
// to print out to simulate the wavinfo(1) command
reader.once('readable', function () {
  console.log('WaveHeader Size:\t%d', 12);
  console.log('ChunkHeader Size:\t%d', 8);
  console.log('FormatChunk Size:\t%d', reader.subchunk1Size);
  console.log('RIFF ID:\t%s', reader.riffId);
  console.log('Total Size:\t%d', reader.chunkSize);
  console.log('Wave ID:\t%s', reader.waveId);
  console.log('Chunk ID:\t%s', reader.chunkId);
  console.log('Chunk Size:\t%d', reader.subchunk1Size);
  console.log('Compression format is of type: %d', reader.audioFormat);
  console.log('Channels:\t%d', reader.channels);
  console.log('Sample Rate:\t%d', reader.sampleRate);
  console.log('Bytes / Sec:\t%d', reader.byteRate);
  console.log('wBlockAlign:\t%d', reader.blockAlign);
  console.log('Bits Per Sample Point:\t%d', reader.bitDepth);
  // TODO: this should end up being "44" or whatever the total length of the WAV
  //       header is. maybe emit "format" at this point rather than earlier???
  console.log('wavDataPtr: %d', 0);
  console.log('wavDataSize: %d', reader.subchunk2Size);
  console.log();
});
