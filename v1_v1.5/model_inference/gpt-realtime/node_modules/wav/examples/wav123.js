
/**
 * Plays the WAVE audio file from stdin out of the computer's speakers
 * via `node-speaker`.
 */

var Reader = require('../').Reader;
var Speaker = require('speaker');

var reader = new Reader();

reader.on('format', function (format) {
  console.error('format:', format);
  var s = new Speaker(format);
  reader.pipe(s);
});

reader.on('error', function (err) {
  console.error('Reader error: %s', err);
});

process.stdin.pipe(reader);
