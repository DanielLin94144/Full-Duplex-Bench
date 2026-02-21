node-wav
========
### `Reader` and `Writer` streams for Microsoft WAVE audio files
[![Build Status](https://secure.travis-ci.org/TooTallNate/node-wav.svg)](http://travis-ci.org/TooTallNate/node-wav)

This module offers streams to help work with Microsoft WAVE files.


Installation
------------

Install through npm:

``` bash
$ npm install wav
```


Example
-------

Here's how you would play a standard PCM WAVE file out of the speakers using
`node-wav` and `node-speaker`:

``` javascript
var fs = require('fs');
var wav = require('wav');
var Speaker = require('speaker');

var file = fs.createReadStream('track01.wav');
var reader = new wav.Reader();

// the "format" event gets emitted at the end of the WAVE header
reader.on('format', function (format) {

  // the WAVE header is stripped from the output of the reader
  reader.pipe(new Speaker(format));
});

// pipe the WAVE file to the Reader instance
file.pipe(reader);
```


API
---

  - [Reader()](#reader)
  - [Writer()](#writeroptions)
  - [FileWriter()](#filewriterpath-options)

### Reader()

The `Reader` class accepts a WAV audio file written to it and outputs the raw
audio data with the WAV header stripped (most of the time, PCM audio data will
be output, depending on the `audioFormat` property).

A `"format"` event gets emitted after the WAV header has been parsed.

### Writer(options)

The `Writer` class accepts raw audio data written to it (only PCM audio data is
currently supported), and outputs a WAV file with a valid WAVE header at the
beginning specifying the formatting information of the audio stream.

Note that there's an interesting problem, because the WAVE header also
specifies the total byte length of the audio data in the file, and there's no
way that we can know this ahead of time. Therefore the WAVE header will contain
a byte-length if `0` initially, which most WAVE decoders will know means to
just read until `EOF`.

Optionally, if you are in a situation where you can seek back to the beginning
of the destination of the WAVE file (like writing to a regular file, for
example), then you may listen for the `"header"` event which will be emitted
_after_ all the data has been written, and you can go back and rewrite the new
header with proper audio byte length into the beginning of the destination
(though if your destination _is_ a regular file, you should use the the
`FileWriter` class instead).

Default options: 

```JSON
{
  "channels": 2,
  "sampleRate": 44100,
  "bitDepth": 16
}
```  


### FileWriter(path, options)

The `FileWriter` class is, essentially, a combination of `fs.createWriteStream()` and the above `Writer()` class, except it automatically corrects the header after the file is written. Options are passed to both `Writer()` and `fs.createWriteStream()`.

Example usage with `mic`:

```js
var FileWriter = require('wav').FileWriter;
var mic = require('mic'); // requires arecord or sox, see https://www.npmjs.com/package/mic

var micInstance = mic({
  rate: '16000',
  channels: '1',
  debug: true
});

var micInputStream = micInstance.getAudioStream();

var outputFileStream = new FileWriter('./test.wav', {
  sampleRate: 16000,
  channels: 1
});

micInputStream.pipe(outputFileStream);

micInstance.start();

setTimeout(function() {
  micInstance.stop();
}, 5000);
```
