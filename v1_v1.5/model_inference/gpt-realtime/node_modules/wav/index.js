
/**
 * References:
 *  - http://tools.ietf.org/html/rfc2361
 *  - http://www.sonicspot.com/guide/wavefiles.html
 *  - https://ccrma.stanford.edu/courses/422/projects/WaveFormat/
 *  - http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
 *  - http://www.blitter.com/~russtopia/MIDI/~jglatt/tech/wave.htm
 */

/**
 * The `Reader` class accepts a WAVE audio file, emits a "format" event, and
 * outputs the raw "data" from the WAVE file (usually raw PCM data, but if the
 * WAVE file uses compression then the compressed data will be output, you are
 * responsible for uncompressing in that case if necessary).
 */

exports.Reader = require('./lib/reader');

/**
 * The `Writer` class outputs a valid WAVE file from the audio data written to
 * it. You may set any of the "channels", "sampleRate" or "bitsPerSample"
 * properties before writing the first chunk. You may also set the "dataLength" to
 * the number of bytes expected in the "data" portion of the WAVE file. If
 * "dataLength" is not set, then the maximum valid length for a WAVE file is
 * written.
 */

exports.Writer = require('./lib/writer');

/**
 * The `FileWriter` is a subclass of `Writer` that automatically takes care of
 * writing the "header" event at the end of the stream to the beginning of the
 * output file.
 */

exports.FileWriter = require('./lib/file-writer');
