/* eslint-env mocha */

/**
 * Module dependencies.
 */

var fs = require('fs');
var path = require('path');
var assert = require('assert');
var Reader = require('../').Reader;

describe('Reader', function () {
  describe('RIFF - Little-endian', function () {
    describe('1up.wav', function () {
      var fixture = path.resolve(__dirname, 'fixtures', '1up.wav');

      it('should emit a "format" event', function (done) {
        var reader = new Reader();
        reader.on('format', function (format) {
          assert.equal(1, format.audioFormat);
          assert.equal(1, format.channels);
          assert.equal(11025, format.sampleRate);
          assert.equal(8, format.bitDepth);
          assert.equal(false, format.signed);
          done();
        });
        fs.createReadStream(fixture).pipe(reader).resume();
      });

      it('should emit an "end" event', function (done) {
        var reader = new Reader();
        reader.on('end', done);
        fs.createReadStream(fixture).pipe(reader).resume();
      });
    });

    describe('gameover.wav', function () {
      var fixture = path.resolve(__dirname, 'fixtures', 'gameover.wav');

      it('should emit a "format" event', function (done) {
        var reader = new Reader();
        reader.on('format', function (format) {
          // wave format
          assert.equal('RIFF', this.riffId);
          assert.equal(1, format.audioFormat);

          // pcm format
          assert.equal('LE', format.endianness);
          assert.equal(1, format.channels);
          assert.equal(22050, format.sampleRate);
          assert.equal(16, format.bitDepth);
          assert.equal(true, format.signed);
          done();
        });
        fs.createReadStream(fixture).pipe(reader).resume();
      });

      it('should emit an "end" event', function (done) {
        var reader = new Reader();
        reader.on('end', done);
        fs.createReadStream(fixture).pipe(reader).resume();
      });
    });

    describe('M1F1-float32-AFsp.wav', function () {
      var fixture = path.resolve(__dirname, 'fixtures', 'M1F1-float32-AFsp.wav');

      it('should emit a "format" event', function (done) {
        var reader = new Reader();
        reader.on('format', function (format) {
          assert.equal(3, format.audioFormat);
          assert.equal(2, format.channels);
          assert.equal(8000, format.sampleRate);
          assert.equal(32, format.bitDepth);
          assert.equal(true, format.signed);
          assert.equal(true, format.float);
          done();
        });
        fs.createReadStream(fixture).pipe(reader).resume();
      });

      it('should emit an "end" event', function (done) {
        var reader = new Reader();
        reader.on('end', done);
        fs.createReadStream(fixture).pipe(reader).resume();
      });
    });

    describe('M1F1-float64-AFsp.wav', function () {
      var fixture = path.resolve(__dirname, 'fixtures', 'M1F1-float64-AFsp.wav');

      it('should emit a "format" event', function (done) {
        var reader = new Reader();
        reader.on('format', function (format) {
          assert.equal(3, format.audioFormat);
          assert.equal(2, format.channels);
          assert.equal(8000, format.sampleRate);
          assert.equal(64, format.bitDepth);
          assert.equal(true, format.signed);
          assert.equal(true, format.float);
          done();
        });
        fs.createReadStream(fixture).pipe(reader).resume();
      });

      it('should emit an "end" event', function (done) {
        var reader = new Reader();
        reader.on('end', done);
        fs.createReadStream(fixture).pipe(reader).resume();
      });
    });
  });

  describe('RIFX - Big-endian', function () {
    describe('gameover-rifx.wav', function () {
      var fixture = path.resolve(__dirname, 'fixtures', 'gameover-rifx.wav');

      it('should emit a "format" event', function (done) {
        var reader = new Reader();
        reader.on('format', function (format) {
          // wave format
          assert.equal('RIFX', this.riffId);
          assert.equal(1, format.audioFormat);

          // pcm format
          assert.equal('BE', format.endianness);
          assert.equal(1, format.channels);
          assert.equal(22050, format.sampleRate);
          assert.equal(16, format.bitDepth);
          assert.equal(true, format.signed);
          done();
        });
        fs.createReadStream(fixture).pipe(reader).resume();
      });

      it('should emit an "end" event', function (done) {
        var reader = new Reader();
        reader.on('end', done);
        fs.createReadStream(fixture).pipe(reader).resume();
      });
    });
  });
});
