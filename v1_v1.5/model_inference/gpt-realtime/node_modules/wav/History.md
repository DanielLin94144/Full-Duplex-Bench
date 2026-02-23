
1.0.0 / 2015-05-01
==================

  * add MIT license file
  * add appveyor.yml file for Windows testing
  * examples: fix comment
  * index: add link to RFC2361
  * reader: add clarifying comment
  * reader: add initial `float` WAV file support
  * reader: add a few more formats defined by the RFC
  * reader: add `formats` map and set `float`, `alaw` and `ulaw` on the "format" object
  * reader: use %o debug v1 formatters
  * reader, writer: always use "readable-stream" copy of Transform
  * package: remove "engines" field
  * package: update all dependency versions
  * README: use svg for Travis badge
  * travis: don't test node v0.7 and v0.9, test v0.11

0.1.2 / 2014-01-11
==================

  * package: update `readable-stream` dep to v1.1.10
  * travis: test node v0.10 and v0.11
  * Writer: bypassed `stream-parser` to avoid assertion error (#1, #5)

0.1.1 / 2013-12-12
==================

  * Fix package.json repository URL so npm link isn't broken (@cbebry)

0.1.0 / 2013-03-07
==================

  * reader: passthrough the audio data chunk until EOF
  * test: begin testing with Travis-ci
  * add experimental RIFX support
  * reader, writer: integrate the "stream-parser" mixin
  * test: add initial Reader tests

0.0.1 / 2012-02-05
==================

  * Initial release
