# OpenAI GPT-Realtime Console

This is an example application showing how to use the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) with [WebRTC](https://platform.openai.com/docs/guides/realtime-webrtc).

## Installation and usage

Before you begin, you'll need an OpenAI API key. You can specify it in the file `cli.js` by ```const apiKey = "YOUR_OPENAI_API_KEY";```

Running this application locally requires [Node.js](https://nodejs.org/) to be installed. Install dependencies for the application with:

```bash
npm install
```

## Run the inference
The main inference script is `inference.sh`, which uses the `cli.js` script to process audio files.
To run the inference, you can use the following command:

```bash
bash inference.sh {/path/to/dataset} {MAX_JOBS}
```
You should replace `{path/to/dataset}` with the path to your dataset directory and `{MAX_JOBS}` with the maximum number of parallel jobs you want to run (default is 4).
