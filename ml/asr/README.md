# Converse Streaming ASR Engine

Automatic Speech Recognition (ASR) subsystem for the Converse real-time bidirectional ASL <-> English communication engine.

---

## 1. Pipeline Architecture

```text
Audio (WAV / WebM / Opus / PCM)
  │
  ▼
Audio decoder/resampler (16kHz mono float32)
  │
  ▼
Silero VAD (Utterance boundary & speech activity detection)
  │
  ▼
ASR abstraction (AsrEngineProtocol)
  │
  ▼
Amazon Transcribe Streaming (HTTP/2 signed 16-bit PCM)
  │
  ▼
ASREvent (Canonical @converse/contracts schema)
  │
  ▼
English normalization
  │
  ▼
ASL-oriented grammar compiler (Topic-Comment / TSOV syntax)
  │
  ▼
Lexicon lookup / OOV fingerspelling decomposition
  │
  ▼
Non-Manual Markers (NMM facial blendshapes & head motion)
  │
  ▼
Spatial loci tracking (3D coordinate anchors)
  │
  ▼
Timing computation (Monotonic lead-in, hold, lead-out)
  │
  ▼
SignRepresentation (Targeted avatar animation tokens)
```

The downstream speech-to-sign translation pipeline remains provider-independent, operating exclusively on canonical `AsrTranscriptEvent` contracts.

---

## 2. ASR Engine Architecture

```text
                ASREngine interface (AsrEngineProtocol)
                                │
          ┌─────────────────────┴─────────────────────┐
          ↓                                           ↓
 Amazon Transcribe Streaming                     Faster-Whisper
   Production Default                              Local / Dev
   (ASR_BACKEND=aws)                           (ASR_BACKEND=whisper)
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ↓
                        AsrTranscriptEvent
                                ↓
                     Existing Translation
                                ↓
                        SignRepresentation
```

---

## 3. AWS Transcribe Setup

### 3.1 Prerequisites

To run the production ASR backend with Amazon Transcribe Streaming:

1. **AWS Account**: Active AWS account with Amazon Transcribe service enabled.
2. **IAM Permissions**: Identity-based policy granting:
   * `transcribe:StartStreamTranscription`
3. **AWS CLI**: Installed and configured on the developer or host machine.
4. **Credential Chain**: Standard AWS credential provider chain (`~/.aws/credentials`, `~/.aws/config`, environment variables, or IAM role).

### 3.2 Configuration

Configure the AWS CLI:

```bash
aws configure
```

Set the environment variables for Converse:

```bash
export ASR_BACKEND=aws
export AWS_REGION=ap-south-1
export AWS_TRANSCRIBE_LANGUAGE=en-IN
```

*Never hard-code, commit, or print AWS access keys or secrets.*

### 3.3 Local Offline Development (Faster-Whisper)

For offline development, local CI, or environments without AWS credentials, select Faster-Whisper explicitly:

```bash
export ASR_BACKEND=whisper
```

### 3.4 Strict Failure Policy (Zero Silent Fallback)

In production (`ASR_BACKEND=aws`), the engine strictly does **NOT** silently fall back to Whisper, mock transcripts, or dummy text upon AWS failure.

```text
AWS failure / Missing credentials
         │
         ▼
Explicit typed Domain Error (AUTHENTICATION_FAILED / SERVICE_UNAVAILABLE)
         │
         ▼
Caller handles error explicitly
```

If AWS credentials are missing or invalid, an explicit `AUTHENTICATION_FAILED` error is returned. If Amazon Transcribe is unreachable, an explicit `SERVICE_UNAVAILABLE` error is returned.

---

## 4. Audio Input Specification

Amazon Transcribe Streaming receives:

* **Format**: RAW signed 16-bit linear PCM (little-endian, `<i2`)
* **Channels**: Mono (1 channel)
* **Sample Rate**: 16,000 Hz
* **Containers**: No WAV/RIFF headers sent to AWS stream

Incoming payloads in WAV, WebM, Ogg, or Opus format are decoded and normalized via `decode_audio_payload` before raw PCM stream transmission.

---

## 5. Testing and Verification

### 5.1 Unit and Contract Tests

```bash
uv run pytest
```

Runs mocked unit tests covering client creation, PCM byte packing, chunk forwarding, partial/final transcript mapping, timestamp alignment without fabrication, explicit flush, stream cleanup, and error handling.

### 5.2 Live AWS Integration Smoke Test

The live smoke test connects to AWS Transcribe Streaming in `ap-south-1` using real environment credentials. It is gated behind an environment variable to prevent unintentional charges during CI:

```bash
RUN_AWS_TRANSCRIBE_TESTS=1 uv run pytest tests/test_live_aws.py -rs
```

### 5.3 Comparative Latency Benchmark

Measures separate latencies for Silero VAD, Faster-Whisper, and Amazon Transcribe Streaming:

```bash
uv run python scripts/benchmark_asr.py
```

### 5.4 Word Error Rate (WER) Evaluation

Evaluates predicted transcripts against labeled ground-truth speech:

```bash
uv run python scripts/evaluate_asr.py --backend both
```

---

## 6. Cost and Resource Notes

Amazon Transcribe is a metered cloud service billed based on the seconds of streaming audio processed per month.

* Tiered pricing applies based on monthly volume.
* Always review the official AWS Transcribe pricing documentation for up-to-date regional rates:
  [Amazon Transcribe Pricing](https://aws.amazon.com/transcribe/pricing/)
