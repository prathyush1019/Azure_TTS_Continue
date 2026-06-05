# tts_service.py
# Azure Text-to-Speech streaming wrapper

import os
import queue
import threading
import azure.cognitiveservices.speech as speechsdk
from config import AZURE_SUBSCRIPTION_KEY, AZURE_REGION, VOICE_NAME, SAMPLE_RATE


class AzureStreamer:
    """Wrap Azure Speech SDK to provide a Cardesia‑like streaming interface.
    The implementation uses a PullAudioOutputStream so that audio data can be
    read incrementally with ``receive_audio``.
    """

    def __init__(self):
        # Configure speech service
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SUBSCRIPTION_KEY,
            region=AZURE_REGION,
        )
        speech_config.speech_synthesis_voice_name = VOICE_NAME
        # Request WAV (RIFF) output matching SAMPLE_RATE
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )
        # Create a pull stream that we can read from
        self.pull_stream = speechsdk.audio.PullAudioOutputStream()
        self.audio_config = speechsdk.audio.AudioOutputConfig(stream=self.pull_stream)
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config, self.audio_config)
        self._started = False
        # Track when the stream should stop yielding data
        self._finished = False

    def synthesize(self, text: str) -> bytes:
        """Synchronously synthesize *text* and return raw PCM audio bytes.
        This uses the SpeechSynthesizer's ``speak_text`` method which returns a result
        containing the complete audio data in ``audio_data``.
        """
        result = self.synthesizer.speak_text(text)
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")
        return result.audio_data

    def stream_audio(self, text: str, chunk_size: int = 4096):
        """Yield PCM audio chunks as they are synthesized by Azure.

        Uses the ``synthesizing`` event callback so chunks are emitted the
        instant Azure delivers them — no pull-stream polling needed.
        """
        audio_queue: queue.Queue = queue.Queue()
        error_holder: list = []

        def _on_synthesizing(evt):
            """Callback fired each time a chunk of audio is ready."""
            if evt.result.audio_data:
                audio_queue.put(evt.result.audio_data)

        def _on_completed(evt):
            """Callback fired when synthesis finishes."""
            audio_queue.put(None)  # sentinel

        def _on_canceled(evt):
            """Callback fired on cancellation / error."""
            details = evt.result.cancellation_details
            error_holder.append(
                RuntimeError(f"Azure TTS canceled: {details.reason} – {details.error_details}")
            )
            audio_queue.put(None)  # sentinel

        # Wire up event handlers
        self.synthesizer.synthesizing.connect(_on_synthesizing)
        self.synthesizer.synthesis_completed.connect(_on_completed)
        self.synthesizer.synthesis_canceled.connect(_on_canceled)

        try:
            # Kick off synthesis asynchronously
            self.synthesizer.speak_text_async(text)

            # Yield chunks as they arrive
            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    break
                yield chunk

            # If there was an error, raise it after draining the queue
            if error_holder:
                raise error_holder[0]
        finally:
            # Disconnect event handlers to allow re-use
            self.synthesizer.synthesizing.disconnect_all()
            self.synthesizer.synthesis_completed.disconnect_all()
            self.synthesizer.synthesis_canceled.disconnect_all()

    def __enter__(self):
        # Prepare streamer for use
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources
        self.synthesizer = None
        self.pull_stream = None
        return False

    # The original Cartesia API has an explicit start_stream – we keep it for parity
    def start_stream(self):
        self._started = True
        # No extra action required for Azure; the stream is ready.

    def push_text(self, text: str):
        if not self._started:
            raise RuntimeError("Stream not started – call start_stream() first")
        # Perform synthesis synchronously; result audio lands in the pull stream.
        result = self.synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")

    def finish(self):
        # Azure SDK does not require a separate finish call; we simply close the stream.
        # Mark the stream as completed so ``receive_audio`` can stop.
        self._finished = True

    def receive_audio(self, chunk_size: int = 1024):
        """Yield PCM chunks from the Azure pull stream.
        The stream returns ``bytes`` objects. It continues reading until the stream is marked finished and no more data is available.
        """
        while True:
            data = self.pull_stream.read(chunk_size)
            if data:
                yield data
                continue
            # No data returned; check if the stream has been marked finished
            if self._finished:
                break
            # If not finished, wait briefly and retry
            import time
            time.sleep(0.01)
