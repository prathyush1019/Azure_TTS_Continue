# server.py
# FastAPI server exposing Azure TTS via WebSocket streaming, REST endpoints, and benchmarks

import os
import io
import json
import time
import wave
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tts_service import AzureStreamer
from latency_tracker import LatencyTracker
from config import SAMPLE_RATE, AZURE_SUBSCRIPTION_KEY, AZURE_REGION, VOICE_NAME

app = FastAPI(title="Azure TTS Streaming")

# CORS – allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/api/config")
async def get_config():
    return {
        "subscription_key": AZURE_SUBSCRIPTION_KEY,
        "region": AZURE_REGION,
        "voice_name": VOICE_NAME,
        "sample_rate": SAMPLE_RATE,
    }


class SynthesizeRequest(BaseModel):
    text: str


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """REST fallback – returns a complete WAV file (non-streaming)."""
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Text is required"}, status_code=400)

    tracker = LatencyTracker(phrase_text=text)

    try:
        def _run():
            tracker.mark("text_send_start")
            with AzureStreamer() as tts:
                audio_bytes = tts.synthesize(text)
            tracker.mark("first_chunk_received")
            tracker.mark("First Audio Chunk")
            return audio_bytes

        audio_bytes = await asyncio.to_thread(_run)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if not audio_bytes:
        return JSONResponse({"error": "No audio received from Azure TTS"}, status_code=500)

    # Build WAV from raw PCM (24 kHz, 16-bit mono)
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)
    wav_io.seek(0)

    latency_ms = tracker.get_metrics()["speech_generation_latency_ms"] or 0.0

    return Response(
        content=wav_io.read(),
        media_type="audio/wav",
        headers={
            "X-First-Chunk-Latency": f"{latency_ms:.2f}",
            "Access-Control-Expose-Headers": "X-First-Chunk-Latency",
        },
    )


# ── WebSocket Streaming Endpoint ──────────────────────────────────────────────

@app.websocket("/ws/synthesize")
async def ws_synthesize(ws: WebSocket):
    """Stream Azure TTS audio chunks to the client over WebSocket.

    Uses an asyncio.Queue as a thread→async bridge so that each PCM chunk
    is sent to the client the *instant* Azure delivers it — true real-time
    streaming, not buffered.

    Protocol:
    1. Client sends a JSON text frame: {"text": "Hello world"}
    2. Server streams binary frames (raw PCM 24 kHz 16-bit mono) as they arrive
    3. Server sends a final JSON text frame: {"done": true, "latency_ms": 123.45}
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            # Wait for client to send text
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            text = msg.get("text", "").strip()
            if not text:
                await ws.send_text(json.dumps({"error": "Text is required"}))
                continue

            # asyncio.Queue bridges the blocking producer thread → async consumer
            chunk_queue: asyncio.Queue = asyncio.Queue()
            tracker = LatencyTracker(phrase_text=text)
            first_chunk_received = False

            def _produce():
                """Run in a background thread: synthesize and push each chunk
                into the async queue the moment Azure delivers it."""
                nonlocal first_chunk_received
                try:
                    with AzureStreamer() as tts:
                        tracker.mark("text_send_start")
                        for chunk in tts.stream_audio(text):
                            if not first_chunk_received:
                                tracker.mark("first_chunk_received")
                                tracker.mark("First Audio Chunk")
                                first_chunk_received = True
                            # Thread-safe push into the async queue
                            asyncio.run_coroutine_threadsafe(
                                chunk_queue.put(chunk), loop
                            ).result()   # block until enqueued
                except Exception as e:
                    # Push error as a dict so the consumer can distinguish it
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put({"error": str(e)}), loop
                    ).result()
                finally:
                    # Sentinel: signals the consumer to stop
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(None), loop
                    ).result()

            # Launch the blocking producer in a daemon thread
            producer = threading.Thread(target=_produce, daemon=True)
            producer.start()

            # Consume chunks and send over WebSocket as they arrive
            error_msg = None
            chunks_sent = 0
            try:
                while True:
                    item = await chunk_queue.get()
                    if item is None:
                        # Producer finished
                        break
                    if isinstance(item, dict) and "error" in item:
                        error_msg = item["error"]
                        break
                    # Send raw PCM binary frame immediately
                    await ws.send_bytes(item)
                    chunks_sent += 1
            except WebSocketDisconnect:
                raise

            producer.join(timeout=5)

            if error_msg:
                await ws.send_text(json.dumps({"error": error_msg}))
            else:
                latency_ms = tracker.get_metrics()["speech_generation_latency_ms"] or 0.0
                await ws.send_text(json.dumps({
                    "done": True,
                    "latency_ms": round(latency_ms, 2),
                    "chunks_sent": chunks_sent,
                }))

    except WebSocketDisconnect:
        pass


# ── Benchmark Endpoints ───────────────────────────────────────────────────────

@app.get("/api/benchmark")
async def benchmark():
    """REST benchmark – returns full results after all tests complete."""
    input_file = "input.json"
    if not os.path.exists(input_file):
        return JSONResponse({"error": "input.json not found"}, status_code=404)

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        return JSONResponse({"error": f"Error reading JSON: {str(e)}"}, status_code=500)

    test_cases = data.get("test_cases", {})
    if not test_cases:
        return JSONResponse({"error": "No test cases found in input.json"}, status_code=400)

    def _run_benchmark():
        results = {}
        with AzureStreamer() as tts:
            groups = sorted(
                test_cases.keys(),
                key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 999,
            )
            for group_name in groups:
                phrases = test_cases[group_name]
                latencies = []
                for phrase in phrases:
                    tracker = LatencyTracker(phrase_text=phrase)
                    tracker.mark("Start")
                    tracker.mark("text_send_start")
                    audio_bytes = tts.synthesize(phrase)
                    tracker.mark("first_chunk_received")
                    tracker.mark("First Audio Chunk")
                    latency_ms = tracker.get_metrics()["speech_generation_latency_ms"]
                    if latency_ms is not None:
                        latencies.append(latency_ms)
                    time.sleep(0.05)

                if latencies:
                    results[group_name] = {
                        "avg": round(sum(latencies) / len(latencies), 2),
                        "min": round(min(latencies), 2),
                        "max": round(max(latencies), 2),
                    }
                else:
                    results[group_name] = None
        return results

    try:
        results = await asyncio.to_thread(_run_benchmark)
    except Exception as e:
        return JSONResponse({"error": f"Benchmarking failed: {str(e)}"}, status_code=500)

    return {"status": "success", "results": results}


@app.websocket("/ws/benchmark")
async def ws_benchmark(ws: WebSocket):
    """Stream benchmark results live over WebSocket.

    Protocol:
    1. Client sends: {"action": "start"}
    2. Server streams JSON text frames:
       - {"type": "info", "total_groups": N, "total_phrases": M}
       - {"type": "group_start", "group": "1_word", "group_index": 0, "phrase_count": 10}
       - {"type": "phrase_result", "group": "1_word", "phrase": "Hello", "latency_ms": 123.4, "phrase_index": 0, "status": "success"}
       - {"type": "group_done", "group": "1_word", "avg_ms": ..., "min_ms": ..., "max_ms": ...}
       - {"type": "done", "summary": {...}}
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    try:
        raw = await ws.receive_text()
        msg = json.loads(raw)
        if msg.get("action") != "start":
            await ws.send_text(json.dumps({"error": "Send {\"action\": \"start\"} to begin"}))
            return

        input_file = "input.json"
        if not os.path.exists(input_file):
            await ws.send_text(json.dumps({"error": "input.json not found"}))
            return

        with open(input_file, "r") as f:
            data = json.load(f)
        test_cases = data.get("test_cases", {})
        if not test_cases:
            await ws.send_text(json.dumps({"error": "No test cases in input.json"}))
            return

        groups = sorted(
            test_cases.keys(),
            key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 999,
        )
        total_phrases = sum(len(test_cases[g]) for g in groups)

        # Send overview info
        await ws.send_text(json.dumps({
            "type": "info",
            "total_groups": len(groups),
            "total_phrases": total_phrases,
            "groups": groups,
        }))

        # Use asyncio.Queue to stream results from blocking thread
        result_queue: asyncio.Queue = asyncio.Queue()

        def _run():
            try:
                summary = {}
                with AzureStreamer() as tts:
                    for gi, group_name in enumerate(groups):
                        phrases = test_cases[group_name]
                        asyncio.run_coroutine_threadsafe(
                            result_queue.put({
                                "type": "group_start",
                                "group": group_name,
                                "group_index": gi,
                                "phrase_count": len(phrases),
                            }), loop
                        ).result()

                        latencies = []
                        for pi, phrase in enumerate(phrases):
                            tracker = LatencyTracker(phrase_text=phrase)
                            tracker.mark("Start")
                            tracker.mark("text_send_start")
                            try:
                                tts.synthesize(phrase)
                                tracker.mark("first_chunk_received")
                                tracker.mark("First Audio Chunk")
                                latency_ms = tracker.get_metrics()["speech_generation_latency_ms"]
                                if latency_ms is not None:
                                    latencies.append(latency_ms)
                                asyncio.run_coroutine_threadsafe(
                                    result_queue.put({
                                        "type": "phrase_result",
                                        "group": group_name,
                                        "phrase": phrase,
                                        "latency_ms": round(latency_ms, 2) if latency_ms else None,
                                        "phrase_index": pi,
                                        "status": "success",
                                    }), loop
                                ).result()
                            except Exception as e:
                                asyncio.run_coroutine_threadsafe(
                                    result_queue.put({
                                        "type": "phrase_result",
                                        "group": group_name,
                                        "phrase": phrase,
                                        "latency_ms": None,
                                        "phrase_index": pi,
                                        "status": "error",
                                        "error": str(e),
                                    }), loop
                                ).result()
                            time.sleep(0.05)

                        group_summary = {
                            "type": "group_done",
                            "group": group_name,
                        }
                        if latencies:
                            group_summary.update({
                                "avg_ms": round(sum(latencies) / len(latencies), 2),
                                "min_ms": round(min(latencies), 2),
                                "max_ms": round(max(latencies), 2),
                                "count": len(latencies),
                            })
                            summary[group_name] = {
                                "avg_ms": group_summary["avg_ms"],
                                "min_ms": group_summary["min_ms"],
                                "max_ms": group_summary["max_ms"],
                            }
                        else:
                            group_summary.update({"avg_ms": None, "min_ms": None, "max_ms": None, "count": 0})
                            summary[group_name] = None
                        asyncio.run_coroutine_threadsafe(
                            result_queue.put(group_summary), loop
                        ).result()

                # Final done message
                asyncio.run_coroutine_threadsafe(
                    result_queue.put({"type": "done", "summary": summary}), loop
                ).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    result_queue.put({"type": "error", "error": str(e)}), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    result_queue.put(None), loop
                ).result()

        producer = threading.Thread(target=_run, daemon=True)
        producer.start()

        # Stream results to client as they arrive
        while True:
            item = await result_queue.get()
            if item is None:
                break
            await ws.send_text(json.dumps(item))

        producer.join(timeout=60)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting FastAPI Azure TTS app on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
