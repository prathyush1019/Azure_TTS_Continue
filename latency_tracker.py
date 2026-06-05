# latency_tracker.py
import time
import os
import json

class LatencyTracker:
    def __init__(self, phrase_id=None, phrase_text=None):
        self.phrase_id = phrase_id
        self.phrase_text = phrase_text
        self.start_time = time.perf_counter()
        self.events = {}

    def mark(self, name):
        self.events[name] = time.perf_counter()

    def get_speech_generation_latency_ms(self):
        if "flush_sent" in self.events and "first_chunk_received" in self.events:
            return (self.events["first_chunk_received"] - self.events["flush_sent"]) * 1000
        if "Start" in self.events and "First Audio Chunk" in self.events:
            return (self.events["First Audio Chunk"] - self.events["Start"]) * 1000
        return None

    def get_input_submission_time_ms(self):
        if "text_send_start" in self.events and "flush_sent" in self.events:
            return (self.events["flush_sent"] - self.events["text_send_start"]) * 1000
        return None

    def get_metrics(self):
        latency = self.get_speech_generation_latency_ms()
        input_time = self.get_input_submission_time_ms()
        metrics = {
            "phrase_id": self.phrase_id,
            "phrase_text": self.phrase_text,
            "input_submission_time_ms": round(input_time, 2) if input_time is not None else None,
            "speech_generation_latency_ms": round(latency, 2) if latency is not None else None
        }
        metrics["durations_ms"] = {"time_to_first_byte_ttfb": latency}
        return metrics

    def duration(self, event):
        if event not in self.events:
            return None
        return (self.events[event] - self.start_time) * 1000

    def report(self):
        print("\n====================")
        print("LATENCY REPORT")
        print("====================")
        for event in self.events:
            dur = self.duration(event)
            if dur is not None:
                print(f"{event:<25}{dur:.2f} ms")
        print("====================")
