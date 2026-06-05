# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Azure Speech service configuration – values should be set in a .env file
AZURE_SUBSCRIPTION_KEY = os.getenv("AZURE_SUBSCRIPTION_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "centralindia")
# Voice name as defined in Azure – e.g., "en-US-AriaNeural"
VOICE_NAME = os.getenv("AZURE_VOICE_NAME", "en-US-AriaNeural")

# Sample rate (Hz) – match Azure default (24k) or override as needed
SAMPLE_RATE = 24000
