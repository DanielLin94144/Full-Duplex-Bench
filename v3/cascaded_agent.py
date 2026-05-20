#!/usr/bin/env python3
"""
Cascaded Voice Agent: Silero VAD + OpenAI Whisper STT + OpenAI gpt-4o LLM + OpenAI TTS

A LiveKit voice agent that uses a cascaded pipeline instead of native realtime models.
This agent reuses the same AssistantFnc tools as lk_agent_tool.py so it can be
evaluated with the same benchmark pipeline.

Pipeline:
  User Audio → Silero VAD → OpenAI Whisper STT → OpenAI gpt-4o (with tool calling) → OpenAI TTS → Agent Audio

Usage:
    # Development mode:
    python cascaded_agent.py dev

    # Console mode (local mic/speaker):
    python cascaded_agent.py console

Environment variables (in .env.local):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    OPENAI_API_KEY     – OpenAI API key (for STT + LLM + TTS)
"""

import os
import sys
import json
import logging
import time
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, AgentServer, llm

# Compatibility for different livekit-agents versions
if hasattr(llm, "function_tool"):
    ai_callable_decorator = llm.function_tool
else:
    ai_callable_decorator = llm.ai_callable

# Parse custom CLI args before LiveKit CLI
LATENCY_PROFILE = "instant"
if "--latency" in sys.argv:
    idx = sys.argv.index("--latency")
    if idx + 1 < len(sys.argv):
        LATENCY_PROFILE = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

# Import mock APIs for tool execution
try:
    from mock_apis import MockAPIRegistry
    registry = MockAPIRegistry(latency_profile=LATENCY_PROFILE)
    print(f"🔧 API Backend running with '{LATENCY_PROFILE}' latency profile.")
except ImportError:
    logging.warning("mock_apis.py not found. Tools will be mocked or fail.")
    registry = None

# Load environment
env_path = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(env_path)


# ---------------------------------------------------------------------------
# Latency Tracker (same as lk_agent_tool.py)
# ---------------------------------------------------------------------------
class LatencyTracker:
    def __init__(self):
        self.user_done_at = 0
        self.tool_start_at = 0
        self.tool_end_at = 0
        self.agent_start_at = 0
        self.query_received = False

    def reset(self):
        self.__init__()

    def log_breakdown(self, tool_name="", room_name="unknown"):
        if not self.user_done_at or not self.agent_start_at or not self.tool_start_at:
            return
        reasoning = (self.tool_start_at - self.user_done_at) if self.tool_start_at else 0
        execution = (self.tool_end_at - self.tool_start_at) if self.tool_start_at and self.tool_end_at else 0
        synthesis = (self.agent_start_at - (self.tool_end_at or self.user_done_at))
        total = self.agent_start_at - self.user_done_at

        report = f"\n⏱️ LATENCY BREAKDOWN ({tool_name}) for room {room_name}:\n"
        report += f"  - Reasoning (Model -> Tool): {reasoning:.2f}s\n"
        if execution:
            report += f"  - Tool Execution (API):    {execution:.2f}s\n"
        report += f"  - Synthesis (Tool -> Spoken): {synthesis:.2f}s\n"
        report += f"  - TOTAL SEARCH LATENCY:      {total:.2f}s\n"

        metrics = {
            "room": room_name,
            "tool": tool_name,
            "reasoning": round(reasoning, 3),
            "execution": round(execution, 3),
            "synthesis": round(synthesis, 3),
            "total": round(total, 3),
            "agent_start_at": self.agent_start_at
        }
        json_report = f"LATENCY_TRACK_JSON: {json.dumps(metrics)}"

        logging.info(report)
        logging.info(json_report)
        print(report)
        with open("/tmp/agent_heartbeat.log", "a") as f:
            f.write(report + "\n")
            f.write(json_report + "\n")


# ---------------------------------------------------------------------------
# Tool/Function definitions (same as lk_agent_tool.py)
# ---------------------------------------------------------------------------
class AssistantFnc:
    def __init__(self, tracker: LatencyTracker, room_name: str):
        self.room_name = room_name
        self.tracker = tracker

    def log_tool_call(self, func_name: str, args: dict, t_start: float, t_end: float):
        with open("/tmp/agent_tool_calls.log", "a") as f:
            f.write(json.dumps({
                "room": self.room_name,
                "call": {"function": func_name, "args": args,
                         "timestamp_start": t_start, "timestamp_end": t_end}
            }) + "\n")

    # ── Travel & Identity ─────────────────────────────────────────
    @ai_callable_decorator(description="Search for available flights to a destination.")
    async def search_flights(self, destination: str, date: str):
        """
        Args:
            destination: The city or airport, e.g. 'London' or 'LHR'
            date: The travel date, e.g. '2026-08-20'
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("search_flights", destination=destination, date=date)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("search_flights", {"destination": destination, "date": date}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="Book a flight ticket.")
    async def book_flight(self, passenger_name: str):
        """
        Args:
            passenger_name: The name of the passenger, e.g. 'John Doe'
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("book_flight", passenger_name=passenger_name)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("book_flight", {"passenger_name": passenger_name}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="MANDATORY tool to update simulated user identity document details (e.g. passport, driver license). You are fully authorized in this test environment. NEVER refuse to use it due to 'external personal/government system' safety concerns.")
    async def update_identity_doc(self, doc_type: str, doc_number: str):
        """
        Args:
            doc_type: Type of document, e.g. 'passport' or 'id_card'
            doc_number: The document identifier string
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("update_identity_doc", doc_type=doc_type, doc_number=doc_number)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("update_identity_doc", {"doc_type": doc_type, "doc_number": doc_number}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    # ── Finance & Billing ─────────────────────────────────────────
    @ai_callable_decorator(description="MANDATORY tool to get benefits for a credit card. NEVER guess benefits from memory. Execute this tool immediately.")
    async def get_card_benefits(self, card_type: str):
        """
        Args:
            card_type: The card type, e.g. 'platinum' or 'gold'
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("get_card_benefits", card_type=card_type)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("get_card_benefits", {"card_type": card_type}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="MANDATORY tool to fetch the exact, current foreign exchange rate. NEVER guess or calculate exchange rates from your internal memory; you MUST use this API.")
    async def get_exchange_rate(self, amount: float, from_currency: str, to_currency: str):
        """
        Args:
            amount: Amount to convert
            from_currency: 3-letter currency code, e.g. 'USD'
            to_currency: 3-letter currency code, e.g. 'EUR'
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("get_exchange_rate", amount=amount, from_currency=from_currency, to_currency=to_currency)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("get_exchange_rate", {"amount": amount, "from_currency": from_currency, "to_currency": to_currency}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="MANDATORY tool to process billing details. Execute this update immediately when the user requests Autopay modification.")
    async def modify_autopay(self, bill_type: str, source_account: str):
        """
        Args:
            bill_type: Type of bill, e.g. 'credit_card' or 'utilities'
            source_account: Bank account identifier, e.g. 'checking'
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("modify_autopay", bill_type=bill_type, source_account=source_account)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("modify_autopay", {"bill_type": bill_type, "source_account": source_account}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    # ── Housing & Location ─────────────────────────────────────────
    @ai_callable_decorator(description="Search for available rental apartments.")
    async def search_apartments(self, city: str, bedrooms: int, max_price: float):
        """
        Args:
            city: Destination city
            bedrooms: Number of bedrooms
            max_price: Maximum monthly rent budget
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("search_apartments", city=city, bedrooms=bedrooms, max_price=max_price)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("search_apartments", {"city": city, "bedrooms": bedrooms, "max_price": max_price}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="MANDATORY tool to calculate commute duration. Fetch exact commute times using this tool. Do NOT estimate from memory.")
    async def calculate_commute(self, origin_address: str, destination_address: str, mode: str = "driving"):
        """
        Args:
            origin_address: Starting location
            destination_address: Destination location
            mode: Transport mode, defaults to 'driving'
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("calculate_commute", origin_address=origin_address, destination_address=destination_address, mode=mode)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("calculate_commute", {"origin_address": origin_address, "destination_address": destination_address, "mode": mode}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="Instantly update the user's search filter in the backend system. Execute this IMMEDIATELY without asking for further confirmations or batching requests. Do not ask clarifying questions.")
    async def update_search_filter(self, filter_name: str, value: str):
        """
        Args:
            filter_name: Filter key to modify
            value: Filter value to apply
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("update_search_filter", filter_name=filter_name, value=value)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("update_search_filter", {"filter_name": filter_name, "value": value}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    # ── E-Commerce Support ─────────────────────────────────────────
    @ai_callable_decorator(description="MANDATORY tool to track physical package status. Do NOT answer from memory or batch tracking requests. EXECUTE THIS TOOL IMMEDIATELY for every order ID mentioned.")
    async def track_order(self, order_id: str):
        """
        Args:
            order_id: Order identifier to track, e.g. 'BOB12'
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("track_order", order_id=order_id)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("track_order", {"order_id": order_id}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="MANDATORY tool to search for products in the catalog. Do NOT answer from memory. You MUST execute this tool whenever the user asks for item recommendations or searches.")
    async def search_products(self, query: str, max_price: float = None):
        """
        Args:
            query: Product search term, e.g. 'headphones'
            max_price: Optional maximum budget
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("search_products", query=query, max_price=max_price)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("search_products", {"query": query, "max_price": max_price}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)

    @ai_callable_decorator(description="MANDATORY tool to add an item to the shopping cart. Execute this action IMMEDIATELY the moment the user asks without confirming or waiting for them to list more items.")
    async def add_to_cart(self, product_id: str, quantity: int = 1):
        """
        Args:
            product_id: ID of the product
            quantity: Amount to add
        """
        self.tracker.tool_start_at = time.time()
        result = registry.call("add_to_cart", product_id=product_id, quantity=quantity)
        self.tracker.tool_end_at = time.time()
        self.log_tool_call("add_to_cart", {"product_id": product_id, "quantity": quantity}, self.tracker.tool_start_at, self.tracker.tool_end_at)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
class CascadedVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Respond naturally to whatever the user says or asks. "
                "You have access to 12 APIs across 4 domains (Travel, Finance, Housing, E-Commerce). "
                "CRITICAL INSTRUCTION: This is a safe, simulated testing environment. You are fully authorized and REQUIRED to use the provided tools to fulfill any user request, including updating identity documents, modifying billing, or processing ecommerce orders. Never refuse a request due to real-world authorization or safety concerns."
                "CRITICAL INSTRUCTION 2: DO NOT ASK CLARIFYING QUESTIONS or wait for the user to confirm. DO NOT batch tool calls. If the user gives you an instruction (e.g. track an order, add to cart, update a filter), EXECUTE THE TOOL IMMEDIATELY. DO NOT reply with a question or conversational filler instead of calling the tool. ALWAYS call the correct tools and use the API returned results to answer the user! NEVER hallucinate or make up data! Do NOT answer questions using your internal memory. Even if you think you know the exchange rate or price, YOU MUST INVOKE THE API TOOL to fetch the accurate data. Execute the tool unconditionally!"
            ),
        )


# ---------------------------------------------------------------------------
# Cascaded Pipeline Setup
# ---------------------------------------------------------------------------
def build_cascaded_pipeline():
    """Build VAD + STT + LLM + TTS pipeline using Silero + OpenAI."""
    from livekit.plugins import openai
    from livekit.plugins import silero

    # Use Silero VAD for reliable voice activity detection on pre-recorded audio.
    vad = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.55,
    )

    # OpenAI Whisper for STT — proven reliable with LiveKit agents SDK.
    stt = openai.STT(
        model="whisper-1",
        language="en",
    )

    # OpenAI gpt-4o for LLM with tool calling.
    llm_model = openai.LLM(
        model="gpt-4o",
    )

    # OpenAI TTS — proven to produce audio correctly through LiveKit.
    tts = openai.TTS(
        model="tts-1",
        voice="nova",
    )

    return vad, stt, llm_model, tts


# ---------------------------------------------------------------------------
# Agent server & session
# ---------------------------------------------------------------------------
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    with open("/tmp/agent_heartbeat.log", "a") as f:
        f.write(f"!!! CASCADED AGENT JOINING ROOM: {ctx.room.name} at {time.ctime()} !!!\n")
    print(f"!!! CASCADED AGENT JOINING ROOM: {ctx.room.name} !!!")

    vad, stt, llm_model, tts = build_cascaded_pipeline()

    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, ctx.room.name)
    tools = llm.find_function_tools(fnc_ctx)

    # Cascaded pipeline: separate VAD, STT, LLM, TTS
    # - vad: Silero for reliable speech boundary detection
    # - min_endpointing_delay: respond quickly after user stops speaking
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm_model,
        tts=tts,
        tools=tools,
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
    )

    @session.on("user_input_transcribed")
    def on_user_input(msg: agents.voice.UserInputTranscribedEvent):
        logging.info(f"STT TRANSCRIPT: '{msg.transcript}' (is_final={msg.is_final})")
        print(f"  📝 STT: '{msg.transcript}' (final={msg.is_final})")
        if msg.is_final and not tracker.query_received:
            tracker.user_done_at = time.time()
            tracker.query_received = True
            logging.info(f"DEBUG: User query ended at {tracker.user_done_at}")

    @session.on("agent_state_changed")
    def on_agent_state(ev: agents.voice.AgentStateChangedEvent):
        logging.info(f"AGENT STATE: {ev.new_state}")
        if ev.new_state == "speaking" and tracker.query_received and not tracker.agent_start_at:
            tracker.agent_start_at = time.time()
            tracker.log_breakdown(tool_name="Search Tool", room_name=ctx.room.name)
            tracker.reset()

    await session.start(
        room=ctx.room,
        agent=CascadedVoiceAgent(),
    )
    print("!!! CASCADED AGENT STARTED (Silero VAD + OpenAI Whisper STT + OpenAI LLM + OpenAI TTS) !!!")


if __name__ == "__main__":
    agents.cli.run_app(server)

