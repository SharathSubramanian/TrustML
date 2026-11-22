"""Conversational Ask TrustML.

The NaviGator LLM holds the conversation, but it never computes anything:
  1. it reads the question + recent turns and picks up to MAX_TOOLS TrustML tools (JSON, schema-checked);
  2. each tool is a deterministic FollowUp handler (stored verified results, or a new verified experiment
     on the run's exact split);
  3. it writes a conversational reply from the tool outputs only;
  4. every number in the reply is checked against the tool outputs and the run's verified conclusions.
     If a number is unsupported, the reply is replaced by the tools' own text (and that is shown).
Without an LLM, the deterministic router answers and the reply is phrased from templates.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from trustml.followup import INTENTS, Answer, FollowUp

MAX_TOOLS = 3
HISTORY_TURNS = 6

TOOL_DOCS = {**INTENTS, "general": "Anything else about this run, answered only from its stored conclusions."}


class ToolCall(BaseModel):
    tool: Literal[tuple(TOOL_DOCS)]  # type: ignore[valid-type]
    question: str = Field(min_length=2, description="self-contained question for the tool (resolve 'it', 'that')")


class Plan(BaseModel):
    calls: list[ToolCall] = Field(default_factory=list, max_length=MAX_TOOLS)
    reply: Optional[str] = None  # only for small talk that needs no data


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    tools: list[dict] = []
    verification: dict = {}
    source: str = ""
    seconds: float = 0.0


PLAN_PROMPT = """You are TrustML's assistant for ONE finished ML analysis run. You do not know any numbers yourself;
you must call tools. Choose up to {k} tools that answer the user's latest message. Rewrite each tool question so it is
self-contained (resolve pronouns from the conversation, e.g. "remove it" -> "remove credit_score").
Tools:
{tools}
Run context: target '{target}', task {task}, final model {final}, feature columns: {cols}.
Reply with JSON only: {{"calls": [{{"tool": "...", "question": "..."}}], "reply": null}}.
Use "reply" (with no calls) ONLY for greetings or thanks."""

ANSWER_PROMPT = """You are TrustML's assistant. Answer the user's latest message conversationally, in 2-6 sentences,
like a helpful senior data scientist. Use ONLY the tool results below; copy numbers exactly as written there; never
compute or invent numbers. If tools ran a new experiment, say so and say it was verified (or not). If the tools do not
answer the question, say what you can answer instead. SHAP and correlations are not causal; say so if relevant.
Tool results (JSON): {results}"""


def _extract_json(text: str) -> Optional[dict]:
    """First JSON object in the text (tolerates ```json fences and reasoning text around it)."""
    text = (text or "").strip()
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|thanks?( you)?|thank you|thx|ok(ay)?|cool|great|nice|bye|good (morning|evening))"
                        r"[\s!.]*$", re.I)


class Conversation:
    def __init__(self, run_dir: str | Path, llm_client=None, llm_model: Optional[str] = None):
        self.fu = FollowUp(run_dir, llm_client=None, llm_model=None)  # tools stay deterministic
        self.fu_llm = FollowUp(run_dir, llm_client=llm_client, llm_model=llm_model) if llm_client else None
        self.client, self.model = llm_client, llm_model
        self.history: list[Turn] = []

    # ------------------------------------------------------------------ tools
    def _tool(self, name: str, question: str) -> Answer:
        fu = self.fu_llm if name == "general" and self.fu_llm else self.fu
        fu._ctx = self.fu._ctx  # share the rebuilt split between calls
        handler = getattr(fu, f"_a_{name}", fu._a_general)
        a = handler(question)
        self.fu._ctx = fu._ctx
        a.intent = name
        return a

    def _plan(self, message: str) -> tuple[Plan, str]:
        if SMALL_TALK.match(message):
            return Plan(reply="You're welcome! Ask me anything else about this analysis: for example where the model "
                              "fails, what happens if a feature is removed, or whether its probabilities can be trusted."
                        if re.search(r"thank|thx", message, re.I) else
                        "Hi! I can explain this run's findings or run a new verified experiment on its exact split. "
                        "What would you like to know?"), "small talk"
        if self.client is None:
            intent, how = self.fu.route(message)
            return Plan(calls=[ToolCall(tool=intent, question=message)]), f"router:{how}"
        from trustml.llm import chat
        fin = (self.fu.summary.get("final_model") or {})
        sys = PLAN_PROMPT.format(k=MAX_TOOLS, tools="\n".join(f"- {k}: {v}" for k, v in TOOL_DOCS.items()),
                                 target=self.fu.target, task=(self.fu.findings.get("profile", {}).get("target") or {})
                                 .get("task_type"), final=f"{fin.get('exp_id')} ({(fin.get('spec') or {}).get('model')})",
                                 cols=[c for c in self.fu.columns if c != self.fu.target][:40])
        msgs = [{"role": "system", "content": sys}]
        for t in self.history[-HISTORY_TURNS:]:
            msgs.append({"role": t.role, "content": t.content[:800]})
        msgs.append({"role": "user", "content": message})
        try:
            r = chat(self.client, self.model, messages=msgs)
            obj = _extract_json(r.choices[0].message.content or "")
            plan = Plan.model_validate(obj or {})
            if not plan.calls and not plan.reply:
                raise ValueError("empty plan")
            if plan.calls and plan.calls[0].tool == "remove_feature" and not self.fu.find_feature(plan.calls[0].question):
                raise ValueError("remove_feature without a known column")
            return plan, f"llm:{self.model}"
        except (ValidationError, ValueError, Exception) as e:  # noqa: BLE001
            intent, how = self.fu.route(message)
            return Plan(calls=[ToolCall(tool=intent, question=message)]), f"router:{how} (LLM plan rejected: {type(e).__name__})"

    # ------------------------------------------------------------------ main
    def send(self, message: str) -> Turn:
        t0 = time.time()
        plan, planned_by = self._plan(message)
        if plan.reply and not plan.calls:
            from trustml.narrative import numbers_supported
            ok, _ = numbers_supported(plan.reply, [])
            reply = plan.reply if ok else "Happy to help. Ask me anything about this analysis."
            turn = Turn(role="assistant", content=reply, source=f"{planned_by} (no tools needed)")
            return self._push(message, turn, t0)

        answers: list[Answer] = []
        for c in plan.calls[:MAX_TOOLS]:
            try:
                answers.append(self._tool(c.tool, c.question))
            except Exception as e:  # noqa: BLE001
                answers.append(Answer(question=c.question, intent=c.tool, text=f"The tool failed: {e}",
                                      source="tool error"))
        tools = [{"tool": a.intent, "question": a.question, "text": a.text, "source": a.source,
                  "new_experiments": a.new_experiments, "verification": a.verification} for a in answers]
        verified = [a.verification.get("passed") for a in answers if a.verification]
        verification = {"experiments_verified": all(verified) if verified else None,
                        "n_new_experiments": sum(len(a.new_experiments) for a in answers)}

        reply, source = None, ""
        if self.client is not None:
            reply, source = self._compose(message, tools)
        if reply is None:
            reply = self._template(tools)
            source = source or "deterministic (no LLM)"
        turn = Turn(role="assistant", content=reply, tools=tools, verification=verification,
                    source=f"plan: {planned_by} · reply: {source}")
        return self._push(message, turn, t0)

    def _compose(self, message: str, tools: list[dict]) -> tuple[Optional[str], str]:
        from trustml.llm import chat
        from trustml.narrative import numbers_supported
        msgs = [{"role": "system", "content": ANSWER_PROMPT.format(results=json.dumps(tools)[:12000])}]
        for t in self.history[-HISTORY_TURNS:]:
            msgs.append({"role": t.role, "content": t.content[:800]})
        msgs.append({"role": "user", "content": message})
        try:
            r = chat(self.client, self.model, messages=msgs)
            text = (r.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            return None, f"deterministic (LLM unavailable: {type(e).__name__})"
        facts = [t["text"] for t in tools] + self.fu.summary.get("conclusions", []) + self.fu.summary.get("caveats", [])
        ok, bad = numbers_supported(text, facts)
        if text and ok:
            return text, f"llm:{self.model} (every number checked against tool output)"
        return None, f"deterministic (LLM reply rejected: unsupported numbers {bad[:4]})"

    @staticmethod
    def _template(tools: list[dict]) -> str:
        if len(tools) == 1:
            t = tools[0]
            lead = ("I ran a new experiment on the run's exact split to answer that. " if t["new_experiments"] else "")
            return lead + t["text"]
        return "\n\n".join(f"**{t['tool'].replace('_', ' ')}** – {t['text']}" for t in tools)

    def _push(self, message: str, turn: Turn, t0: float) -> Turn:
        turn.seconds = round(time.time() - t0, 2)
        self.history += [Turn(role="user", content=message), turn]
        with open(self.fu.dir / "chat.jsonl", "a") as f:
            f.write(json.dumps({"user": message, **turn.model_dump()}) + "\n")
        return turn
