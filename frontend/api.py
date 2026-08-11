import json
import requests

from config.frontend import CHAT_API, STREAM_API, TIMEOUT


class ChatAPI:
    def __init__(self, language="auto"):
        self.language = language

    def _build_payload(self, query, history):
        return {
            "query": query,
            "language": self.language,
            "history": history or [],
        }

    def chat(self, query, history=None):
        response = requests.post(
            CHAT_API,
            json=self._build_payload(query, history),
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "answer": data.get("answer", ""),
            "sources": data.get("sources", []),
            "latency_ms": data.get("latency_ms"),
        }

    def stream(self, query, history=None):
        response = requests.post(
            STREAM_API,
            json=self._build_payload(query, history),
            timeout=TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        return response

    @staticmethod
    def parse_sse(response):
        event = None
        data_lines = []

        for raw in response.iter_lines(chunk_size=8192, decode_unicode=True):
            if raw is None:
                continue

            line = raw.strip()

            if not line:
                if event and data_lines:
                    payload = json.loads("\n".join(data_lines))
                    yield event, payload

                event = None
                data_lines = []
                continue

            if line.startswith("event:"):
                event = line[6:].strip()

            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

    @staticmethod
    def token_generator(response, holder):
        for event, payload in ChatAPI.parse_sse(response):
            if event == "delta":
                yield payload.get("text", "")

            elif event == "sources":
                holder["sources"] = payload.get("sources", [])

            elif event == "done":
                holder["latency_ms"] = payload.get("latency_ms")

            elif event == "error":
                holder["error"] = payload.get("message")
                return


def ask(query, language="auto", history=None):
    api = ChatAPI(language)
    return api.chat(query, history)


def ask_stream(query, language="auto", history=None):
    api = ChatAPI(language)
    return api.stream(query, history)