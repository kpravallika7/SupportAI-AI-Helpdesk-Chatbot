"""
SupportAI — Helpdesk Agent

An AI-powered customer support agent that combines:
- FAQ knowledge base
- Keyword-based retrieval
- TF-IDF similarity matching
- Hybrid FAQ search
- Groq LLM responses
- Confidence-based fallback
- Conversation history
- Human-agent escalation
"""

import os
import random
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Please create a .env file and add your Groq API key."
    )


# ============================================================
# FAQ Knowledge Base
# ============================================================

faqs = [
    {
        "id": "faq-001",
        "category": "Account",
        "question": "How do I reset my password?",
        "answer": (
            "Click 'Forgot Password' on the login page. "
            "Enter your registered email address and check your inbox "
            "for a reset link valid for 24 hours."
        ),
        "keywords": ["password", "reset", "forgot", "login"],
    },
    {
        "id": "faq-002",
        "category": "Billing",
        "question": "What is your refund policy?",
        "answer": (
            "We offer full refunds within 30 days of purchase for unused "
            "subscriptions. Partial refunds are available for annual plans "
            "cancelled after 30 days."
        ),
        "keywords": ["refund", "money back", "cancel", "billing"],
    },
    {
        "id": "faq-003",
        "category": "Shipping",
        "question": "How long does shipping take?",
        "answer": (
            "Standard shipping takes 5–7 business days. Express shipping "
            "(2–3 business days) is available at checkout for an additional fee."
        ),
        "keywords": ["shipping", "delivery", "tracking", "express"],
    },
    {
        "id": "faq-004",
        "category": "Account",
        "question": "How do I update my email address?",
        "answer": (
            "Go to Settings → Account → Email. Enter your new email and "
            "confirm via the verification link sent to the new address."
        ),
        "keywords": ["email", "update", "change", "account"],
    },
    {
        "id": "faq-005",
        "category": "Technical",
        "question": "The app keeps crashing. What should I do?",
        "answer": (
            "First, update to the latest version from the App Store or "
            "Google Play. If the issue persists, clear the app cache in "
            "Settings → Storage, then restart your device."
        ),
        "keywords": ["crash", "bug", "error", "technical", "app"],
    },
]


# ============================================================
# Keyword Search
# ============================================================

def search_by_keyword(faqs, query):
    """
    Search FAQs using meaningful keyword matches.
    Generic/common words are ignored.
    """

    stop_words = {
        "i", "me", "my", "mine",
        "the", "a", "an",
        "is", "are", "am", "was", "were",
        "do", "does", "did",
        "how", "what", "where", "when", "why",
        "can", "could", "would", "should",
        "will",
        "get", "give", "tell",
        "please",
        "to", "for", "of", "on", "in",
        "and", "or", "with",
        "your", "you",
        "about",
    }

    query_words = {
        word.strip(".,?!'\"")
        for word in query.lower().split()
    }

    query_words = {
        word
        for word in query_words
        if word and word not in stop_words
    }

    if not query_words:
        return []

    scored_results = []

    for faq in faqs:

        faq_keywords = {
            str(keyword).lower().strip()
            for keyword in faq.get("keywords", [])
        }

        faq_question_words = {
            word.strip(".,?!'\"")
            for word in faq["question"].lower().split()
            if word not in stop_words
        }

        searchable_words = faq_keywords | faq_question_words

        matched_words = query_words.intersection(searchable_words)

        if matched_words:
            scored_results.append(
                (len(matched_words), faq)
            )

    scored_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        faq
        for _, faq in scored_results
    ]


# ============================================================
# FAQ Utility Functions
# ============================================================

def get_faq_by_id(faqs, faq_id):
    """Return FAQ by ID, or None if not found."""

    for faq in faqs:
        if faq["id"] == faq_id:
            return faq

    return None


def get_faqs_by_category(faqs, category):
    """Return all FAQs in a category (case-insensitive)."""

    cat = category.lower()

    return [
        faq
        for faq in faqs
        if faq.get("category", "").lower() == cat
    ]


# ============================================================
# Groq LLM Client
# ============================================================

FAQ_SYSTEM_PROMPT = """You are SupportAI, a friendly and professional customer support agent.

Your job is to answer the user's question using ONLY the provided FAQ.

Rules:
- Use the FAQ answer as the source of truth.
- If the user's question is a paraphrase of the FAQ question, answer it using the FAQ answer.
- Preserve all important instructions, conditions, limits, and time periods from the official answer.
- Rephrase the answer naturally rather than copying it word-for-word.
- Do not add new policies, procedures, guarantees, or facts.
- Do not assume information that is not present in the FAQ.
- Keep the response concise and under 150 words.
- If the FAQ genuinely does not contain enough information to answer the user's question, say:
  "I don't have enough information in the FAQ to answer that."
"""

class LLMClient:
    """Client for generating responses using the Groq API."""

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.base_url = GROQ_URL

    def generate(self, messages, temperature=0.2, max_tokens=300):
        """Send a chat-completion request to Groq."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if not response.ok:
            print("\nGroq API Error:")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            print("\nCheck the GROQ_API_KEY and GROQ_MODEL values in your .env/configuration.")
            response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    def generate_faq_response(self, question, faq):
        """Generate an answer using the selected FAQ as context."""

        messages = [
            {
                "role": "system",
                "content": FAQ_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
Customer question:
{question}

Relevant FAQ:
Question: {faq["question"]}
Answer: {faq["answer"]}

Use the FAQ information above to answer the customer's question.
Do not invent information that is not present in the FAQ.
""",
            },
        ]

        return self.generate(messages)

# ============================================================
# TF-IDF FAQ Matcher
# ============================================================

class FAQMatcher:
    """Match queries to FAQs using TF-IDF cosine similarity."""

    def __init__(self, faqs):

        self.faqs = faqs

        # Build corpus using FAQ question + keywords
        self.corpus = [
            f"{faq['question']} {' '.join(faq.get('keywords', []))}"
            for faq in faqs
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.corpus
        )

    def match(self, query, top_k=3):
        """Return top-k FAQs ranked by cosine similarity."""

        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vec,
            self.tfidf_matrix
        ).flatten()

        ranked = sorted(
            zip(self.faqs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        ranked = [
            (faq, score)
            for faq, score in ranked
            if score > 0
        ]

        return [
            (faq, round(float(score), 4))
            for faq, score in ranked[:top_k]
        ]

    def best_match(self, query, threshold=0.15):
        """Return the best FAQ if similarity meets the threshold."""

        results = self.match(query, top_k=1)

        if results and results[0][1] >= threshold:
            return results[0]

        return None

    def explain_match(self, query):
        """Return formatted top matches with scores."""

        results = self.match(query, top_k=3)

        if not results:
            return "  (no results)"

        return "\n".join(
            f"  {i}. [{score:.4f}] {faq['question']}"
            for i, (faq, score) in enumerate(results, 1)
        )


# ============================================================
# Hybrid Search
# ============================================================

def hybrid_search(faqs, query, matcher, top_k=3):
    """
    Combine keyword and TF-IDF search results.

    Keyword matches receive a score of 0.5.
    TF-IDF similarity is used when it provides a stronger score.
    """

    score_map = {}

    # -----------------------------
    # Keyword search
    # -----------------------------

    keyword_results = search_by_keyword(
        faqs,
        query
    )

    for faq in keyword_results:
        score_map[faq["id"]] = (
            faq,
            0.5
        )

    # -----------------------------
    # TF-IDF search
    # -----------------------------

    tfidf_results = matcher.match(
        query,
        top_k=len(faqs)
    )

    for faq, score in tfidf_results:

        faq_id = faq["id"]

        if faq_id in score_map:

            existing_faq, existing_score = score_map[faq_id]

            score_map[faq_id] = (
                existing_faq,
                max(existing_score, score)
            )

        else:

            score_map[faq_id] = (
                faq,
                score
            )

    # -----------------------------
    # Sort results
    # -----------------------------

    merged = sorted(
        score_map.values(),
        key=lambda x: x[1],
        reverse=True
    )

    return merged[:top_k]


# ============================================================
# Conversation Data Structure
# ============================================================

@dataclass
class ConversationTurn:
    role: str
    content: str
    faq_id: str = None
    confidence: float = None


# ============================================================
# SupportAI Agent
# ============================================================

class SupportAgent:
    """Orchestrates hybrid search, LLM responses, and escalation."""

    def __init__(
        self,
        faqs,
        llm_client,
        matcher,
        confidence_threshold=0.15
    ):

        self.faqs = faqs
        self.llm_client = llm_client
        self.matcher = matcher
        self.confidence_threshold = confidence_threshold

        self.history = []
        self.escalated = False
        self._low_confidence_streak = 0

    def handle_message(self, user_message):

        self.history.append(
            ConversationTurn(
                role="user",
                content=user_message
            )
        )

        if self.escalated:

            msg = "Your case is already with our support team."

            self.history.append(
                ConversationTurn(
                    role="assistant",
                    content=msg
                )
            )

            return msg

        results = hybrid_search(
            self.faqs,
            user_message,
            self.matcher,
            top_k=1
        )

        if (
            results
            and results[0][1] >= self.confidence_threshold
        ):

            faq, confidence = results[0]

            response = self.llm_client.generate_faq_response(
                user_message,
                faq
            )

            self._low_confidence_streak = 0

            self.history.append(
                ConversationTurn(
                    role="assistant",
                    content=response,
                    faq_id=faq["id"],
                    confidence=confidence
                )
            )

            return response

        # -----------------------------
        # No confident match
        # -----------------------------

        self._low_confidence_streak += 1

        response = (
            "I don't have information about that in my FAQ "
            "knowledge base. Would you like me to connect you "
            "with a human support agent?"
        )

        if self._low_confidence_streak >= 3:

            response += (
                "\n\nYou have had several questions outside "
                "my FAQ scope. Type 'escalate' to connect "
                "with a human agent."
            )

        self.history.append(
            ConversationTurn(
                role="assistant",
                content=response,
                confidence=0.0
            )
        )

        return response

    def escalate(self, reason="User requested human support"):

        self.escalated = True

        ticket_id = (
            f"TICKET-{random.randint(10000, 99999)}"
        )

        response = (
            "Your request has been escalated to our support team.\n"
            f"Ticket ID: {ticket_id}\n"
            "Estimated response time: within 4 business hours.\n"
            "A support agent will follow up via email."
        )

        self.history.append(
            ConversationTurn(
                role="assistant",
                content=response
            )
        )

        return response

    def get_conversation_summary(self):

        lines = [
            "=== Conversation Summary ==="
        ]

        for i, turn in enumerate(
            self.history,
            1
        ):

            label = (
                "You"
                if turn.role == "user"
                else "SupportAI"
            )

            line = (
                f"{i}. [{label}] "
                f"{turn.content[:80]}"
            )

            if turn.faq_id:
                line += (
                    f" (faq: {turn.faq_id}, "
                    f"conf: {turn.confidence})"
                )

            lines.append(line)

        return "\n".join(lines)

    def reset(self):

        self.history.clear()
        self.escalated = False
        self._low_confidence_streak = 0


# ============================================================
# Interactive SupportAI Chat
# ============================================================

def run_supportai_chat(faqs, llm_client, matcher):

    agent = SupportAgent(
        faqs,
        llm_client,
        matcher
    )

    print()
    print("╔══════════════════════════════════════════╗")
    print("║       SupportAI — Helpdesk Agent         ║")
    print("╚══════════════════════════════════════════╝")

    print()
    print("Commands:")
    print("  history  → View conversation history")
    print("  reset    → Reset the conversation")
    print("  escalate → Connect with a human agent")
    print("  quit     → Exit SupportAI")

    while True:

        try:
            user_input = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nThank you for using SupportAI. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # -----------------------------
        # Exit
        # -----------------------------

        if cmd in ("quit", "exit"):

            print(
                "\nThank you for using SupportAI. Goodbye!"
            )

            break

        # -----------------------------
        # Escalation
        # -----------------------------

        elif cmd == "escalate":

            print(
                f"\nSupportAI:\n{agent.escalate()}"
            )

        # -----------------------------
        # History
        # -----------------------------

        elif cmd == "history":

            print(
                f"\n{agent.get_conversation_summary()}"
            )

        # -----------------------------
        # Reset
        # -----------------------------

        elif cmd == "reset":

            agent.reset()

            print(
                "\nConversation reset."
            )

        # -----------------------------
        # Normal question
        # -----------------------------

        else:

            response = agent.handle_message(
                user_input
            )

            turn = agent.history[-1]

            meta = ""

            if turn.confidence is not None:

                meta = (
                    f" [confidence: "
                    f"{turn.confidence:.2f}"
                )

                if turn.faq_id:

                    meta += (
                        f", faq: "
                        f"{turn.faq_id}"
                    )

                meta += "]"

            print(
                f"\nSupportAI{meta}:\n{response}"
            )


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":

    matcher = FAQMatcher(faqs)
    llm = LLMClient(GROQ_API_KEY, GROQ_MODEL)

    run_supportai_chat(
        faqs,
        llm,
        matcher
    )