"""
message.py
----------
Represents a single unit of information passed between agents.
Encoding is intentionally simple (JSON over UTF-8) — the focus of this
project is topology evolution, not protocol design.
"""

import json


class Message:
    """
    A discrete packet of information sent from one agent to another.

    Attributes
    ----------
    sender_id : int
        ID of the agent that created this message.
    receiver_id : int
        ID of the intended recipient agent.
    content : dict
        Arbitrary key-value payload (must be JSON-serialisable).
    is_distorted : bool
        Flag that can be set to True by middleware to indicate the message
        was altered in transit (reserved for future use; default False).
    """

    def __init__(
        self,
        sender_id: int,
        receiver_id: int,
        content: dict,
        is_distorted: bool = False,
    ):
        """
        Create a message.

        Parameters
        ----------
        sender_id : int
            ID of the sending agent.
        receiver_id : int
            ID of the receiving agent.
        content : dict
            Data payload.  All values must be JSON-serialisable.
        is_distorted : bool
            Whether the message has been tampered with (default False).
        """
        self.sender_id: int = sender_id
        self.receiver_id: int = receiver_id
        self.content: dict = content
        self.is_distorted: bool = is_distorted

    def encode(self) -> bytes:
        """
        Serialise the message to bytes using JSON + UTF-8.

        The full message envelope (sender, receiver, distortion flag, and
        content) is encoded so that a receiver can reconstruct the object
        from the raw bytes if needed.

        Returns
        -------
        bytes
            UTF-8 encoded JSON representation of this message.
        """
        payload = {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "is_distorted": self.is_distorted,
            "content": self.content,
        }
        return json.dumps(payload).encode("utf-8")

    def __repr__(self) -> str:
        return (f"Message(from={self.sender_id}, to={self.receiver_id}, "
                f"distorted={self.is_distorted}, content={self.content})")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Message Test ===\n")

    # --- 1. Basic creation and repr ---
    msg = Message(sender_id=0, receiver_id=3, content={"task_id": 2, "position": [4, 7]})
    print("Created:", msg)
    assert msg.sender_id == 0
    assert msg.receiver_id == 3
    assert msg.is_distorted is False

    # --- 2. encode() produces bytes ---
    encoded = msg.encode()
    print("\nEncoded bytes:", encoded)
    assert isinstance(encoded, bytes), "encode() must return bytes"

    # --- 3. Encoded bytes decode back to correct structure ---
    import json
    decoded = json.loads(encoded.decode("utf-8"))
    print("Decoded dict :", decoded)
    assert decoded["sender_id"] == 0
    assert decoded["receiver_id"] == 3
    assert decoded["is_distorted"] is False
    assert decoded["content"]["task_id"] == 2
    assert decoded["content"]["position"] == [4, 7]
    print("Round-trip decode check passed.")

    # --- 4. is_distorted flag ---
    distorted = Message(sender_id=1, receiver_id=2,
                        content={"info": "fake"}, is_distorted=True)
    print("\nDistorted message:", distorted)
    assert distorted.is_distorted is True
    distorted_decoded = json.loads(distorted.encode().decode("utf-8"))
    assert distorted_decoded["is_distorted"] is True
    print("Distortion flag preserved in encoding.")

    # --- 5. Empty content is valid ---
    empty_msg = Message(sender_id=5, receiver_id=5, content={})
    encoded_empty = empty_msg.encode()
    assert json.loads(encoded_empty)["content"] == {}
    print("\nEmpty content message encoded correctly.")

    print("\nAll tests passed.")
