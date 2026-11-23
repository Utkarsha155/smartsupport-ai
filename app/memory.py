"""Per-user conversation memory stored in MongoDB.

Each message is one document in the `messages` collection:
    { user_id, role: "user" | "assistant", content, created_at }
"""
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING, MongoClient

from app import config

_client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
_messages = _client[config.MONGO_DB_NAME]["messages"]


def init_db() -> None:
    """Checks the connection and creates an index for fast per-user lookups."""
    _client.admin.command("ping")
    _messages.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])


def is_connected() -> bool:
    try:
        _client.admin.command("ping")
        return True
    except Exception:
        return False


def save_message(user_id: str, role: str, content: str) -> None:
    _messages.insert_one(
        {
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        }
    )


def get_recent_messages(user_id: str, limit: int = config.HISTORY_LIMIT) -> list[dict]:
    """Returns the last `limit` messages of a user, oldest first."""
    cursor = (
        _messages.find({"user_id": user_id}, {"_id": 0})
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
    return list(cursor)[::-1]


def get_all_messages(user_id: str) -> list[dict]:
    return list(
        _messages.find({"user_id": user_id}, {"_id": 0}).sort("created_at", ASCENDING)
    )


def clear_messages(user_id: str) -> int:
    return _messages.delete_many({"user_id": user_id}).deleted_count
