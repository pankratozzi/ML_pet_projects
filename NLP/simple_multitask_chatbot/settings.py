import os
import logging


logger = logging.getLogger("pankrobox_bot")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Telegram bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    logging.error(
        "BOT_TOKEN env var is not found, cannot start the bot without it, create it with @BotFather Telegram bot! "
        "Now setting default token"
    )
    BOT_TOKEN = "YOUR-TOKEN-HERE"
else:
    logging.info("BOT_TOKEN found, starting the bot")

# Model configuration
DEFAULT_MODEL_NAME = "pankratozzi/rugpt3small_based_on_gpt2-finetuned-for-chat_3"
MODEL_NAME = os.getenv("MODEL_NAME")
if not MODEL_NAME:
    MODEL_NAME = DEFAULT_MODEL_NAME
    logging.info(f"MODEL_NAME env var is not found, using default model {MODEL_NAME}")
else:
    logging.info(f"MODEL_NAME is {MODEL_NAME}")
