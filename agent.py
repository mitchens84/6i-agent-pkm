import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langsmith import Client
from langchain.callbacks.tracers import ConsoleCallbackHandler
import logging
from src.agents.pkm_agent import PKMAgent
from src.workflows.research_workflow import ResearchWorkflow
from src.utils.data_utils import download_voice_message, transcribe_audio
import asyncio
from aiohttp import web

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Load environment variables
load_dotenv(dotenv_path="config/.env")
api_key = os.getenv('LANGSMITH_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LangSmith client
client = Client()
tracer = ConsoleCallbackHandler()

# Initialize PKM Agent and Research Workflow
research_workflow = ResearchWorkflow()
pkm_agent = research_workflow.pkm_agent

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hi! I'm your AI research assistant. How can I help you today?")
    logger.info(f"Start command issued by user {update.effective_user.id}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text
    user_id = update.effective_user.id
    
    logger.info(f"Received query from user {user_id}: {query}")
    await update.message.reply_text("Processing your query. This may take a few moments...")
    
    try:
        response = pkm_agent.process_query(query)
        await update.message.reply_text(response)
        logger.info(f"Response sent to user {user_id}")
    except Exception as e:
        error_message = f"An error occurred while processing your request: {str(e)}"
        await update.message.reply_text(error_message)
        logger.error(f"Error processing query for user {user_id}: {str(e)}", exc_info=True)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    logger.info(f"Received voice message from user {user_id}")
    await update.message.reply_text("Processing your voice message. This may take a few moments...")

    try:
        file_path = await download_voice_message(update.message.voice)
        transcription = await transcribe_audio(file_path)
        response = pkm_agent.process_query(transcription)
        await update.message.reply_text(f"Transcription: {transcription}\n\nResponse: {response}")
        logger.info(f"Response sent to user {user_id}")
    except Exception as e:
        error_message = f"An error occurred while processing your voice message: {str(e)}"
        await update.message.reply_text(error_message)
        logger.error(f"Error processing voice message for user {user_id}: {str(e)}", exc_info=True)

async def handle_webhook(request):
    update = await request.json()
    await application.process_update(Update.de_json(update, application.bot))
    return web.Response(text="OK")

async def start_webhook(app, webhook_path):
    await application.bot.set_webhook(url=f"{os.getenv('WEBHOOK_URL')}{webhook_path}")
    return app

async def run_bot():
    global application
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    webhook_path = f"/{os.getenv('TELEGRAM_BOT_TOKEN')}"
    port = int(os.environ.get("PORT", 10000))

    app = web.Application()
    app.router.add_post(webhook_path, handle_webhook)
    
    runner = web.AppRunner(await start_webhook(app, webhook_path))
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    
    await site.start()
    logger.info(f"Webhook set up on port {port}")

    # Keep the bot running
    while True:
        await asyncio.sleep(3600)

def main():
    try:
        asyncio.run(run_bot())
    except Exception as e:
        logger.error(f"Critical error in main function: {str(e)}", exc_info=True)
    finally:
        logger.info("Shutting down Telegram bot")

if __name__ == "__main__":
    main()