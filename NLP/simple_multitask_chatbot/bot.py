from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text

from settings import BOT_TOKEN, MODEL_NAME, logger
from generator import Conversation
from transformers import pipeline

import numpy as np


# TODO: ruBert for question matching: <cls>...question_input...<sep>...question_candidate<sep> -> probability similarity
# TODO: annoy search for similar embedding
# TODO: gpt + Bert -> ranking
answer_handler = Conversation(MODEL_NAME, block_size=128, num_seq=3)
mask_filler = pipeline("fill-mask", "pankratozzi/ruRoberta-large-finetuned-for-chat")
calculator = pipeline("fill-mask", "pankratozzi/ruRoberta-large-arithmetics")

bot = Bot(token=BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)


@dp.message_handler(commands=["start"])
async def welcome(message: types.Message):
    await message.answer("Hello! Let's talk a little...")


@dp.message_handler()
async def chatter(message: types.Message):
    try:
        user_input = message.text.strip()
        if not user_input:
            await message.reply(
                "Для разговора начните печатать, для подсказки введеите предложение с неизвестными словами '___',"
                "для совершения простых арифметиеских операций начните фразу с 'calc:'"
            )
            return
        # naive intent classification - parsing. TODO: count-based classifier baseline; NER classification
        # intent = baseline_clf.predict(user_input)  # sklearn make_pipeline()
        if "___" in user_input and not user_input.startswith("calc:"):
            user_input = user_input.replace("___", "<mask>")
            answer_message = mask_filler(user_input, top_k=3)[np.random.randint(3)]["sequence"]

        elif user_input.startswith("calc:"):
            user_input = "Q: " + user_input.replace("calc:", "").strip() + " A: <mask>"
            answer_message = calculator(user_input, top_k=3)
            answer_message = max(answer_message, key=lambda x: x["score"])["sequence"]
            answer_message = answer_message.split()[-1].strip()
        else:
            user_input = "Вопрос " + user_input.strip()
            answer_message = answer_handler(user_input)
            answer_message = answer_message.replace("Ответ", "").strip()

        await message.answer(answer_message)

    except Exception as err:
        logger.error("Error while attempting to answer", exc_info=err)
        await message.reply(f"Error while attempting to answer:\n\n{err}")


@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return
    await state.finish()
    await message.reply('Cancelled.', reply_markup=types.ReplyKeyboardRemove())


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
