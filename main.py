from flask import Flask, render_template, request
import json
import os
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "id": "Indonesian",
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "it": "Italian",
    "pt": "Portuguese",
    "ko": "Korean"
}

def generate_story(judul, pesan_moral, jumlah_halaman, lang1_code, lang2_code):
    lang1_name = LANGUAGES.get(lang1_code, "English")
    lang2_name = LANGUAGES.get(lang2_code, "Spanish")

    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": "You are an award-winning bilingual children's story writer. Always respond with valid JSON only, no markdown formatting, no extra text. Never use double quotes inside story text - use single quotes for dialogue instead."
            },
            {
                "role": "user",
                "content": f"""
Write a children's story in TWO languages: {lang1_name} and {lang2_name}.
Both versions must have EXACTLY the same story content, only different language.

Title/Theme: {judul}
Moral / Lesson: {pesan_moral}

Output MUST be valid JSON, no markdown backticks, no extra text.
Create exactly {jumlah_halaman} pages.
IMPORTANT: Do not use double quotes (") inside the story text. Use single quotes (') instead.

Format:
{{
  "title_lang1": "",
  "title_lang2": "",
  "cover_description_lang1": "",
  "cover_description_lang2": "",
  "pages": [
    {{
      "page": 1,
      "story_lang1": "",
      "story_lang2": ""
    }}
  ]
}}
"""
            }
        ],
        temperature=0.8,
        top_p=0.95,
        max_tokens=8000,
        extra_body={"chat_template_kwargs": {"thinking": False}},
        stream=False
    )
    raw = completion.choices[0].message.content

    if raw is None:
        raise Exception("Model did not return content. Please try again.")

    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise Exception(f"AI generated invalid format, please click 'Create Story' again. (Detail: {e})")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.route("/generate", methods=["POST"])
def generate():
    judul = request.form.get("judul")
    pesan_moral = request.form.get("pesan_moral")
    jumlah_halaman = request.form.get("jumlah_halaman", "3")
    lang1_code = request.form.get("bahasa_cerita", "en")
    lang2_code = request.form.get("bahasa_latihan", "es")

    try:
        story = generate_story(judul, pesan_moral, jumlah_halaman, lang1_code, lang2_code)
        return render_template(
            "result.html",
            story=story,
            lang1_code=lang1_code,
            lang2_code=lang2_code,
            lang1_name=LANGUAGES.get(lang1_code, "Language 1"),
            lang2_name=LANGUAGES.get(lang2_code, "Language 2")
        )
    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)