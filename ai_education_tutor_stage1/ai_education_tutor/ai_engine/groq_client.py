from groq import Groq


def create_client(api_key):
    """
    Create and return a Groq client.
    """

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(api_key=api_key)


def call_groq(
    client,
    model,
    system_prompt,
    user_prompt,
):
    """
    Send a request to Groq and return the generated text.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    if not response.choices:
        raise RuntimeError(
            "Groq returned no response."
        )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return answer.strip()
