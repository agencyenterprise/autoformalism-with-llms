import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro
from openai import OpenAI

from autoformalism_with_llms import prompt
from autoformalism_with_llms.config import config
from autoformalism_with_llms.dataset import MiniF2FMATH

CLIENT = OpenAI(api_key=config.OPENAI_API_KEY)


@dataclass
class Args:
    name: str
    model: str = "gpt-4-turbo-2024-04-09"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0


@dataclass(frozen=True)
class FEWSHOTIDS:
    """IDS of the few-shot learning examples used in the paper

    We found these by copying snippets of the questions from the paper appendix and
    searching the dataset for matching questions.  See the accompying notebook
    `find_fewshot_ids.ipynb` for more details.
    """

    algebra: tuple[str, ...] = (
        "245",
        "76",
        "478",
        "338",
        "422",
        "43",
        "756",
        "149",
        "48",
        "410",
    )

    numbertheory: tuple[str, ...] = (
        "709",
        "461",
        "466",
        "257",
        "34",
        "780",
        "233",
        "764",
        "345",
        "227",
    )


def run_experiment(dataset, fewshot_ids, log_dir, **kwargs):
    """Runs the experiment for a given dataset"""
    messages = make_fewshot_prompt(dataset, fewshot_ids)

    for question in dataset:
        if question.question_number in fewshot_ids:
            continue

        fname = Path(log_dir) / f"{question.question_number}.json"
        if fname.exists():
            continue

        try:
            response = get_question_response(question, messages, **kwargs)
            data = response.dict()
            data["metadata"] = asdict(question)
            with open(fname, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error processing {question.question_number}: {e}")


def get_question_response(question, messages, **kwargs):
    """Gets the response for a given question."""
    _messages = messages + [prompt.get_natural_language_message(question)]
    return call_openai(_messages, **kwargs)


def make_fewshot_prompt(dataset, question_ids):
    """Makes a few-shot prompt for a given dataset and question ids."""
    questions = [dataset.get_question(qid) for qid in question_ids]
    messages = [system_message()]
    messages.extend(prompt.informal_to_formal_messages(questions))
    return messages


def system_message():
    return {
        "role": "system",
        "content": (
            "Translate the following natural language math problem to the "
            "Isabelle theorem proving language.  Do not provide a proof of the "
            "statement. Use dilligence when translating the problem and make "
            "certain you capture all the necessary assumptions as hypotheses."
        ),
    }


def call_openai(messages, **kwargs):
    return CLIENT.chat.completions.create(messages=messages, **kwargs)


def main():
    args = tyro.cli(Args)

    dataset = MiniF2FMATH()
    algebra = dataset.get_subject("algebra")
    algebra_ids = FEWSHOTIDS.algebra

    numtheory = dataset.get_subject("numbertheory")
    numtheory_ids = FEWSHOTIDS.numbertheory

    algebra_data = ("algebra", algebra, algebra_ids)
    numtheory_data = ("numbertheory", numtheory, numtheory_ids)

    for data in (algebra_data, numtheory_data):
        dataset_name, dataset, ids = data
        log_dir = Path(__file__).parent / "artifacts" / args.name / dataset_name
        log_dir.mkdir(parents=True, exist_ok=True)
        params = asdict(args)
        with open(log_dir / "params", "w") as f:
            json.dump(params, f)

        params.pop("name")
        run_experiment(dataset, ids, log_dir, **params)


if __name__ == "__main__":
    main()
