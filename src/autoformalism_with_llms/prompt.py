"""
Prompt utilities for the AutoFormalism with LLMS project.
Author: Mike Vaiana
"""
import re

from autoformalism_with_llms.dataset import MathQuestion


def informal_to_formal_messages(questions: list[MathQuestion]) -> list[dict]:
    """Convert the list of MathQuestions to a message string."""
    messages = []
    for question in questions:
        example = make_example(question)
        messages.append(get_natural_language_message(question))
        messages.append(get_formal_language_message(question))
    return messages


def get_natural_language_message(question: MathQuestion, role: str = "user") -> dict:
    """Convert a MathQuestion object to an OpenAI message dictionary.

    This message is the natural language message, i.e. the informal statement of a
    math problem.

    Args:
        question (MathQuestion): The MathQuestion object which contains informal
            and formal statements of a question.
        role (str, optional): The role of the speaker. Defaults to "user".

    Returns:
        dict: The message dictionary.
    """
    example = make_example(question)
    return {"role": role, "content": example["natural_question"]}


def get_formal_language_message(
    question: MathQuestion, role: str = "assistant"
) -> dict:
    """Convert a MathQuestion object to an OpenAI message dictionary.

    This message is the formal language message, i.e. the formal statement of a
    math problem in a theorem prover language.

    Args:
        question (MathQuestion): The MathQuestion object which contains informal
            and formal statements of a question.
        role (str, optional): The role of the speaker. Defaults to "assistant".

    Returns:
        dict: The message dictionary.
    """
    example = make_example(question)
    return {"role": role, "content": example["formal_question"]}


def make_example(question: MathQuestion) -> dict[str, str]:
    """Convert a MathQuestion object to a single example for translation.


    Args:
        question (MathQuestion): The MathQuestion object which contains informal
            and formal statements of a question.

    Returns:
        dict[str, str]: A dictionary containing the natural language question and
            the formal question. This can be used to contrust few shot learning
            examples for the translation task.
    """
    question_prompt = question_with_answer_prompt(
        question.informal_statement, question.informal_solution
    )
    theorem_prompt = remove_content_after_theorem_shows(question.formal_statement)
    theorem_prompt = remove_content_before_theorem(theorem_prompt)
    theorem_prompt = remove_theorem_name(theorem_prompt)
    theorem_prompt = theorem_prompt.strip()

    return {
        "natural_question": question_prompt,
        "formal_question": theorem_prompt,
    }


def make_question(question: MathQuestion) -> str:
    """Convert a MathQuestion object to a question string."""
    return make_example(question)["natural_question"]


def question_with_answer_prompt(question: str, solution: str) -> str:
    r"""Convert the question and solution strings to a natural language string.

    Args:
        question (str): The question string.
        solution (str): The solution string.

    Returns:
        str: The natural language string.

    """
    final_answer = get_boxed_answer(solution)
    return f"{question} The final answer is ${final_answer}$."


def remove_content_after_theorem_shows(formal_statement: str) -> str:
    """Remove the content after the shows statement in the theorem.

    Note:
        This is not applicable to metamath or hollight datasets.
    """
    for line_number, line in enumerate(formal_statement.splitlines()):
        if re.search(r"^\s*shows", line):
            return "\n".join(formal_statement.splitlines()[: line_number + 1])
    return formal_statement


def remove_content_before_theorem(formal_statement: str) -> str:
    """Removes all the content before the theorem statement."""
    for line_number, line in enumerate(formal_statement.splitlines()):
        if re.search(r"^\s*theorem", line):
            return "\n".join(formal_statement.splitlines()[line_number:])
    return formal_statement


def remove_theorem_name(formal_statement: str) -> str:
    """Removes the theorem name from the formal statement."""
    return re.sub(r"(.*theorem).*(?:|$)", r"\1", formal_statement, re.M)


def get_boxed_answer(question: str) -> str | None:
    r"""Extract the boxed answer from the string.

    We assume the question has a latex boxed answer in the form `\boxed{answer}`.

    Args:
        question (str): The question string.

    Returns:
        str: The boxed answer string.

    """
    phrase = r"\boxed{"
    try:
        index = question.index(phrase) + len(phrase) 
    except ValueError:
        return None
    open_count = 1 # since we start after \boxed{ we have one open brace
    close_count = 0
    end_index = None
    for i, c in enumerate(question[index:]):
        if c == "{":
            open_count += 1
        elif c == "}":
            close_count += 1
        if open_count == close_count:
            end_index = i
            break
    if end_index is None:
        return None
    return question[index: index + end_index]
