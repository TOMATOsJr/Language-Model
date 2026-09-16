from reasoning import (
    compute_relation,
    build_transitive_chain,
    generate_reasoning_example,
)
from validate import validate_dataset


def test_compute_relation():
    assert compute_relation(10, 7) == ">"
    assert compute_relation(4, 9) == "<"
    assert compute_relation(5, 5) == "="


def test_transitive_chain():
    relations = [("A", ">", "B"), ("B", ">", "C")]
    assert build_transitive_chain(relations)[0] == ("A", ">", "C")


def test_generate_reasoning_example():
    example = generate_reasoning_example("height", seed=11)
    assert example["domain"] == "height"
    assert example["answer_relation"] in {">", "<", "="}
    assert isinstance(example["hop_count"], int)
    assert example["hop_count"] >= 1


def test_validate_dataset_allows_same_question_with_distinct_answers():
    records = [
        {
            "id": "hi_a",
            "language": "hindi",
            "question": "रोहन और अमन में से कौन बड़ा है?",
            "answer": "रोहन",
            "entities": ["रोहन", "अमन"],
            "relations": [("रोहन", ">", "अमन")],
            "answer_relation": ">",
        },
        {
            "id": "hi_b",
            "language": "hindi",
            "question": "रोहन और अमन में से कौन बड़ा है?",
            "answer": "अमन",
            "entities": ["रोहन", "अमन"],
            "relations": [("रोहन", "<", "अमन")],
            "answer_relation": "<",
        },
    ]
    assert validate_dataset(records) == []
