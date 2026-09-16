from reasoning import (
    compute_relation,
    build_transitive_chain,
    generate_reasoning_example,
)


def test_compute_relation():
    assert compute_relation(10, 7) == ">"
    assert compute_relation(4, 9) == "<"
    assert compute_relation(5, 5) == "="


def test_transitive_chain():
    relations = [("A", ">", "B"), ("B", ">", "C")]
    assert build_transitive_chain(relations)[0] == ("A", ">", "C")


def test_generate_reasoning_example():
    example = generate_reasoning_example("age", seed=21)
    assert example["domain"] == "age"
    assert example["answer_relation"] in {">", "<", "="}
    assert isinstance(example["hop_count"], int)
    assert example["hop_count"] >= 1
