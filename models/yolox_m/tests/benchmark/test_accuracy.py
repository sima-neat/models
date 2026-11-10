from model_registry.model_tests import measured_shortfalls


def test_accuracy_meets_published_expectations(accuracy, model_dir, build_profile):
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
