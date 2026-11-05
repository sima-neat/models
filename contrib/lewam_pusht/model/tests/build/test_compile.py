"""Standard package structure, not final-ELF precision or numerical qualification.

The candidate compiler remains fail-closed until those independent gates exist.
This test must never stand in for them.
"""

from model_registry.artifact import validate_compile_outputs
from model_registry.dependencies import resolve_model_compiler


def test_all_four_standard_components(compiled, build_profile, model_dir, load_script):
    compiler = load_script("compile")
    interface = load_script("validation")
    report = validate_compile_outputs(
        compiled / "compile-outputs.json",
        compiled,
        model_id="lewam_pusht",
        build_profile_id=build_profile,
        compile_invocation_id=build_profile,
        component_ids=["front", "body", "head", "policy"],
    )
    assert len(report["outputs"]) == 4
    version = resolve_model_compiler(model_dir.parents[1])["version"]
    for row in report["outputs"]:
        component = row["component_id"]
        input_shape, output_shape = interface.SHAPES[component]
        proof = compiler.audit_archive(
            compiled / row["mpk_path"],
            compiler.component_name(build_profile, component),
            version,
            {"input": interface.INPUT_NAMES[component], "input_shape": input_shape, "output_shape": output_shape},
        )
        assert proof["mpk_sha256"] == row["mpk_sha256"]
