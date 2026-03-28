import ast
from pathlib import Path


def _has_explicit_seed(call_node: ast.Call) -> bool:
    if call_node.args:
        first_arg = call_node.args[0]
        if isinstance(first_arg, ast.Constant) and first_arg.value is None:
            return False
        return True
    for keyword in call_node.keywords:
        if keyword.arg == "seed":
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is None:
                return False
            return True
    return False


def _is_numpy_random_call(call_node: ast.Call, numpy_aliases: set[str]) -> bool:
    func = call_node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id in numpy_aliases
        and func.value.attr == "random"
    )


def test_examples_do_not_use_unseeded_randomness():
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    disallowed_calls: list[str] = []

    for example_file in sorted(examples_dir.rglob("*.py")):
        source = example_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(example_file))

        numpy_aliases: set[str] = set()
        numpy_random_imports: dict[str, str] = {}
        random_module_aliases: set[str] = set()
        random_imported_names: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "numpy":
                        numpy_aliases.add(alias.asname or alias.name)
                    if alias.name == "random":
                        random_module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "numpy.random":
                    for alias in node.names:
                        imported_as = alias.asname or alias.name
                        numpy_random_imports[imported_as] = alias.name
                if node.module == "random":
                    for alias in node.names:
                        random_imported_names.add(alias.asname or alias.name)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if _is_numpy_random_call(node, numpy_aliases):
                random_fn_name = node.func.attr
                if random_fn_name == "default_rng":
                    if not _has_explicit_seed(node):
                        disallowed_calls.append(f"{example_file}:{node.lineno} np.random.default_rng called without seed")
                elif random_fn_name != "seed":
                    disallowed_calls.append(f"{example_file}:{node.lineno} np.random.{random_fn_name}")
                continue

            if isinstance(node.func, ast.Name) and node.func.id in numpy_random_imports:
                imported_name = numpy_random_imports[node.func.id]
                if imported_name == "default_rng":
                    if not _has_explicit_seed(node):
                        disallowed_calls.append(f"{example_file}:{node.lineno} default_rng called without seed")
                elif imported_name != "seed":
                    disallowed_calls.append(f"{example_file}:{node.lineno} numpy.random.{imported_name}")
                continue

            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in random_module_aliases:
                    disallowed_calls.append(f"{example_file}:{node.lineno} random.{node.func.attr}")
                    continue

            if isinstance(node.func, ast.Name) and node.func.id in random_imported_names:
                disallowed_calls.append(f"{example_file}:{node.lineno} random.{node.func.id}")

    assert not disallowed_calls, "Unseeded or disallowed randomness in examples:\n" + "\n".join(disallowed_calls)
