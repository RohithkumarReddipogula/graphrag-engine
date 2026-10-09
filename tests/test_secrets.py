from graphrag.config import ROOT, Settings


def test_secrets_are_hidden_in_repr():
    s = Settings(_env_file=None, openrouter_api_key="sk-very-secret", neo4j_password="pw-very-secret")
    assert "sk-very-secret" not in repr(s)
    assert "pw-very-secret" not in repr(s)


def test_blank_secrets_count_as_missing():
    s = Settings(_env_file=None, openrouter_api_key="   ", neo4j_password="")
    assert s.openrouter_api_key is None and s.neo4j_password is None


def test_env_files_are_gitignored():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines and ".env.*" in lines
