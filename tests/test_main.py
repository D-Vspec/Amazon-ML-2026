import main


def test_load_config_skips_comments_and_env_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nA=1\n  B = two words  \nC=\nD=x=y\n")
    monkeypatch.setenv("A", "from-env")
    monkeypatch.setenv("UNLISTED", "ignored")
    assert main.load_config(env) == {"A": "from-env", "B": "two words", "C": "", "D": "x=y"}


def test_main_runs_from_env_and_creates_missing_sets(tmp_path, monkeypatch, capsys):
    train = tmp_path / "ds" / "train"
    train.mkdir(parents=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    s1 = [f"S1-{i}" for i in range(50)]
    (train / "train_source1.tsv").write_text(header + "".join(f"{x}\tAcme Pvt Ltd\t1 Main St\tIndia\n" for x in s1))
    (train / "train_source2.tsv").write_text(header + "".join(f"S2-{i}\tAcme\tx\tIndia\n" for i in range(50)))
    (train / "train_source3.tsv").write_text(header + "".join(f"S3-{i}\tAcme\tx\tIndia\n" for i in range(50)))
    (train / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" +
                                                  "".join(f"S1-{i}\tS2-{i}\n" for i in range(50)))
    (tmp_path / ".env").write_text(f"DATA_DIR={tmp_path / 'ds'}\nSETS=0,1\nSPLIT=train\nNROWS=\n"
                                   "TRANSLITERATOR=anyascii\nNORMALIZER=rules\n")
    monkeypatch.chdir(tmp_path)
    main.main()
    assert (tmp_path / "data/splits/set_9/source1.tsv").exists()
    assert "writing the 10 training sets" in capsys.readouterr().out
