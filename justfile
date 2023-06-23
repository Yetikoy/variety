set positional-arguments
set shell := ["bash", "-uc"]
 
# test provided files - or all project
test *args='test':
    pytest -rA $@
# run unit tests for given source file
infer-test file:
    just test {{replace(without_extension(file), "src", "test")}}_test.py
# watch /src and run unit tests on file changes
watch:
    watchmedo shell-command --patterns="*.py" --drop --recursive --command='bash -c "clear; just infer-test \"$watch_src_path\""' /common/src
# shell access
bash:
    bash
