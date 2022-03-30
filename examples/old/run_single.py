import os
from sim21.old.cli import CommandInterface


def run_file(source):
    orig_path = os.getcwd()
    containing_path = os.path.dirname(os.path.realpath(source.name))
    os.chdir(containing_path)
    CommandInterface.run(source)
    os.chdir(orig_path)


run_file(open('scripts/passed/seader_10_4.tst'))
