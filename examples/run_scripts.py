from sim21.cli.CommandInterface import run
import sys
import os
import pathlib
from tabulate import tabulate
import time

# Go through each directory
for dir_type in ['flowsheets', 'passed', 'almost']:

    source_dir = os.path.join(os.getcwd(), 'scripts', dir_type)
    all_scripts = list(pathlib.Path(source_dir).glob('*.tst'))
    old_stdout, old_stderr = sys.stdout, sys.stderr
    script_summary = []
    total_run_time = 0

    for inp_script in all_scripts:
        # Open up the input/output file
        out_fname = str(inp_script).rsplit('.', 1)[0] + '.out'
        inp_file, out_file = open(inp_script, 'rt'), open(out_fname, 'wt')

        # Redirect stdout/stderr
        sys.stdout, sys.stderr = out_file, out_file

        # Get orig. working directory
        orig_path = os.getcwd()
        # Get working directory of input file
        containing_path = os.path.dirname(os.path.realpath(inp_file.name))
        # Change to that directory
        os.chdir(containing_path)

        # Run and time the execution
        script_run_time = time.time()
        run(inp_file, out_file, out_file)
        script_run_time = time.time() - script_run_time
        total_run_time += script_run_time

        # Restore working directory
        os.chdir(orig_path)

        # Restore stdout/stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Print solved cases
        print('Solved', inp_script, )
        script_summary.append([os.path.basename(os.path.normpath(inp_script)), "{:.4f}".format(script_run_time)])

    summary_file = open(os.path.join(os.getcwd(), 'scripts', dir_type, 'summary.out'), 'wt')
    script_summary.append(['', ''])
    script_summary.append(['TOTAL', "{:.4f}".format(total_run_time)])
    summary_file.write(tabulate(script_summary, ['File', 'Time(secs)'], tablefmt="pretty", colalign=("left",)))

