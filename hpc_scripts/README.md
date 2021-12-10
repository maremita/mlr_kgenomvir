### How to Run mlr\_eval\_sim scripts on Slurm-based clusters

Create un new folder for the project

```sh
mkdir -p  mlr_exps
cd mlr_exps
```

Download any software if it's not not available in your cluster environment, 
such us `phylodm` in Compute Canada environment

```sh
pip download --no-deps phylodm
```

Or download and build a wheel archive

```sh
pip wheel --no-deps phylodm
```

Downlowd and clone `mlr_kgenomvir` package

```sh
if [ ! -d "mlr_kgenomvir" ] ; then
    git clone git@github.com:maremita/mlr_kgenomvir.git
fi
```

Go to `mlr_kgenomvir/hpc_scripts/` directory

```sh
cd mlr_kgenomvir/hpc_scripts/
```

Edit script `submit_mlr_exp.sh` to update modules and pip install the downloaded packages

Make a copy of the template file `eval_config_template.ini`

```sh
cp eval_config_template.ini eval_config_exp1.ini
```

Complete and update information about account and resources in `eval_config_exp1.ini`

Update the parameters of the evaluations in `eval_config_exp1.ini`


Run the script `write_run_mlr_sim_exps.py`

```sh
python write_run_mlr_sim_exps.py -c eval_config_exp1.ini -t coverages --submit

```

Once all jobs have completed successfully, you can collect and plot the results 
by submitting a new job with the command

```sh
python write_run_mlr_sim_exps.py -c eval_config_exp1.ini -t coverages --plot-only --submit
```

For more options:

```sh
python write_run_mlr_sim_exps.py --help
```
