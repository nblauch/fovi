#!/bin/bash

# allow to specify -d for debugging or -s subset for subset
debug=0
node=':nvidia_a100-sxm4-40gb'
queue='kempner'
config='fovi_alexnet'
ngpus=1
cpuspergpu=16
batchsize=512

LOG_DIR=$SLOW_DIR/saccadenet/log

# Argument parser using getopts
while getopts ":dhlc:n:b:" opt; do
    case ${opt} in
        d )
            debug=1
            ;;
        h )
            echo "Usage: $0 [-d] [-l] [-c config] [hydra_args...]"
            echo "  -d: debug mode (no wandb, no slurm, use pdb)"
            echo "  -l: large mode (use h100 queue)"
            echo "  -c: config name (default: fovi_alexnet)"
            echo "  hydra_args: arbitrary arguments to pass to hydra (e.g., logging.wandb_entity=nblauch)"
            return
            ;;
        l) # large
            node=''
            queue='kempner_h100'
            cpuspergpu=24
            ;;
        c)
            config=$OPTARG
            ;;
        n) # num gpus
            ngpus=$OPTARG
            ;;
        b )
            batchsize=$OPTARG
            ;;

    esac
done

if [ $ngpus -gt 1 ]; then
    distributed=1
    batchsize=$((batchsize / ngpus))
else
    distributed=0
fi

# Shift to skip the parsed options
shift $((OPTIND - 1))

# Output for debugging
echo "Debug: $debug"

if [ $debug -eq 1 ]; then
    wandb=0
    slurm=0
    python_command='python -m pdb'
    # python_command='python'
else
    wandb=1
    slurm=1
    python_command='python'
fi

# Build the command with config and wandb setting
COMMAND="${python_command} scripts/train.py --config-name ${config}.yaml logging.use_wandb=${wandb} training.distributed=${distributed} dist.world_size=${ngpus} dist.ngpus=${ngpus} training.batch_size=${batchsize}"

# Append any additional hydra arguments that were passed
if [ $# -gt 0 ]; then
    COMMAND="$COMMAND $@"
fi

echo $COMMAND

if [ $slurm -eq 1 ]; then
    sbatch --export="COMMAND=$COMMAND" --job-name knncnn -p ${queue} --cpus-per-task=$((12*${ngpus})) --gres=gpu${node}:${ngpus} --mem=$((128*${ngpus}))GB --time 72:00:00 --output=$LOG_DIR/%j.log scripts/run_slurm.sbatch
else
    $COMMAND
fi

echo ""

# break after first iteration if debugging
if [ $debug -eq 1 ]; then
    exit 0
fi