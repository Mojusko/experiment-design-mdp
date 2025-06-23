PYTHONPATH=python
TIMESTAMP := $(shell date +"%Y-%m-%d-%H-%M")
export TIMESTAMP

###########################
###########################
##  BANDITS
###########################
###########################


bandits_reward:
	cd bandits && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_bandits.py --repeats=5 --save="test.txt" --type=REWARD --cores=1 --seed=1
bandits_best_arm_linear:
	cd bandits && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_bandits.py --repeats=5 --save="test.txt" --type=BEST_ARM_LINEAR --cores=1 --seed=1

###########################
###########################
##  BELS
###########################
###########################


REPEATS_BELS=20
NUMBERS_BELS := $(shell seq 1 ${REPEATS_BELS})
BELS_EPISODES=128
ADA_COMPONENTS_BEL:=50
NUM_COMPONENTS_BEL:=500

JOBS_BELS_ADAOPT := $(addprefix job-bels-adaptive-opt,${NUMBERS_BELS})
JOBS_BELS_ADAAPP := $(addprefix job-bels-adaptive-app,${NUMBERS_BELS})
JOBS_BELS_DEN := $(addprefix job-bels-adaptive-den,${NUMBERS_BELS})
JOBS_BELS_MIX := $(addprefix job-bels-adaptive-mix,${NUMBERS_BELS})
JOBS_BELS_RANDOM := $(addprefix job-bels-adaptive-ran,${NUMBERS_BELS})
JOBS_BELS_TRACK := $(addprefix job-bels-adaptive-track,${NUMBERS_BELS})

${JOBS_BELS_ADAOPT}: job-bels-adaptive-opt%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/adaptive-$*.txt" --num_components=${ADA_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$*
${JOBS_BELS_ADAAPP}: job-bels-adaptive-app%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/adaptive-1-$*.txt" --num_components=1 --episodes=${BELS_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=0.05
${JOBS_BELS_DEN}: job-bels-adaptive-den%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/density-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_#EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.05
${JOBS_BELS_MIX}: job-bels-adaptive-mix%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/mixture-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.05
${JOBS_BELS_TRACK}: job-bels-adaptive-track%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/tracking-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=0.05 --linesearch=None
${JOBS_BELS_RANDOM}: job-bels-adaptive-ran%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/random-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.05 --random="true"

bels_static:  ${JOBS_BELS_DEN} ${JOBS_BELS_MIX} ${JOBS_BELS_RANDOM} ${JOBS_BELS_TRACK}
bels_ada: bels_ada_approx bels_ada_opt
bels_ada_approx: ${JOBS_BELS_ADAAPP}
bels_ada_opt: ${JOBS_BELS_ADAOPT}
bels_track: ${JOBS_BELS_TRACK}
bels_opt:
	cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/none-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=0.05
bels: bels_ada bels_opt bels_static



JOBS_BELS_ADAOPT_UN := $(addprefix job-bels-adaptive-opt-un,${NUMBERS_BELS})
JOBS_BELS_ADAAPP_UN := $(addprefix job-bels-adaptive-app-un,${NUMBERS_BELS})
JOBS_BELS_DEN_UN := $(addprefix job-bels-adaptive-den-un,${NUMBERS_BELS})
JOBS_BELS_MIX_UN := $(addprefix job-bels-adaptive-mix-un,${NUMBERS_BELS})
JOBS_BELS_RANDOM_UN := $(addprefix job-bels-adaptive-ran-un,${NUMBERS_BELS})
JOBS_BELS_TRACK_UN := $(addprefix job-bels-adaptive-track-un,${NUMBERS_BELS})

${JOBS_BELS_ADAOPT_UN}: job-bels-adaptive-opt-un%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/adaptive-un-$*.txt" --num_components=${ADA_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --uncertain="true"
${JOBS_BELS_ADAAPP_UN}: job-bels-adaptive-app-un%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/adaptive-un-1-$*.txt" --num_components=1 --episodes=${BELS_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=0.05 --uncertain="true"
${JOBS_BELS_DEN_UN}: job-bels-adaptive-den-un%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/density-un-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.05 --uncertain="true"
${JOBS_BELS_MIX_UN}: job-bels-adaptive-mix-un%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/mixture-un-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.05 --uncertain="true" --linesearch=None
${JOBS_BELS_RANDOM_UN}: job-bels-adaptive-ran-un%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/random-un-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.05 --uncertain="true" --random="true"
${JOBS_BELS_TRACK_UN}: job-bels-adaptive-track-un%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/tracking-un-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=0.05 --uncertain="true"

bels_static_un:  ${JOBS_BELS_DEN_UN} ${JOBS_BELS_MIX_UN} ${JOBS_BELS_RANDOM_UN} ${JOBS_BELS_TRACK_UN}
bels_ada_un: bels_ada_approx_un bels_ada_opt_un
bels_ada_approx_un: ${JOBS_BELS_ADAAPP_UN}
bels_ada_opt_un: ${JOBS_BELS_ADAOPT_UN}
bels_track_un: ${JOBS_BELS_TRACK_UN}
bels_random_un: ${JOBS_BELS_RANDOM_UN}
bels_opt_un:
	cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/opt-un-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=0.05 --uncertain="true"
bels_un: bels_ada_un bels_opt_un bels_static_un


###########################
###########################
##  PHARMA
###########################
###########################

REPEATS_PHARMA=20
NUMBERS_PHARMA := $(shell seq 1 ${REPEATS_PHARMA})
PHARMA_EPISODES=128

JOBS_PHARMA_ADAOPT := $(addprefix job-pharma-adaptive-opt-un,${NUMBERS_PHARMA})
JOBS_PHARMA_ADAAPP := $(addprefix job-pharma-adaptive-app-un,${NUMBERS_PHARMA})
JOBS_PHARMA_DEN := $(addprefix job-pharma-adaptive-den-un,${NUMBERS_PHARMA})
JOBS_PHARMA_MIX := $(addprefix job-pharma-adaptive-mix-un,${NUMBERS_PHARMA})
JOBS_PHARMA_RANDOM := $(addprefix job-pharma-adaptive-ran-un,${NUMBERS_PHARMA})
JOBS_PHARMA_TRACK := $(addprefix job-pharma-adaptive-track-un,${NUMBERS_PHARMA})


${JOBS_PHARMA_ADAOPT}: job-pharma-adaptive-opt-un%: ; cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/adaptive-un-$*.txt" --num_components=10 --episodes=${PHARMA_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=1e-7
${JOBS_PHARMA_ADAAPP}: job-pharma-adaptive-app-un%: ; cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/adaptive-un-1-$*.txt" --num_components=1 --episodes=${PHARMA_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=1e-7
${JOBS_PHARMA_DEN}: job-pharma-adaptive-den-un%: ; cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/density-un-$*.txt" --num_components=100 --episodes=${PHARMA_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=1e-7
${JOBS_PHARMA_MIX}: job-pharma-adaptive-mix-un%: ; cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/mixture-un-$*.txt" --num_components=100 --episodes=${PHARMA_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=1e-7
${JOBS_PHARMA_RANDOM}: job-pharma-adaptive-ran-un%: ; cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/random-un-$*.txt" --num_components=100 --episodes=${PHARMA_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=1e-7 --random="true"
${JOBS_PHARMA_TRACK}: job-pharma-adaptive-track-un%: ; cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/tracking-un-$*.txt" --num_components=100 --episodes=${PHARMA_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=1e-7

pharma_static:  ${JOBS_PHARMA_DEN} ${JOBS_PHARMA_RANDOM} ${JOBS_PHARMA_MIX} pharma_opt
pharma_ada: pharma_ada_approx pharma_ada_opt
pharma_ada_approx: ${JOBS_PHARMA_ADAAPP}
pharma_tracking: ${JOBS_PHARMA_TRACK}
pharma_ada_opt: ${JOBS_PHARMA_ADAOPT}
pharma_opt:
	cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/opt-un-$*.txt" --num_components=100 --episodes=${PHARMA_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=1e-7
pharma: pharma_ada pharma_opt pharma_static pharma_tracking

NUMBERS_PHARMA_CON:=$(shell seq 5 1 15)
JOBS_PHARMA_CONC := $(addprefix job-pharma-con,${NUMBERS_PHARMA_CON})
${JOBS_PHARMA_CONC}: job-pharma-con%: ; cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/plot-1-$*.txt" --num_components=10 --episodes=64 --adaptive="No" --policy="mixed" --cores=1 --seed=1 --accuracy=1e-7 --savetrajectory="data/plot_N_$*.txt" --measurements=$*

pharma_concentration: ${JOBS_PHARMA_CONC}
#	cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/plot-1-1.txt" --num_components=1 --episodes=32 --adaptive="No" --policy="density" --cores=1 --seed=1 --accuracy=1e-7 --measurements=5 --savetrajectory="data/plot_N_5.txt" &
#	cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/plot-1-2.txt" --num_components=1 --episodes=32 --adaptive="No" --policy="density" --cores=1 --seed=1 --accuracy=1e-7 --measurements=6 --savetrajectory="data/plot_N_6.txt" &
#	cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/plot-1-3.txt" --num_components=1 --episodes=32 --adaptive="No" --policy="density" --cores=1 --seed=1 --accuracy=1e-7 --measurements=7 --savetrajectory="data/plot_N_7.txt" &
#	cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/plot-1-4.txt" --num_components=1 --episodes=32 --adaptive="No" --policy="density" --cores=1 --seed=1 --accuracy=1e-7 --measurements=8 --savetrajectory="data/plot_N_8.txt" &
#	cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/plot-1-5.txt" --num_components=1 --episodes=32 --adaptive="No" --policy="density" --cores=1 --seed=1 --accuracy=1e-7 --measurements=9 --savetrajectory="data/plot_N_9.txt" &
#	cd pharmacokinetics && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_pharmaco.py --repeats=1 --save="results/plot-1-6.txt" --num_components=1 --episodes=32 --adaptive="No" --policy="density" --cores=1 --seed=1 --accuracy=1e-7 --measurements=10 --savetrajectory="data/plot_N_10.txt" &


###########################
###########################
##  SWISSFEL
###########################
###########################

REPEATS_SWISSFEL=20
NUMBERS_SWISSFEL := $(shell seq 1 ${REPEATS_SWISSFEL})

JOBS_SWISSFEL_CON := $(addprefix job-swissfel-con,${NUMBERS_SWISSFEL})
JOBS_SWISSFEL_WORST := $(addprefix job-swissfel-worst,${NUMBERS_SWISSFEL})

${JOBS_SWISSFEL_CON}: job-swissfel-con%: ; cd swissfel && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} swissfel.py --repeats=1 --save="results/con-$*.txt" --worst="No" --cores=1 --seed=$* --accuracy=1e-7
${JOBS_SWISSFEL_WORST}: job-swissfel-worst%: ; cd swissfel && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} swissfel.py --repeats=1 --save="results/worst-$*.txt" --worst="Yes" --cores=1 --seed=$* --accuracy=1e-7

swissfel_worst: ${JOBS_SWISSFEL_WORST}
swissfel_con: ${JOBS_SWISSFEL_CON}

swissfel: swissfel_worst swissfel_con


###########################
###########################
##  GRID
###########################
###########################

REPEATS_GRID=25
NUMBERS_GRID := $(shell seq 1 ${REPEATS_GRID})
GRID_EPISODES=128

JOBS_GRID_ADAOPT := $(addprefix job-GRID-adaptive-opt,${NUMBERS_GRID})
JOBS_GRID_ADAAPP := $(addprefix job-GRID-adaptive-app,${NUMBERS_GRID})
JOBS_GRID_DEN := $(addprefix job-GRID-adaptive-den,${NUMBERS_GRID})
JOBS_GRID_MIX := $(addprefix job-GRID-adaptive-mix,${NUMBERS_GRID})
JOBS_GRID_RANDOM := $(addprefix job-GRID-adaptive-ran,${NUMBERS_GRID})
JOBS_GRID_TRACK := $(addprefix job-GRID-adaptive-track,${NUMBERS_GRID})

${JOBS_GRID_ADAOPT}: job-GRID-adaptive-opt%: ; cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --repeats=1 --save="results/adaptive-$*.txt" --num_components=50 --episodes=${GRID_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$*  --accuracy=0.0001
${JOBS_GRID_ADAAPP}: job-GRID-adaptive-app%: ; cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --repeats=1 --save="results/adaptive-1-$*.txt" --num_components=1 --episodes=${GRID_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=0.0001
${JOBS_GRID_DEN}: job-GRID-adaptive-den%: ; cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --repeats=1 --save="results/density-$*.txt" --num_components=200 --episodes=${GRID_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.0001
${JOBS_GRID_MIX}: job-GRID-adaptive-mix%: ; cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --repeats=1 --save="results/mixture-$*.txt" --num_components=200 --episodes=${GRID_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.0001
${JOBS_GRID_RANDOM}: job-GRID-adaptive-ran%: ; cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --repeats=1 --save="results/random-$*.txt" --num_components=200 --episodes=${GRID_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.0001 --random="true"
${JOBS_GRID_TRACK}: job-GRID-adaptive-track%: ; cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --repeats=1 --save="results/tracking-$*.txt" --num_components=200 --episodes=${GRID_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=0.0001

GRID_static:  ${JOBS_GRID_DEN} ${JOBS_GRID_MIX} ${JOBS_GRID_RANDOM} ${JOBS_GRID_TRACK}
GRID_ada: GRID_ada_approx GRID_ada_opt
GRID_ada_approx: ${JOBS_GRID_ADAAPP}
GRID_ada_opt: ${JOBS_GRID_ADAOPT}
GRID_opt:
	cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --repeats=1 --save="results/none-$*.txt" --num_components=200 --episodes=${GRID_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=0.0001
GRID: GRID_ada GRID_opt GRID_static
GRID_plot:
	cd classical_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_classical_grid.py --episodes 10 --savetrajectory="test"
	cd classical_grid && ${PYTHONPATH} ../show_grid.py



###########################
###########################
## STOCHASTIC GRID
###########################
###########################


REPEATS_STOCH=25
PROB=0.1
NUMBERS_STOCH := $(shell seq 1 ${REPEATS_STOCH})
STOCH_EPISODES=128
COMPONENTS=200
ADA_COMPONENTS=50
JOBS_STOCH_ADAOPT1 := $(addprefix job-STOCH-adaptive-opt1,${NUMBERS_STOCH})
JOBS_STOCH_ADAAPP1 := $(addprefix job-STOCH-adaptive-app1,${NUMBERS_STOCH})
JOBS_STOCH_DEN1 := $(addprefix job-STOCH-adaptive-den1,${NUMBERS_STOCH})
JOBS_STOCH_MIX1 := $(addprefix job-STOCH-adaptive-mix1,${NUMBERS_STOCH})
JOBS_STOCH_RANDOM1 := $(addprefix job-STOCH-adaptive-ran1,${NUMBERS_STOCH})
JOBS_STOCH_TRACK1 := $(addprefix job-STOCH-adaptive-track1,${NUMBERS_STOCH})

${JOBS_STOCH_ADAOPT1}: job-STOCH-adaptive-opt1%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-prob-${PROB}-$*.txt" --num_components=${ADA_COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --probability=${PROB} --accuracy=0.001
${JOBS_STOCH_ADAAPP1}: job-STOCH-adaptive-app1%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-1-prob-${PROB}-$*.txt" --num_components=1 --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=${PROB}
${JOBS_STOCH_DEN1}: job-STOCH-adaptive-den1%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/density-prob-${PROB}-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=${PROB}
${JOBS_STOCH_MIX1}: job-STOCH-adaptive-mix1%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/mixture-prob-${PROB}-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.001 --probability=${PROB}
${JOBS_STOCH_RANDOM1}: job-STOCH-adaptive-ran1%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/random-prob-${PROB}-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.001 --probability=${PROB} --random="true"
${JOBS_STOCH_TRACK1}: job-STOCH-adaptive-track1%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/tracking-prob-${PROB}-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=0.001 --probability=${PROB}

STOCH_static1:  ${JOBS_STOCH_DEN1} ${JOBS_STOCH_MIX1} ${JOBS_STOCH_RANDOM1} ${JOBS_STOCH_TRACK1} STOCH_opt1
STOCH_ada1: STOCH_ada_approx1 STOCH_ada_opt1
STOCH_ada_approx1: ${JOBS_STOCH_ADAAPP1}
STOCH_ada_opt1: ${JOBS_STOCH_ADAOPT1}
STOCH_opt1:
	cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/none-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=0.001 --probability=${PROB}
STOCH1: STOCH_ada1 STOCH_opt1 STOCH_static1




JOBS_STOCH_ADAOPT2 := $(addprefix job-STOCH-adaptive-opt2,${NUMBERS_STOCH})
JOBS_STOCH_ADAAPP2:= $(addprefix job-STOCH-adaptive-app2,${NUMBERS_STOCH})
JOBS_STOCH_DEN2 := $(addprefix job-STOCH-adaptive-den2,${NUMBERS_STOCH})
JOBS_STOCH_MIX2 := $(addprefix job-STOCH-adaptive-mix2,${NUMBERS_STOCH})
JOBS_STOCH_RANDOM2 := $(addprefix job-STOCH-adaptive-ran2,${NUMBERS_STOCH})
JOBS_STOCH_TRACK2 := $(addprefix job-STOCH-adaptive-track2,${NUMBERS_STOCH})

${JOBS_STOCH_ADAOPT2}: job-STOCH-adaptive-opt2%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-prob-0.3-$*.txt" --num_components=${ADA_COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --probability=0.3 --accuracy=0.001
${JOBS_STOCH_ADAAPP2}: job-STOCH-adaptive-app2%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-1-prob-0.3-$*.txt" --num_components=1 --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=0.3
${JOBS_STOCH_DEN2}: job-STOCH-adaptive-den2%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/density-prob-0.3-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=0.3
${JOBS_STOCH_MIX2}: job-STOCH-adaptive-mix2%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/mixture-prob-0.3-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.001 --probability=0.3
${JOBS_STOCH_RANDOM2}: job-STOCH-adaptive-ran2%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/random-prob-0.3-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.001 --probability=0.3 --random="true"
${JOBS_STOCH_TRACK2}: job-STOCH-adaptive-track2%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/tracking-prob-0.3-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=0.001 --probability=0.3

STOCH_static2:  ${JOBS_STOCH_DEN2} ${JOBS_STOCH_MIX2} ${JOBS_STOCH_RANDOM2} ${JOBS_STOCH_TRACK2}
STOCH_ada2: STOCH_ada_approx2 STOCH_ada_opt2
STOCH_ada_approx2: ${JOBS_STOCH_ADAAPP2}
STOCH_ada_opt2: ${JOBS_STOCH_ADAOPT2}
STOCH_opt2:
	cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/none-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=0.001 --probability=0.3
STOCH2: STOCH_ada2 STOCH_opt2 STOCH_static2

JOBS_STOCH_ADAOPT3 := $(addprefix job-STOCH-adaptive-opt3,${NUMBERS_STOCH})
JOBS_STOCH_ADAAPP3:= $(addprefix job-STOCH-adaptive-app3,${NUMBERS_STOCH})
JOBS_STOCH_DEN3 := $(addprefix job-STOCH-adaptive-den3,${NUMBERS_STOCH})
JOBS_STOCH_MIX3 := $(addprefix job-STOCH-adaptive-mix3,${NUMBERS_STOCH})
JOBS_STOCH_RANDOM3 := $(addprefix job-STOCH-adaptive-ran3,${NUMBERS_STOCH})
JOBS_STOCH_TRACK3 := $(addprefix job-STOCH-adaptive-track3,${NUMBERS_STOCH})

${JOBS_STOCH_ADAOPT3}: job-STOCH-adaptive-opt3%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-prob-0.5-$*.txt" --num_components=${ADA_COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --probability=0.5 --accuracy=0.001
${JOBS_STOCH_ADAAPP3}: job-STOCH-adaptive-app3%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-1-prob-0.5-$*.txt" --num_components=1 --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=0.5
${JOBS_STOCH_DEN3}: job-STOCH-adaptive-den3%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/density-prob-0.5-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=0.5
${JOBS_STOCH_MIX3}: job-STOCH-adaptive-mix3%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/mixture-prob-0.5-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.001 --probability=0.5
${JOBS_STOCH_RANDOM3}: job-STOCH-adaptive-ran3%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/random-prob-0.5-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.001 --probability=0.5 --random="true"
${JOBS_STOCH_TRACK3}: job-STOCH-adaptive-track3%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/tracking-prob-0.5-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=0.001 --probability=0.5

STOCH_static3:  ${JOBS_STOCH_DEN3} ${JOBS_STOCH_MIX3} ${JOBS_STOCH_RANDOM3}  ${JOBS_STOCH_TRACK3}
STOCH_ada3: STOCH_ada_approx3 STOCH_ada_opt3
STOCH_ada_approx3: ${JOBS_STOCH_ADAAPP3}
STOCH_ada_opt3: ${JOBS_STOCH_ADAOPT3}
STOCH_opt3:
	cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/none-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=0.001 --probability=0.5
STOCH3: STOCH_ada3 STOCH_opt3 STOCH_static3

JOBS_STOCH_ADAOPT0 := $(addprefix job-STOCH-adaptive-opt0,${NUMBERS_STOCH})
JOBS_STOCH_ADAAPP0:= $(addprefix job-STOCH-adaptive-app0,${NUMBERS_STOCH})
JOBS_STOCH_DEN0 := $(addprefix job-STOCH-adaptive-den0,${NUMBERS_STOCH})
JOBS_STOCH_RANDOM0 := $(addprefix job-STOCH-adaptive-ran0,${NUMBERS_STOCH})
JOBS_STOCH_TRACK0 := $(addprefix job-STOCH-adaptive-track0,${NUMBERS_STOCH})

${JOBS_STOCH_ADAOPT0}: job-STOCH-adaptive-opt0%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-prob-0-$*.txt" --num_components=${ADA_COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --probability=0 --accuracy=0.001
${JOBS_STOCH_ADAAPP0}: job-STOCH-adaptive-app0%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/adaptive-1-prob-0-$*.txt" --num_components=1 --episodes=${STOCH_EPISODES} --adaptive="Bayes" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=0
${JOBS_STOCH_DEN0}: job-STOCH-adaptive-den0%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/density-prob-0-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.001 --probability=0
${JOBS_STOCH_RANDOM0}: job-STOCH-adaptive-ran0%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/random-prob-0-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="mixed" --cores=1 --seed=$* --accuracy=0.001 --probability=0 --random="true"
${JOBS_STOCH_TRACK0}: job-STOCH-adaptive-track0%: ; cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/tracking-prob-0-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="tracking" --cores=1 --seed=$* --accuracy=0.001 --probability=0

STOCH_static0:  ${JOBS_STOCH_DEN0}  ${JOBS_STOCH_RANDOM0}  ${JOBS_STOCH_TRACK0}
STOCH_ada0: STOCH_ada_approx0 STOCH_ada_opt0
STOCH_ada_approx0: ${JOBS_STOCH_ADAAPP0}
STOCH_ada_opt0: ${JOBS_STOCH_ADAOPT0}
STOCH_opt0:
	cd stochastic_grid && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_stochastic_grid.py --repeats=1 --save="results/none-$*.txt" --num_components=${COMPONENTS} --episodes=${STOCH_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=1 --opt="true" --accuracy=0.001 --probability=0
STOCH0: STOCH_ada0 STOCH_opt0 STOCH_static0

STOCH: STOCH1 STOCH2 STOCH3 STOCH0

#
llm-generate-image:
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) image_generator.py --prompt '$(PROMPT)' $(if $(OUTPUT_DIR),--output_dir '$(OUTPUT_DIR)',) $(if $(SEED),--seed $(SEED),) $(if $(IMAGE_SIZE),--image_size $(IMAGE_SIZE),) $(if $(NUM_STEPS),--num_inference_steps $(NUM_STEPS),) $(if $(GUIDANCE_PROMPT),--guidance_prompt $(GUIDANCE_PROMPT),) $(if $(GUIDANCE_ESTIMATOR),--guidance_estimator $(GUIDANCE_ESTIMATOR),) $(if $(ESTIMATOR),--estimator '$(ESTIMATOR)',)

########################### LLMS
REPEATS_LLM=20
NUMBERS_LLM := $(shell seq 1 ${REPEATS_LLM})

# Jobs for V comparison (non-adaptive design)
JOBS_LLM_V_COMP_WITHV := $(addprefix job-llm-v-comp-withV-,$(NUMBERS_LLM))
JOBS_LLM_V_COMP_NOV := $(addprefix job-llm-v-comp-noV-,$(NUMBERS_LLM))

$(JOBS_LLM_V_COMP_WITHV): job-llm-v-comp-withV-%:
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py experiment=benchmark_V algorithm=design feedback=multinomial feedback.pass_V=true feedback.adaptive_design_frequency=0 results_dir="results/feedback-V" experiment_id="design-withV-$*" seed=$*

$(JOBS_LLM_V_COMP_NOV): job-llm-v-comp-noV-%:
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py experiment=benchmark_V algorithm=design feedback=multinomial feedback.pass_V=false feedback.adaptive_design_frequency=0 results_dir="results/feedback-V" experiment_id="design-noV-$*" seed=$*

llm-V-comparison: $(JOBS_LLM_V_COMP_WITHV) $(JOBS_LLM_V_COMP_NOV)

# Jobs for additivity experiment
JOBS_LLM_ALG_ADD := $(addprefix job-llm-add-alg,${NUMBERS_LLM})
JOBS_LLM_RAN_ADD := $(addprefix job-llm-add-ran,${NUMBERS_LLM})
JOBS_LLM_DSN_ADD := $(addprefix job-llm-add-dsn,${NUMBERS_LLM})

$(JOBS_LLM_ALG_ADD): job-llm-add-alg%:
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py experiment=additivity algorithm="optim" feedback="numerical" results_dir="results/additivity" experiment_id="$*" seed=$*

$(JOBS_LLM_RAN_ADD): job-llm-add-ran%:
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py experiment=additivity algorithm="random" feedback="numerical" results_dir="results/additivity" experiment_id="$*" seed=$*

$(JOBS_LLM_DSN_ADD): job-llm-add-dsn%:
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py experiment=additivity algorithm="design" feedback="numerical" results_dir="results/additivity" experiment_id="$*" seed=$*

llm-additivity: $(JOBS_LLM_ALG_ADD) $(JOBS_LLM_RAN_ADD) $(JOBS_LLM_DSN_ADD)

# Jobs for feedback comparison
#EPISODES_LLM_FEEDBACK_COMPARISON := 10 25 50 100 # Define episode counts to iterate over
#EPISODES_LLM_FEEDBACK_COMPARISON := 10 25 50 100 200
#EPISODES_LLM_FEEDBACK_COMPARISON := 10 30 50 70 90 110
EPISODES_LLM_FEEDBACK_COMPARISON := 30 50 

# Numerical Feedback Jobs
JOBS_LLM_DSN_NUM := \
  $(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-feedback-dsn-num-ep$(episodes)-$(n)))

JOBS_LLM_RAN_NUM := \
  $(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-feedback-ran-num-ep$(episodes)-$(n)))

define feedback_dsn_num_job_template
job-llm-feedback-dsn-num-ep$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=feedback_comparison \
	  algorithm=design \
	  feedback=numerical \
	  experiment.episodes=$(1) \
	  seed=$(2) \
	  results_dir="results/feedback-comparison" \
	  experiment_id="ep$(1)-$(2)"
endef

define feedback_ran_num_job_template
job-llm-feedback-ran-num-ep$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=feedback_comparison \
	  algorithm=random \
	  feedback=numerical \
	  experiment.episodes=$(1) \
	  seed=$(2) \
	  results_dir="results/feedback-comparison" \
	  experiment_id="ep$(1)-$(2)"
endef

$(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call feedback_dsn_num_job_template,$(episodes),$(n)))))

$(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call feedback_ran_num_job_template,$(episodes),$(n)))))

# Multinomial Feedback Jobs
JOBS_LLM_DSN_MULT := \
  $(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-feedback-dsn-mult-ep$(episodes)-$(n)))

JOBS_LLM_RAN_MULT := \
  $(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-feedback-ran-mult-ep$(episodes)-$(n)))

define feedback_dsn_mult_job_template
job-llm-feedback-dsn-mult-ep$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=feedback_comparison \
	  algorithm=design \
	  feedback=multinomial \
	  experiment.episodes=$(1) \
	  seed=$(2) \
	  results_dir="results/feedback-comparison" \
	  experiment_id="ep$(1)-$(2)"
endef

define feedback_ran_mult_job_template
job-llm-feedback-ran-mult-ep$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=feedback_comparison \
	  algorithm=random \
	  feedback=multinomial \
	  experiment.episodes=$(1) \
	  seed=$(2) \
	  results_dir="results/feedback-comparison" \
	  experiment_id="ep$(1)-$(2)"
endef

$(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call feedback_dsn_mult_job_template,$(episodes),$(n)))))

$(foreach episodes,$(EPISODES_LLM_FEEDBACK_COMPARISON), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call feedback_ran_mult_job_template,$(episodes),$(n)))))

# Combined targets
llm-feedback-comparison-num: $(JOBS_LLM_DSN_NUM) $(JOBS_LLM_RAN_NUM)
llm-feedback-comparison-mul: $(JOBS_LLM_DSN_MULT) $(JOBS_LLM_RAN_MULT)
llm-feedback-comparison: llm-feedback-comparison-num llm-feedback-comparison-mul

# Test-only / Inference target using config_inference.yaml
# Supports three modes based on provided paths:
# 1. Load Estimator: Provide ESTIMATOR_PATH only. Loads estimator, runs testers/savers.
#    Example: make llm-test-only ESTIMATOR_PATH=path/to/estimator.pt
# 2. Estimate from Visits: Provide VISITS_PATH only. Loads visits, trains estimator using the configured feedback mechanism (e.g., PairwiseFeedback), runs testers/savers.
#    Example: make llm-test-only VISITS_PATH=path/to/visits.pkl FEEDBACK=pairwise # Ensure feedback supports training from visits
# 3. Load Estimator & Feedback: Provide ESTIMATOR_PATH and FEEDBACK_PATH. Loads estimator, loads feedback (e.g., for base prompt evaluation), runs testers/savers.
#    Example: make llm-test-only ESTIMATOR_PATH=path/to/estimator.pt FEEDBACK_PATH=path/to/feedback.json
# Any other combination of ESTIMATOR_PATH, VISITS_PATH, FEEDBACK_PATH will cause an error.
# Allows optional ALGORITHM and FEEDBACK overrides (defaults are design, numerical).
# Allows optional RESULTS_DIR override.
llm-test-only:
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py --config-name=config_inference algorithm=$(ALGORITHM) feedback=$(FEEDBACK) experiment_id="test-$(ALGORITHM)-$(FEEDBACK)-$(shell date +%s)" seed=1 test_only=true $(if $(ESTIMATOR_PATH),estimator_path="$(ESTIMATOR_PATH)",estimator_path=null) $(if $(VISITS_PATH),visits_path="$(VISITS_PATH)",visits_path=null) $(if $(FEEDBACK_PATH),feedback_path="$(FEEDBACK_PATH)",+feedback_path=null) $(if $(RESULTS_DIR),results_dir="$(RESULTS_DIR)" override_results_dir=true,)

# Target to re-evaluate an entire experiment directory with new models.
# Finds all visits*.pkl files in EXP_DIR and runs test-only mode on them.
# This is useful for testing new ground-truth models against old visit data.
# Example: make llm-re-evaluate EXP_DIR=results/feedback-comparison-2024-05-20-10-00

# Check if EXP_DIR is provided. Note: Path should be relative to the experiments/llm directory.
ifeq ($(strip $(EXP_DIR)),)
    # This block is for the llm-re-evaluate target specifically.
    # We use a dummy dependency to show the error only when this target is invoked.
    llm-re-evaluate: check-exp-dir
    check-exp-dir:
	    $(error EXP_DIR must be set, e.g., make llm-re-evaluate EXP_DIR=results/feedback-comparison-...)
else
    # Find all visit files within the experiment directory, prepending llm/ to the path
    VISITS_FILES := $(shell find llm/$(EXP_DIR) -name "visits*.pkl")

    # Define a single, timestamped output directory for this entire re-evaluation run
    NEW_RESULTS_DIR := llm/$(EXP_DIR)/additional_tests/re-evaluation-$(TIMESTAMP)

    # Generate a unique job name for each visit file
    RE_EVAL_JOBS := $(patsubst %,job-re-eval-%,$(VISITS_FILES))

    # The main target that depends on all individual jobs
    llm-re-evaluate: $(RE_EVAL_JOBS)

    # Define a template for the individual job rules
    # $(1) will be the full path to the visit file, e.g., llm/results/exp/visits.pkl
    define re_eval_job_template
job-re-eval-$(1):
	$(MAKE) llm-test-only VISITS_PATH=$(patsubst llm/%,%,$(1)) RESULTS_DIR=$(patsubst llm/%,%,$(NEW_RESULTS_DIR))
endef

    # Instantiate a rule for each visit file found
    $(foreach file,$(VISITS_FILES),$(eval $(call re_eval_job_template,$(file))))
endif


# Test different numbers of rounds for feedback optimization
ROUNDS_VALUES := 5 10 15 20

JOBS_LLM_ROUNDS := \
  $(foreach rounds,$(ROUNDS_VALUES), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-rounds-$(rounds)-$(n)))

define rounds_job_template
job-llm-rounds-$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py experiment=rounds_finding feedback=multinomial algorithm=design feedback.num_rounds=$(1) seed=$(2) results_dir="results/rounds-finding" experiment_id="$(1)-$(2)"
endef

$(foreach rounds,$(ROUNDS_VALUES), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call rounds_job_template,$(rounds),$(n)))))

llm-find-rounds: $(JOBS_LLM_ROUNDS)

# Lambda-finding
# Define the range for the single lambda parameter
#LAMBDA_VALUES := 1 15 25 50 
LAMBDA_VALUES :=  50 75 100 125 200

# Generate job names for multinomial feedback
JOBS_LLM_LAMBDA_MUL := \
  $(foreach lambda_val,$(LAMBDA_VALUES), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-lambda-mul-$(lambda_val)-$(n)))

# Generate job names for numerical feedback
JOBS_LLM_LAMBDA_NUM := \
  $(foreach lambda_val,$(LAMBDA_VALUES), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-lambda-num-$(lambda_val)-$(n)))

# Macros to define each job type
define lambda_job_template_mul
job-llm-lambda-mul-$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=lambda_finding \
	  feedback=multinomial \
	  algorithm=design \
	  feedback.lambda=$(1) \
	  seed=$(2) \
	  results_dir="results/lambda-finding" \
	  experiment_id="mul-lambda$(1)-$(2)"
endef

define lambda_job_template_num
job-llm-lambda-num-$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=lambda_finding \
	  feedback=numerical \
	  algorithm=design \
	  feedback.lambda=$(1) \
	  seed=$(2) \
	  results_dir="results/lambda-finding" \
	  experiment_id="num-lambda$(1)-$(2)"
endef

# Instantiate all jobs from the macros
$(foreach lambda_val,$(LAMBDA_VALUES), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call lambda_job_template_mul,$(lambda_val),$(n)))))

$(foreach lambda_val,$(LAMBDA_VALUES), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call lambda_job_template_num,$(lambda_val),$(n)))))

# Split targets to run multinomial and numerical separately
llm-find-lambda-mul: $(JOBS_LLM_LAMBDA_MUL)
llm-find-lambda-num: $(JOBS_LLM_LAMBDA_NUM) # Keep numerical target name consistent for now
llm-find-lambda: llm-find-lambda-mul llm-find-lambda-num


# Lambda Sparsity-finding (similar to lambda-finding, but tunes lambda_sparsity)
# Define the range for the lambda_sparsity parameter
LAMBDA_SPARSITY_VALUES := 0 0.01 0.1 0.5 1 10

# Generate job names for multinomial feedback (sparsity tuning)
JOBS_LLM_LAMBDA_SPARSITY_MUL := \
  $(foreach lambda_s_val,$(LAMBDA_SPARSITY_VALUES), \
    $(foreach n,$(NUMBERS_LLM), \
      job-llm-lambda-sparsity-mul-$(lambda_s_val)-$(n)))

# Macros to define each job type for lambda_sparsity tuning
define lambda_sparsity_job_template_mul
job-llm-lambda-sparsity-mul-$(1)-$(2):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=lambda_finding \
	  feedback=multinomial \
	  algorithm=design \
	  feedback.lambda_sparsity=$(1) \
	  seed=$(2) \
	  results_dir="results/lambda-sparsity-finding" \
	  experiment_id="muls-lambda_sparsity$(1)-$(2)"
endef

# Instantiate all lambda_sparsity jobs from the macros
$(foreach lambda_s_val,$(LAMBDA_SPARSITY_VALUES), \
  $(foreach n,$(NUMBERS_LLM), \
    $(eval $(call lambda_sparsity_job_template_mul,$(lambda_s_val),$(n)))))

# Aggregate targets for lambda_sparsity tuning
llm-find-lambda-sparsity-mul: $(JOBS_LLM_LAMBDA_SPARSITY_MUL)
llm-find-lambda-sparsity: llm-find-lambda-sparsity-mul


# Design Frequency Finding (using lambda_finding base config)
# Iterates over episodes and adaptive_design_frequency
EPISODES_DESIGN_FREQ := 50 100 
DESIGN_FREQ_VALUES := 20 25 50 

# Generate job names: ep<episodes>-df<design_freq>-<seed>
JOBS_LLM_DESIGN_FREQUENCY := \
  $(foreach episodes,$(EPISODES_DESIGN_FREQ), \
    $(foreach freq,$(DESIGN_FREQ_VALUES), \
      $(foreach n,$(NUMBERS_LLM), \
        job-llm-design-freq-ep$(episodes)-df$(freq)-$(n))))

# Macro to define each design frequency job
# Note: adaptive_estimation_frequency=0 and adaptive_estimation_start=0 are defaults in multinomial.yaml
define design_frequency_job_template
job-llm-design-freq-ep$(1)-df$(2)-$(3):
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py \
	  experiment=design_frequency_finding \
	  feedback=multinomial \
	  algorithm=design \
	  experiment.episodes=$(1) \
	  feedback.adaptive_design_frequency=$(2) \
	  seed=$(3) \
	  results_dir="results/design-frequency-finding" \
	  experiment_id="ep$(1)-df$(2)-$(3)"
endef

# Instantiate all design frequency jobs
$(foreach episodes,$(EPISODES_DESIGN_FREQ), \
  $(foreach freq,$(DESIGN_FREQ_VALUES), \
    $(foreach n,$(NUMBERS_LLM), \
      $(eval $(call design_frequency_job_template,$(episodes),$(freq),$(n))))))

# Target to run all design frequency finding jobs
llm-find-design-frequency: $(JOBS_LLM_DESIGN_FREQUENCY)

# Target for processing human feedback: trains estimator, benchmarks, and generates images.
# Requires VISITS_PATH and FEEDBACK_PATH
# Example: make llm-human-feedback VISITS_PATH=results/.../visits.pkl FEEDBACK_PATH=results/.../feedback.json
llm-human-feedback:
	@$(if $(VISITS_PATH),,$(error VISITS_PATH is required))
	@$(if $(FEEDBACK_PATH),,$(error FEEDBACK_PATH is required))
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py --config-name=config_human_feedback algorithm=$(ALGORITHM) visits_path="$(VISITS_PATH)" feedback_path="$(FEEDBACK_PATH)" $(if $(RESULTS_DIR),results_dir="$(RESULTS_DIR)" override_results_dir=true,) experiment_id="run-$(shell basename $(VISITS_PATH) .pkl)" seed=1

# Define jobs for inspecting visits using VisitsImageSaver
JOBS_LLM_INSPECT_VISITS := $(addprefix job-llm-inspect-visit-,$(NUMBERS_LLM))

# Default values for ALGORITHM and FEEDBACK. `?=` means they can be overridden from the command line.
ALGORITHM ?= design
FEEDBACK ?= multinomial

# If VISITS_PATH is provided, try to derive ALGORITHM and FEEDBACK from it.
# This will override the defaults for any target that uses VISITS_PATH.
ifneq ($(strip $(VISITS_PATH)),)
    # We use := for immediate assignment based on the command-line VISITS_PATH.
    # The existing value (from default or user override) is used as a fallback if no pattern matches.
    ifeq ($(findstring -rand-,$(VISITS_PATH)),-rand-)
        ALGORITHM := random
    else ifeq ($(findstring -dsn-,$(VISITS_PATH)),-dsn-)
        ALGORITHM := design
    else ifeq ($(findstring -opt-,$(VISITS_PATH)),-opt-)
        ALGORITHM := optim
    endif

    ifeq ($(findstring -mult-,$(VISITS_PATH)),-mult-)
        FEEDBACK := multinomial
    else ifeq ($(findstring -num-,$(VISITS_PATH)),-num-)
        FEEDBACK := numerical
    endif
endif


# Define the algorithm code for experiment ID
algorithm_code = $(if $(findstring design,$(ALGORITHM)),dsn,$(if $(findstring random,$(ALGORITHM)),rand,$(if $(findstring optim,$(ALGORITHM)),opt,unk)))

# Define the feedback code for experiment ID
feedback_code = $(if $(findstring multinomial,$(FEEDBACK)),mult,$(if $(findstring numerical,$(FEEDBACK)),num,unk))

# Use config_name=config_inspect for inspection runs
$(JOBS_LLM_INSPECT_VISITS): job-llm-inspect-visit-%:
	@$(if $(VISITS_PATH),,$(error VISITS_PATH is required, e.g., make llm-inspect-visits VISITS_PATH=results/.../visits.pkl))
	cd llm && TIMESTAMP=$(TIMESTAMP) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHONPATH) run_exp.py --config-name=config_inspect feedback=$(FEEDBACK) algorithm=$(ALGORITHM) test_only=true visits_path="$(VISITS_PATH)" estimator_path=null $(if $(RESULTS_DIR),results_dir="$(RESULTS_DIR)" override_results_dir=true,) experiment_id="inspect-$(algorithm_code)-$(feedback_code)-$*" seed=$* +experiment.repeats=$(REPEATS_LLM)

# Target for inspecting visits
llm-inspect-visits: $(JOBS_LLM_INSPECT_VISITS)
	@$(if $(VISITS_PATH),,$(error VISITS_PATH is required, e.g., make llm-inspect-visits VISITS_PATH=results/.../visits.pkl))


PLOTS:
	${PYTHONPATH} plot_grids.py && ${PYTHONPATH} plot_pharma.py && ${PYTHONPATH} plot.py

PLOTS_STOCH: 
	${PYTHONPATH} plot_grids.py

PLOTS_BELS:
	${PYTHONPATH} plot_bels.py

move:
	cp figs/*.png ../../pic/.

grids: STOCH GRID

all: STOCH GRID bels pharma
