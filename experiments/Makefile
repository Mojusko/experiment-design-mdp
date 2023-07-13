PYTHONPATH=pythonconda

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
${JOBS_BELS_DEN}: job-bels-adaptive-den%: ; cd beilschmiedia && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${PYTHONPATH} run_beilschmiedia.py --repeats=1 --save="results/density-$*.txt" --num_components=${NUM_COMPONENTS_BEL} --episodes=${BELS_EPISODES} --adaptive="No" --policy="density" --cores=1 --seed=$* --accuracy=0.05
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

PLOTS: ${PYTHONPATH} plot_grids.py && ${PYTHONPATH} plot_pharma.py && ${PYTHONPATH} plot.py

move:
	cp figs/*.png ../../pic/.

grids: STOCH GRID
all: STOCH GRID bels pharma
