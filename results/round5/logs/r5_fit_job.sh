set -u
JOBDIR=$PWD; KIT=$JOBDIR/r5_fit_kit.tgz; DEST=$HOME/durian_r5/fitkit
echo "kit_md5 $(md5sum $KIT | cut -d' ' -f1)"
mkdir -p tmpkit && tar xzf $KIT -C tmpkit src/foundation/round5.py
KMD5=$(md5sum tmpkit/src/foundation/round5.py | cut -d' ' -f1); echo "round5.py md5 in kit: $KMD5"
if [ -f $DEST/src/foundation/round5.py ]; then
  HMD5=$(md5sum $DEST/src/foundation/round5.py | cut -d' ' -f1); echo "existing fitkit round5.py md5: $HMD5"
  if [ "$HMD5" != "$KMD5" ]; then echo "MISMATCH: existing fitkit differs from kit; aborting"; exit 3; fi
  echo "reusing existing fitkit"
else
  mkdir -p $DEST && tar xzf $KIT -C $DEST && echo "unpacked kit to $DEST"
  echo "unpacked round5.py md5: $(md5sum $DEST/src/foundation/round5.py | cut -d' ' -f1)"
fi
source ~/miniconda3/etc/profile.d/conda.sh && conda activate durian-fm
export HF_HOME=~/durian_r5/hf DURIAN_DATA_ROOT=~/durian_r5/zips NUMBA_CACHE_DIR=~/durian_r5/numba_cache CUDA_VISIBLE_DEVICES=""
python -c "import sys,sklearn,numpy,pandas,scipy,pyarrow;print('python',sys.version.split()[0]);print('scikit-learn',sklearn.__version__);print('numpy',numpy.__version__);print('pandas',pandas.__version__);print('scipy',scipy.__version__);print('pyarrow',pyarrow.__version__)" | tee $JOBDIR/versions.txt
python -c "import numpy; numpy.show_config()" 2>/dev/null | grep -iA3 "blas" | head -12 >> $JOBDIR/versions.txt || true
cd $DEST
TIMER=""; [ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"
for MODE in fit diagnose; do
  echo "=== $MODE start $(date -Is)"
  T0=$(date +%s); RC=0
  R5_EMB=$HOME/durian_r5/emb ROUND5_DIR=$HOME/durian_r5/fitkit/results/round5 OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore DURIAN_INTERMEDIATE=intermediate_common \
    nice -n 10 $TIMER python src/foundation/round5.py $MODE > $JOBDIR/round5_$MODE.log 2>&1 || RC=$?
  T1=$(date +%s)
  echo "=== $MODE exit $RC wall $((T1-T0))s end $(date -Is)" | tee -a $JOBDIR/round5_$MODE.log
  if [ $RC -ne 0 ]; then tail -40 $JOBDIR/round5_$MODE.log; exit $RC; fi
done
cp results/round5/probe_sessions.parquet results/round5/batch_decodability.csv $JOBDIR/
md5sum results/round5/probe_sessions.parquet results/round5/batch_decodability.csv
wc -l < results/round5/batch_decodability.csv
ls -l $JOBDIR/*.parquet $JOBDIR/*.csv $JOBDIR/*.log || true
