set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate durian-fm
export HF_HOME=~/durian_r5/hf DURIAN_DATA_ROOT=~/durian_r5/zips NUMBA_CACHE_DIR=~/durian_r5/numba_cache
WD=$(pwd); K=~/durian_r5/fitkit; FT=~/durian_r5/ft; LOG=$FT/beats_ft.log
test "$(md5sum < r5_fit_kit.tgz | cut -d' ' -f1)" = 513a1aadc902ee7602dc57748566e8c5
if [ ! -e $K ]; then mkdir -p $K && tar xzf r5_fit_kit.tgz -C $K; fi
B=$(md5sum < $K/src/foundation/beats_ft.py | cut -d' ' -f1); E=$(md5sum < ~/durian_r5/extract_r5.py | cut -d' ' -f1)
test "$E" = 1f089bb8b17f7f20520522798b2bbe72
if [ "$B" = d5c531dc05841ad00d6c7b3e8de37001 ]; then cp $K/src/foundation/beats_ft.py $K/src/foundation/beats_ft.py.orig_tagged; cp beats_ft.py $K/src/foundation/beats_ft.py; fi
B=$(md5sum < $K/src/foundation/beats_ft.py | cut -d' ' -f1); test "$B" = 99e1112246aa7fbdd17f83fc78a761fe
B0=$(md5sum < $K/src/foundation/beats_ft.py.orig_tagged | cut -d' ' -f1); test "$B0" = d5c531dc05841ad00d6c7b3e8de37001
mkdir -p $FT; [ -e $LOG ] && mv $LOG $FT/beats_ft_attempt1_crash.log
{
echo "== A1ft BEATs LP-FT run (prereg-round5 kit) =="
echo "start $(date -Is)  host $(hostname)"
echo "kit tgz md5 513a1aadc902ee7602dc57748566e8c5; beats_ft.py tagged md5 $B0, patched md5 $B (crash fix: sys.modules registration of extract_r5); extract_r5.py md5 $E"
python -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'device',torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "cmd: R5_EXTRACT=\$HOME/durian_r5/extract_r5.py python src/foundation/beats_ft.py $K $FT lobo:Immature lobo:Mature lobo:Overmature outer:0 outer:1 outer:2 outer:3 outer:4"
diff $K/src/foundation/beats_ft.py.orig_tagged $K/src/foundation/beats_ft.py || true
} > $LOG 2>&1
cd $K
T0=$(date +%s)
set +e
R5_EXTRACT=$HOME/durian_r5/extract_r5.py python src/foundation/beats_ft.py $K $FT lobo:Immature lobo:Mature lobo:Overmature outer:0 outer:1 outer:2 outer:3 outer:4 >> $LOG 2>&1 &
PID=$!
( while kill -0 $PID 2>/dev/null; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$PID '$1==p{print $2}'; sleep 5; done ) > $FT/gpumem_samples_MiB.txt 2>/dev/null &
SAMP=$!
wait $PID; RC=$?
wait $SAMP
T1=$(date +%s)
PEAK=$(sort -n $FT/gpumem_samples_MiB.txt | tail -1)
{
echo "exit code $RC"
echo "wall time $((T1-T0)) s (python process incl. decode)"
echo "peak GPU memory of the beats_ft process (nvidia-smi per-process used_memory, sampled every 5 s): ${PEAK:-NA} MiB over $(wc -l < $FT/gpumem_samples_MiB.txt) samples"
echo "end $(date -Is)"
} >> $LOG
set -e
cd $FT && ls -l beats_ft_*.npy >> $LOG 2>&1 || true
tar czf $FT/r5_beats_ft.tgz beats_ft_*.npy 2>/dev/null || true
cd "$WD"
cp $LOG $FT/r5_beats_ft.tgz $FT/gpumem_samples_MiB.txt ./ 2>/dev/null || true
md5sum $FT/beats_ft_*.npy $FT/r5_beats_ft.tgz >> beats_ft_md5.txt 2>/dev/null || true
ls -lh ./*.tgz ./*.log ./*.txt || true
exit $RC
