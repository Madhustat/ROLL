#!/usr/bin/env bash
# CPU job: unpack a user-space GPU driver (NEO + IGC + GMM) into /workspace/remote/neo_<ver>; used when NEO=<ver> is set. usage: setup_neo.sh [26.31|26.14]
set -euo pipefail
V=${1:-26.31}; R=${R:-/workspace/remote}
get() { if command -v curl >/dev/null; then curl -fsSL "$1"; else python3 -c "import sys,urllib.request; sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1], timeout=300).read())" "$1"; fi; }
case $V in
  26.31) N=26.31.39395.13; I=2.40.13; IB=22418; G=22.10.0;;
  26.14) N=26.14.37833.4; I=$(get https://api.github.com/repos/intel/compute-runtime/releases/tags/26.14.37833.4 | grep -oE 'igc-core-2_[0-9.]+\+[0-9]+' | head -1 | sed 's/igc-core-2_//'); IB=${I#*+}; I=${I%+*}; G=22.9.0;;
  *) echo "unknown version $V"; exit 1;;
esac
D=$R/neo_$V; T=$(mktemp -d); cd $T
C=https://github.com/intel/compute-runtime/releases/download/$N; IG=https://github.com/intel/intel-graphics-compiler/releases/download/v$I
for u in $IG/intel-igc-core-2_${I}+${IB}_amd64.deb $IG/intel-igc-opencl-2_${I}+${IB}_amd64.deb $C/intel-opencl-icd_${N}-0_amd64.deb $C/libze-intel-gpu1_${N}-0_amd64.deb $C/libigdgmm12_${G}_amd64.deb; do
  echo "get $(basename $u)"; get "$u" > "$(basename $u)"
done
rm -rf $D; mkdir -p $D; for f in *.deb; do dpkg-deb -x $f $D; done
ls $D/usr/lib/x86_64-linux-gnu | grep -E "libze_intel_gpu|libigc|libigdfcl|libigdgmm" | tr '\n' ' '; echo
echo "NEO $V READY in $D"
