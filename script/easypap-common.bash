#!/usr/bin/env bash

# Search in build/ first, then in bin/
find_easypap()
{
    if [[ -f ${EASYPAPDIR}/build/bin/easypap ]]; then
        EASYPAP=${EASYPAPDIR}/build/bin/easypap
    elif [[ -f ${EASYPAPDIR}/bin/easypap ]]; then
        EASYPAP=${EASYPAPDIR}/bin/easypap
    else
        echo "Error: easypap binary not found" >&2
        exit 1
    fi
}

# Search in build/ first, then in bin/
find_easyview()
{
    if [[ -f ${EASYPAPDIR}/build/bin/easyview ]]; then
        EASYVIEW=${EASYPAPDIR}/build/bin/easyview
    elif [[ -f ${EASYPAPDIR}/lib/ezm/view/bin/easyview ]]; then
        EASYVIEW=${EASYPAPDIR}/lib/ezm/view/bin/easyview
    else
        echo "Error: easyview binary not found" >&2
        exit 1
    fi
}

if [[ -z ${EASYPAPDIR+x} ]]; then
    echo "Error: EASYPAPDIR variable is not defined (easypap-common.bash)" >&2
fi

TRACEDIR=${TRACEDIR:-${EASYPAPDIR}/data/traces}
CUR_TRACEFILE=ezv_trace_current.evt
PREV_TRACEFILE=ezv_trace_previous.evt

HASHDIR=${EASYPAPDIR}/data/hash
DUMPDIR=${EASYPAPDIR}/data/dump
PERFDIR=${EASYPAPDIR}/data/perf
