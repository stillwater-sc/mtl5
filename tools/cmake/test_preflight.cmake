# Contract test for benchmarks/preflight.{sh,ps1} (#442).
#
# There are two implementations of one contract, and they run on disjoint sets of
# machines -- nobody executes the PowerShell one on Linux, so the two drift
# silently and a Zen 4 sidecar quietly stops being comparable with an i7 one.
# This runs whichever applies to THIS platform and holds it to the SAME key list,
# so each CI platform proves its own half. That is the only place the drift can
# be caught before a measurement depends on it.
#
# It also catches the plain failure the last PowerShell harness shipped with: a
# syntax or parameter error nobody sees until someone tries to take a
# measurement on the one machine that runs it (#437/#438).
#
# --report-only is deliberate: ctest runs *during* a build, so the competing-load
# gate would fire, and CI checkouts of a PR branch are not always clean. The
# gates themselves are exercised on real machines; what this asserts is that the
# script RUNS and emits a complete record.
#
# Expects: INTERP, SCRIPT, REPO

# Every key both implementations must emit. Adding one here without adding it to
# both scripts fails on the platform that is missing it.
set(REQUIRED_KEYS
    preflight_version preflight_host preflight_kernel
    tree_git_commit tree_git_dirty
    competing_load
    cpu_online cpu_affinity
    governor turbo power_mode
    thermal_sensor thermal_before_c thermal_limit_c thermal_headroom_c
    preflight_gates)

if(INTERP MATCHES "pwsh|powershell")
    set(_before ${INTERP} -NoProfile -NonInteractive -File ${SCRIPT} -ReportOnly -Threads 1 -Repo ${REPO})
    set(_after  ${INTERP} -NoProfile -NonInteractive -File ${SCRIPT} -Phase after)
else()
    set(_before ${INTERP} ${SCRIPT} --report-only --threads 1 --repo ${REPO})
    set(_after  ${INTERP} ${SCRIPT} --phase after)
endif()

execute_process(COMMAND ${_before}
                RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "--report-only must never fail, got ${_rc}:\n${_err}\n--- stdout ---\n${_out}")
endif()

string(REPLACE "\n" ";" _lines "${_out}")
foreach(key ${REQUIRED_KEYS})
    # Match the VALUE too. An empty value is the failure mode that matters here:
    # a probe that silently found nothing writes `governor=` into the sidecar,
    # which reads as a recorded fact and is not one. Probes that cannot answer
    # are required to say `unavailable` out loud.
    set(_found "")
    foreach(line ${_lines})
        string(STRIP "${line}" line)
        if(line MATCHES "^${key}=(.+)$")
            set(_found "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(_found STREQUAL "")
        message(FATAL_ERROR
            "preflight emitted no value for '${key}'.\n"
            "Both preflight.sh and preflight.ps1 must emit every key in "
            "REQUIRED_KEYS, and must write 'unavailable' rather than nothing "
            "when the platform cannot answer.\n--- stdout ---\n${_out}")
    endif()
endforeach()

# The dirty-check exclusion must stay contained to generated output. An
# unrestricted --ignore-path is a hole straight through the gate: point it at a
# source directory and edits there stop counting, so preflight reports a clean
# tree and the sidecar records tree_git_dirty=0 for a build whose source had
# changed. Asking for `benchmarks` (source) must NOT produce an exclusion.
if(INTERP MATCHES "pwsh|powershell")
    set(_bad_ignore ${INTERP} -NoProfile -NonInteractive -File ${SCRIPT}
                    -ReportOnly -Threads 1 -Repo ${REPO} -IgnorePath ${REPO}/benchmarks)
else()
    set(_bad_ignore ${INTERP} ${SCRIPT} --report-only --threads 1 --repo ${REPO}
                    --ignore-path ${REPO}/benchmarks)
endif()
execute_process(COMMAND ${_bad_ignore}
                RESULT_VARIABLE _rc3 OUTPUT_VARIABLE _out3 ERROR_VARIABLE _err3)
if(NOT _rc3 EQUAL 0)
    message(FATAL_ERROR "--report-only must never fail, got ${_rc3}:\n${_err3}")
endif()
if(_out3 MATCHES "tree_dirty_excluded=")
    message(FATAL_ERROR
        "preflight excluded a SOURCE directory from the dirty check.\n"
        "Only benchmarks/data/<machine> may be excluded; anything else lets a "
        "modified source tree be recorded as clean.\n--- stdout ---\n${_out3}")
endif()

# The after phase is what makes a throttled session visible, and it is the phase
# nobody runs by hand.
execute_process(COMMAND ${_after}
                RESULT_VARIABLE _rc2 OUTPUT_VARIABLE _out2 ERROR_VARIABLE _err2)
if(NOT _rc2 EQUAL 0)
    message(FATAL_ERROR "--phase after failed (${_rc2}):\n${_err2}")
endif()
if(NOT _out2 MATCHES "thermal_after_c=[^ \t\r\n]+")
    message(FATAL_ERROR "--phase after must emit thermal_after_c, got:\n${_out2}")
endif()

message(STATUS "preflight contract: all keys present (${SCRIPT})")
