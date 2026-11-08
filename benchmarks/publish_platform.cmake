# Publish one mtl5_bench platform session (#721, #737, #758).
#
#   cmake -DRESULTS=<results dir> -DMACHINE=<systems.md name> -P benchmarks/publish_platform.cmake
#
# RESULTS is the directory platform.sh / platform.ps1 wrote (by default
# build-platform/results/<host>-<UTC time>); MACHINE is the machine's name in
# docs/benchmarks/systems.md, lower case (i7-12700k, ryzen-9-8945hs, ...).
#
# What it does, checking before it writes anything:
#   1. Validates the session: every <variant>-t<T>.json is schema mtl5-bench/1
#      and has its .json.sysinfo sidecar and its .md card; summary.md exists;
#      every run is the same commit and class; no run has a failed check and
#      none was built from a dirty tree (a FAIL blocks a card, #737).
#   2. Copies the session -- JSON, sidecars, cards, summary.md and the
#      preflight records -- to benchmarks/data/<MACHINE>/mtl5_bench/<session>/,
#      where <session> is YYYYMMDDTHHMMSSZ: the UTC start in the results
#      directory's default name, else summary.md's finish time. The data
#      directory is append-only: a session already there is left alone if its
#      files are identical, and refused if they differ. Every session ever
#      published stays, which is what stillwater-sc/mtl5-python#121 ingests.
#   3. Writes docs/benchmarks/<MACHINE>-mtl5-bench.md, the machine's card page:
#      a provenance table (session, commit, class, runs, CPUs, clock policy,
#      the raw data, earlier sessions) and the session's cards. The page shows
#      the LATEST published session; publishing an older one leaves it alone.
#      Table cells are repaired if the cards came from an mtl5_bench that did
#      not escape `|` (before #758).
#   4. Registers the page in docs-site/sync-content.mjs's FILE_MAP if it is
#      not there, and warns if systems.md does not link it.
#
# Options (-D before -P):
#   PRELIMINARY=ON     mark the page preliminary: measured outside #737's
#                      methodology (unpinned cores, clock policy, one session).
#   NOTE="text"        a sentence for the page, after the preliminary mark.
#   ALLOW_DIRTY=ON     accept runs built from a dirty tree (the page says so).
#   ALLOW_FAIL=ON      accept runs with failed checks (the page says so).
#   SESSION=<stamp>    the session name, if neither the directory name nor
#                      summary.md carries a time.
#   DRY_RUN=ON         validate and report what would be written; write nothing.
#
# Then review and commit the data directory, the page and sync-content.mjs; the
# script prints the paths.

cmake_minimum_required(VERSION 3.22)

get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(fail msg)
    message(FATAL_ERROR "publish_platform: ${msg}")
endfunction()
function(note msg)
    message(STATUS "publish_platform: ${msg}")
endfunction()

# -- arguments -------------------------------------------------------------------
if(NOT DEFINED RESULTS OR NOT DEFINED MACHINE)
    fail("usage: cmake -DRESULTS=<results dir> -DMACHINE=<systems.md name> -P ${CMAKE_CURRENT_LIST_FILE}")
endif()
get_filename_component(RESULTS "${RESULTS}" ABSOLUTE)
if(NOT IS_DIRECTORY "${RESULTS}")
    fail("RESULTS is not a directory: ${RESULTS}")
endif()
if(NOT MACHINE MATCHES "^[a-z0-9][a-z0-9.-]*$")
    fail("MACHINE must be the systems.md name in lower case (letters, digits, '.', '-'): '${MACHINE}'")
endif()

set(SYSTEMS_MD "${REPO}/docs/benchmarks/systems.md")
file(READ "${SYSTEMS_MD}" systems_text)
string(TOLOWER "${systems_text}" systems_lower)
string(FIND "${systems_lower}" "`${MACHINE}`" at)
if(at EQUAL -1)
    fail("'${MACHINE}' is not a machine in docs/benchmarks/systems.md -- register it there first "
         "(the name in backticks in the Systems table)")
endif()

# -- 1. validate the session -----------------------------------------------------
file(GLOB jsons "${RESULTS}/*.json")
list(SORT jsons)
if(NOT jsons)
    fail("no <variant>-t<T>.json in ${RESULTS} -- is this a platform.sh / platform.ps1 results directory?")
endif()
if(NOT EXISTS "${RESULTS}/summary.md")
    fail("${RESULTS}/summary.md is missing -- the session did not finish")
endif()

set(commit "")
set(class "")
set(dirty OFF)
set(total_failed 0)
set(runs "")
set(governor "")
set(cpus "")
set(competing "")
set(copy_list "${RESULTS}/summary.md")

foreach(json IN LISTS jsons)
    get_filename_component(name "${json}" NAME)
    string(REGEX REPLACE "\\.json$" "" base "${json}")
    foreach(need "${json}.sysinfo" "${base}.md")
        if(NOT EXISTS "${need}")
            fail("${name}: ${need} is missing -- every run needs its sidecar and its card")
        endif()
    endforeach()
    list(APPEND copy_list "${json}" "${json}.sysinfo" "${base}.md")

    file(READ "${json}" js)
    string(JSON schema ERROR_VARIABLE err GET "${js}" schema)
    if(err OR NOT schema STREQUAL "mtl5-bench/1")
        fail("${name}: not an mtl5-bench/1 result (schema '${schema}')")
    endif()
    string(JSON label GET "${js}" label)
    string(JSON run_class GET "${js}" class)
    string(JSON n LENGTH "${js}" results)
    set(failed 0)
    if(n GREATER 0)
        math(EXPR last "${n} - 1")
        foreach(i RANGE ${last})
            string(JSON ok GET "${js}" results ${i} check ok)
            string(JSON e ERROR_VARIABLE no_error GET "${js}" results ${i} error)
            if(NOT ok OR (NOT no_error AND NOT e STREQUAL ""))
                math(EXPR failed "${failed} + 1")
            endif()
        endforeach()
    endif()
    math(EXPR total_failed "${total_failed} + ${failed}")

    # The sidecar: key=value, the build's provenance plus preflight's state.
    file(STRINGS "${json}.sysinfo" lines)
    set(sc_commit "")
    set(sc_dirty "")
    set(sc_threads "")
    foreach(line IN LISTS lines)
        if(line MATCHES "^git_commit=(.*)$")
            set(sc_commit "${CMAKE_MATCH_1}")
        elseif(line MATCHES "^git_dirty=(.*)$")
            set(sc_dirty "${CMAKE_MATCH_1}")
        elseif(line MATCHES "^pool_size=(.*)$")
            set(sc_threads "${CMAKE_MATCH_1}")
        elseif(line MATCHES "^governor=(.*)$" AND governor STREQUAL "")
            set(governor "${CMAKE_MATCH_1}")
        elseif(line MATCHES "^cpu_affinity=(.*)$" AND cpus STREQUAL "")
            set(cpus "${CMAKE_MATCH_1}")
        elseif(line MATCHES "^competing_load=(.*)$" AND competing STREQUAL "")
            set(competing "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(sc_commit STREQUAL "")
        fail("${name}.sysinfo has no git_commit -- not an mtl5_bench sidecar")
    endif()
    if(commit STREQUAL "")
        set(commit "${sc_commit}")
        set(class "${run_class}")
    elseif(NOT commit STREQUAL sc_commit OR NOT class STREQUAL run_class)
        fail("${name}: commit ${sc_commit}, class ${run_class}, but an earlier run is ${commit}, ${class} "
             "-- one session is one commit and one class")
    endif()
    if(sc_dirty STREQUAL "1")
        set(dirty ON)
    endif()
    if(sc_threads STREQUAL "1")
        set(t_word "thread")
    else()
        set(t_word "threads")
    endif()
    string(APPEND runs "${label} at ${sc_threads} ${t_word}, ")
    note("${name}: ${label}, ${sc_threads} ${t_word}, ${n} entries, ${failed} failed")
endforeach()
string(REGEX REPLACE ", $" "" runs "${runs}")

if(dirty AND NOT ALLOW_DIRTY)
    fail("the runs were built from a dirty tree (git_dirty=1), so the result cannot be reproduced. "
         "Re-run on a clean tree, or pass -DALLOW_DIRTY=ON and the page will say so.")
endif()
if(total_failed GREATER 0 AND NOT ALLOW_FAIL)
    fail("${total_failed} failed check(s): a FAIL blocks a card (#737) -- file it, fix it, re-run. "
         "-DALLOW_FAIL=ON publishes anyway and the page says so.")
endif()

foreach(rec preflight-before-build.txt preflight.txt)
    if(EXISTS "${RESULTS}/logs/${rec}")
        list(APPEND copy_list "${RESULTS}/logs/${rec}")
    endif()
endforeach()

# The session's name: when it started, from the results directory's default
# name (<host>-<UTC start>); else when it finished, from summary.md's last
# header line, which both platform scripts write at the end; else SESSION.
file(READ "${RESULTS}/summary.md" summary)
if(NOT DEFINED SESSION)
    get_filename_component(dirname "${RESULTS}" NAME)
    if(dirname MATCHES "([0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z)$")
        set(SESSION "${CMAKE_MATCH_1}")
    elseif(summary MATCHES "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])T([0-9][0-9]):([0-9][0-9]):([0-9][0-9])Z")
        set(SESSION "${CMAKE_MATCH_1}${CMAKE_MATCH_2}${CMAKE_MATCH_3}T${CMAKE_MATCH_4}${CMAKE_MATCH_5}${CMAKE_MATCH_6}Z")
    else()
        fail("cannot tell when the session ran -- pass -DSESSION=YYYYMMDDTHHMMSSZ")
    endif()
endif()
# (CMake regexes have no {n} counts, so the digits are spelled out.)
if(NOT SESSION MATCHES "^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z$")
    fail("SESSION must be YYYYMMDDTHHMMSSZ (UTC), got '${SESSION}'")
endif()

# -- 2. copy to benchmarks/data, append-only -------------------------------------
set(DATA_REL "benchmarks/data/${MACHINE}/mtl5_bench/${SESSION}")
set(DATA "${REPO}/${DATA_REL}")
set(already OFF)
if(IS_DIRECTORY "${DATA}")
    foreach(src IN LISTS copy_list)
        get_filename_component(fn "${src}" NAME)
        if(NOT EXISTS "${DATA}/${fn}")
            fail("${DATA_REL} exists but has no ${fn} -- the data directory is append-only; "
                 "inspect it rather than overwrite it")
        endif()
        file(SHA256 "${src}" h_src)
        file(SHA256 "${DATA}/${fn}" h_dst)
        if(NOT h_src STREQUAL h_dst)
            fail("${DATA_REL}/${fn} differs from ${src} -- a published session is never overwritten. "
                 "If this is a different session, pass -DSESSION=<its own stamp>.")
        endif()
    endforeach()
    set(already ON)
    note("${DATA_REL} is already published with identical files")
elseif(NOT DRY_RUN)
    file(MAKE_DIRECTORY "${DATA}")
    foreach(src IN LISTS copy_list)
        get_filename_component(fn "${src}" NAME)
        file(COPY_FILE "${src}" "${DATA}/${fn}" ONLY_IF_DIFFERENT)
    endforeach()
    list(LENGTH copy_list n_files)
    note("copied ${n_files} files to ${DATA_REL}")
endif()

# -- 3. the card page --------------------------------------------------------------
set(PAGE_REL "docs/benchmarks/${MACHINE}-mtl5-bench.md")
set(PAGE "${REPO}/${PAGE_REL}")

file(GLOB sessions LIST_DIRECTORIES true RELATIVE "${REPO}/benchmarks/data/${MACHINE}/mtl5_bench"
     "${REPO}/benchmarks/data/${MACHINE}/mtl5_bench/*")
list(APPEND sessions "${SESSION}")
list(REMOVE_DUPLICATES sessions)
list(SORT sessions)
list(GET sessions -1 latest)

if(NOT SESSION STREQUAL latest)
    note("${SESSION} is older than the latest published session ${latest}: the page is left alone")
else()
    # Cards as one markdown body: summary.md without its H1 and its
    # "Class ..., variants ..." line (both are in the provenance table), and
    # with table cells repaired where an older mtl5_bench left `|` unescaped.
    # Line by line with string(FIND), never as a CMake list: card text may hold
    # ';', which a list would split on.
    set(body "")
    set(rest "${summary}")
    set(header_cells -1)
    set(repaired 0)
    while(NOT rest STREQUAL "")
        string(FIND "${rest}" "\n" nl)
        if(nl EQUAL -1)
            set(line "${rest}")
            set(rest "")
        else()
            string(SUBSTRING "${rest}" 0 ${nl} line)
            math(EXPR from "${nl} + 1")
            string(SUBSTRING "${rest}" ${from} -1 rest)
        endif()
        string(REGEX REPLACE "\r$" "" line "${line}")
        if(line MATCHES "^# " OR line MATCHES "^Class `")
            continue()
        endif()
        if(line MATCHES "^\\|")
            string(REPLACE "\\|" "" bare "${line}")
            string(REGEX MATCHALL "[|]" pipes "${bare}")
            list(LENGTH pipes cells)
            if(header_cells EQUAL -1)
                set(header_cells ${cells})
            elseif(NOT cells EQUAL header_cells)
                # The one known source: an integer shape such as "|v| <= 15".
                string(REGEX REPLACE " [|]([A-Za-z0-9_]+)[|] " " \\\\|\\1\\\\| " fixed "${line}")
                string(REPLACE "\\|" "" bare "${fixed}")
                string(REGEX MATCHALL "[|]" pipes "${bare}")
                list(LENGTH pipes cells)
                if(NOT cells EQUAL header_cells)
                    fail("summary.md has a table row with ${cells} cells under a ${header_cells}-cell header "
                         "that this script cannot repair:\n  ${line}")
                endif()
                set(line "${fixed}")
                math(EXPR repaired "${repaired} + 1")
            endif()
        else()
            set(header_cells -1)
        endif()
        string(APPEND body "${line}\n")
    endwhile()
    string(REGEX REPLACE "^\n+" "" body "${body}")
    if(repaired GREATER 0)
        note("repaired ${repaired} table rows with unescaped '|' (cards from an mtl5_bench before #758)")
    endif()

    string(SUBSTRING "${commit}" 0 12 commit_short)
    set(commit_cell "`${commit_short}`")
    if(dirty)
        string(APPEND commit_cell " (**dirty tree** -- not reproducible from this commit)")
    endif()
    set(failed_cell "${total_failed}")
    if(total_failed GREATER 0)
        set(failed_cell "**${total_failed}** -- see the **FAIL** rows below")
    endif()
    if(cpus STREQUAL "")
        set(cpus "not recorded")
    endif()
    if(governor STREQUAL "")
        set(governor "not recorded")
    endif()
    set(earlier "")
    foreach(s IN LISTS sessions)
        if(NOT s STREQUAL SESSION)
            string(APPEND earlier "`${s}`, ")
        endif()
    endforeach()
    string(REGEX REPLACE ", $" "" earlier "${earlier}")
    if(earlier STREQUAL "")
        set(earlier "none")
    endif()

    set(page "# mtl5_bench capability card -- ${MACHINE}\n\n")
    string(APPEND page
        "<!-- Generated by benchmarks/publish_platform.cmake from ${DATA_REL}/.\n"
        "     Do not edit by hand: publish a new session instead (benchmarks/README.md). -->\n\n"
        "The platform benchmark's capability cards for the `${MACHINE}` machine described on the\n"
        "[benchmark systems](systems.md) page: every operation family, measured against this\n"
        "machine's FMA peak and memory bandwidth (#721). This page shows the latest published\n"
        "session; every session's raw results stay in the data directory.\n\n")
    if(PRELIMINARY)
        string(APPEND page
            "> **Preliminary.** Measured outside #737's methodology: cores not pinned, the machine's\n"
            "> clock policy not applied, one session, so no repeatability band.")
        if(DEFINED NOTE AND NOT NOTE STREQUAL "")
            string(APPEND page " ${NOTE}")
        endif()
        string(APPEND page "\n\n")
    elseif(DEFINED NOTE AND NOT NOTE STREQUAL "")
        string(APPEND page "${NOTE}\n\n")
    endif()
    string(APPEND page
        "| | |\n|---|---|\n"
        "| session | `${SESSION}` (UTC) |\n"
        "| commit | ${commit_cell} |\n"
        "| class | `${class}` |\n"
        "| runs | ${runs} |\n"
        "| failed checks | ${failed_cell} |\n"
        "| CPUs available | ${cpus} |\n"
        "| clock policy | `${governor}` |\n"
        "| raw data | [`${DATA_REL}/`](https://github.com/stillwater-sc/mtl5/tree/main/${DATA_REL}) |\n"
        "| earlier sessions | ${earlier} |\n\n"
        "${body}")

    if(DRY_RUN)
        note("would write ${PAGE_REL}")
    else()
        file(WRITE "${PAGE}" "${page}")
        note("wrote ${PAGE_REL}")
    endif()
endif()

# -- 4. register the page on the site ----------------------------------------------
set(SYNC "${REPO}/docs-site/sync-content.mjs")
file(READ "${SYNC}" sync)
set(key "'benchmarks/${MACHINE}-mtl5-bench.md'")
string(FIND "${sync}" "${key}" have)
if(have EQUAL -1)
    set(anchor "  // benchmarks/publish_platform.cmake (which also adds the entry here).\n")
    string(FIND "${sync}" "${anchor}" at)
    if(at EQUAL -1)
        fail("cannot find the mtl5_bench card block in docs-site/sync-content.mjs to add ${key}")
    endif()
    set(entry "  ${key}:")
    string(LENGTH "${entry}" len)
    while(len LESS 48)
        string(APPEND entry " ")
        math(EXPR len "${len} + 1")
    endwhile()
    string(APPEND entry "${key},\n")
    string(LENGTH "${anchor}" alen)
    math(EXPR insert_at "${at} + ${alen}")
    string(SUBSTRING "${sync}" 0 ${insert_at} head)
    string(SUBSTRING "${sync}" ${insert_at} -1 tail)
    if(DRY_RUN)
        note("would add ${key} to docs-site/sync-content.mjs")
    else()
        file(WRITE "${SYNC}" "${head}${entry}${tail}")
        note("added ${key} to docs-site/sync-content.mjs")
    endif()
endif()

string(FIND "${systems_text}" "(${MACHINE}-mtl5-bench.md)" linked)
if(linked EQUAL -1)
    message(WARNING "publish_platform: docs/benchmarks/systems.md does not link ${MACHINE}-mtl5-bench.md -- "
                    "add a [Card](${MACHINE}-mtl5-bench.md) cell to its Systems table")
endif()

if(NOT DRY_RUN)
    message(STATUS "publish_platform: done. Review, then commit:\n"
                   "  git add ${DATA_REL} ${PAGE_REL} docs-site/sync-content.mjs")
endif()
